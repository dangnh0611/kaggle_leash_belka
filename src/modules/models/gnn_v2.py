from torch import nn
from src.modules import layers
import torch
import logging
from functools import partial
from transformers import AutoConfig, AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers import RobertaModel, RobertaConfig
from src.modules.misc import get_act_fn

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.models import GIN
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn import GINEConv, MLP

logger = logging.getLogger(__name__)


class GINE(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(mlp, **kwargs)


class GNNV2Model(nn.Module):

    def __init__(self, global_cfg):
        """
        Extension of GIN to incorporate edge information by concatenation.

        Args:
            num_layer (int): the number of GNN layers
            emb_dim (int): dimensionality of embeddings
            num_tasks (int): number of tasks in multi-task learning scenario
            drop_ratio (float): dropout rate
            JK (str): last, concat, max or sum.
            graph_pooling (str): sum, mean, max, attention, set2set
            gnn_type: gin, gcn, graphsage, gat
            
        See https://arxiv.org/abs/1810.00826
        JK-net: https://arxiv.org/abs/1806.03536
    """
        super().__init__()
        cfg = global_cfg.model
        pool_type = cfg.pool_type

        self.node_transform = layers.MLP(
            cfg.node_in_dim, [cfg.node_embed_dim, cfg.node_embed_dim],
            norm_layer=None,
            act_layer=nn.GELU,
            dropout=0.0,
            last_norm=True,
            last_activation=True,
            last_dropout=True)

        if cfg.gnn_type == 'gin':
            self.gnn = GIN(cfg.node_embed_dim,
                           cfg.node_embed_dim,
                           cfg.num_layers,
                           cfg.node_embed_dim,
                           act=cfg.act,
                           act_first=False,
                           jk=cfg.jk)
        elif cfg.gnn_type == 'gine':
            self.gnn = GINE(
                cfg.node_embed_dim,
                cfg.node_embed_dim,
                cfg.num_layers,
                cfg.node_embed_dim,
                act=cfg.act,
                act_first=False,
                jk=cfg.jk,
                train_eps=True,
                edge_dim=cfg.edge_embed_dim,
            )
        else:
            raise ValueError

        if self.gnn.supports_edge_attr:
            self.edge_transform = layers.MLP(
                cfg.edge_in_dim, [cfg.edge_embed_dim, cfg.edge_embed_dim],
                norm_layer=None,
                act_layer=nn.GELU,
                dropout=0.0,
                last_norm=True,
                last_activation=True,
                last_dropout=True)
        else:
            self.edge_transform = None

        #Different kind of graph pooling
        emb_dim = cfg.node_embed_dim
        if pool_type == "sum":
            self.pool = global_add_pool
        elif pool_type == "mean":
            self.pool = global_mean_pool
        elif pool_type == "max":
            self.pool = global_max_pool
        elif pool_type == "attention":
            if cfg.jk == "concat":
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear(emb_dim, 1))
        elif pool_type[:-1] == "set2set":
            set2set_iter = int(pool_type[-1])
            if cfg.jk == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim,
                                    set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if pool_type[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        if cfg.jk == "concat":
            head_dim = self.mult * (self.num_layer + 1) * emb_dim
        else:
            head_dim = self.mult * emb_dim

        self.head = layers.MLP(head_dim,
                               cfg.head.mlp_chans + [cfg.head.num_output],
                               norm_layer=cfg.head.norm,
                               act_layer=get_act_fn(cfg.head.act),
                               dropout=cfg.head.dropout,
                               last_norm=False,
                               last_activation=False,
                               last_dropout=False)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[
                3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            # print(x.shape, edge_index.shape, edge_attr.shape, batch.shape)
            # print(x.dtype, edge_index.dtype, edge_attr.dtype, batch.dtype)
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.node_transform(x.float())
        if self.edge_transform is not None:
            edge_attr = self.edge_transform(edge_attr.float())
        else:
            edge_attr = None
        node_representation = self.gnn(x,
                                       edge_index,
                                       edge_attr=edge_attr,
                                       batch=batch,
                                       batch_size=batch.max())
        pooled_output = self.pool(node_representation, batch)
        # print(pooled_output.shape, torch.isnan(pooled_output).any())
        logits = self.head(pooled_output)
        return logits
