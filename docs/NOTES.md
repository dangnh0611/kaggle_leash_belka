# kaggle_leash_belka
Kaggle Competition: Leash Bio - Predict New Medicines with BELKA - https://www.kaggle.com/competitions/leash-BELKA


# EDA
- There are imbalance in molecule/scaffold. 99% of scaffold <= 187 mols, 90% <= 16, 75% <= 4
- Top scaffold by frequency: 0.15% of scaffolds take up 38.64% molecules
- There are some BB1s which significantly affect eSH binds 


# External Data
- https://github.com/seyonechithrananda/deep-bind-tensorflow
- BigBind: https://github.com/molecularmodelinglab/bigbind


# Notes
- https://www.kaggle.com/competitions/leash-BELKA/discussion/491431
    + For context, medicinal chemists that use DEL hits as starting points often use the DNA binding location as a vector to add other chemical probes or as ways to modify the physical properties of the molecule (improve solubility, for example) without modifying the binding. That said, the DNA attachment point can still provide a contribution to the binding
    + This also got me thinking, can the molecule bind to the protein through a building block to which the DNA is attached? Because if not, then the location of Dy could indicate at least, which of the three building blocks is not responsible for the binding. I thought so, but it seems that building blocks can indirectly affect a molecule's ability to bind, and there is no way to determine
    + Yes, a building block attached to the DNA can contribute to the binding and it often does. However, the closer you get to the linker attachment point, the more likely you are to be headed towards solvent (water) and not deeper into the protein.

- https://www.kaggle.com/competitions/leash-BELKA/discussion/491427
    + Not sure how cell painting or gene expression data would help, but there are some good pose prediction datasets out there like pdbbind or maybe plas-5k that might help
- https://www.kaggle.com/competitions/leash-BELKA/discussion/491362#2737103
    + The test set is made from the combination of several different splitting strategies. That statement was only meant to describe one of the strategies: the bb-split. for this split we hold out certain building blocks. But the overall test set also includes molecules from a scaffold split and a random split, hence the overlap.
- https://www.kaggle.com/competitions/leash-BELKA/discussion/503232
    + till, the kinds of physical interactions that drive binding (hydrogen bonds, shape, pi-stacking, Van der Waals, charge distribution, etc.) are universal.
- https://www.kaggle.com/competitions/leash-BELKA/discussion/508613
    + ast time i made a bug, train with "linker [Dy] replaced by C" and infer with "[Dy] only". The model basically collapses and local CV prediction precision drops significantly from 0.65 to 0.22 (which surprises me … just one atom can make such large difference)
- https://github.com/tczhangzhi/awesome-normalization-techniques
- https://projects.volkamerlab.org/teachopencadd


# Features
- Scaffold: Bemis-Murcko scaffold representation, which identifies rings, linkers, frameworks (combinations between linkers and rings) and atomic properties such as atom type, hibridization and bond order
- Molecular Weight
-   from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.ML.Cluster import Butina

- Descriptors:
    + Mordred: https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#mordreddescriptors
    + WHALES for Scaffold-Hopping: https://github.com/grisoniFr/whales_descriptors
    + https://github.com/bp-kelley/descriptastorus

- 3D Graph
    + Molformer: Motif-based Transformer on 3D Heterogeneous Molecular Graphs
    + DimeNet
    + SphereNet

- String
    + AIS: https://github.com/snu-lcbc/atom-in-SMILES
    + SELFIES: https://github.com/HUBioDataLab/SELFormer
    + DeepSMILES: https://github.com/baoilleach/deepsmiles


# Ideas
- Crazy: Convert SMILE to image -> Image classification :))
- Suzuki reaction?: https://github.com/rxn4chemistry/rxnfp

# Conclude
- AvgPool > MaxPool in 3-targets training. Opposite trend with 1-target or not?
- ais > chem_tokenize > char_tokenize

# TODOs
- CharBERT → CNN with multi ksize filter (ngram); combine of multiple kernel sizes as InceptionNet
- TF-IDF features
- Identify key tokens/words which affect binding: **Exploring data-driven chemical SMILES tokenization approaches to identify key protein–ligand binding moieties**
- Determine best neg/pos for cbm
- doRandom=True 5 times for each mol, then count tf/idf to find which keyword is most important?
- MAMBA: [https://arxiv.org/pdf/2406.03344](https://arxiv.org/pdf/2406.03344)
- Sample weights by fraction of pos sample in BB. Larger pos fraction → small weight
- Implement Hit-list comparation between models
- Find a way to create mol from bb in new library. most likely that mol not exist in test has label 0, which host decide to filter out to keep pos/neg fixed at good ratio. Also, pos/neg ratio in new library is very low → need sampling by label → train with high neg weight
- Subsample ->
    - Train multiple models -> ensemble
    - Batch Training/Incremental Learning
- Traing 1 model for each target, not all targets
- Train a model to predict 3 targets -> finetune to predict 1 target -> Reduce effort to balance class weights
- SWA, AWP to improve generalization
- In contrast to fingerprint, the SMILES modality could experience a performance drop on some datasets (HIV and MUV) in the high-data regime -> try to train SMILES string transformer on less data to see the effect? For SMILES string, scaling-law did not follow -> Try to train with less data may give better results
- even the superior RoBERTa model overall does not surpass the performance of graph and fingerprint modalities -> Dive into ECFP Fingerprint and try other fingerprints as well
- How to select relevent features for non-share: Feature selection based on test set, especially nonshare. Select features with highest variance. Only train on these features on train set -> hopefully increase nonshare score
- Noisy Student or other methods of Semi-Supervised Learning
- Train model on `BB1<sep>BB2<sep>BB3`
- Ensemble of multiple weak learner (CNN, Tree) outperform stronger one ?
- DoRandom training reduce CV score -> model actually learn a chunk of chars as feature -> Random Permutation as TTA -> Max()/Mean() to get probability
- GNN:
    + Pretrained GIN (2D)
    + GINE (2D)
    + GCN (2D)
    + GraphSAGE (2D)
    + GAT (2D)
    + AttentiveFP (2D)
    + DimeNet++ (3D)
    + GraphFormer (NLP?)


- ~~Most common substructure on test data, then find N blocks with highest pct of having that structure~~
- ~~Reg: very large dropout in MLP, large dropout in input (fingerprint dropout, random erase,..)~~
- ~~retrain on pseudo label → better order~~
- ~~sample weights by mol_per_scaffold: A strategy to address this bias is to diversify the ligands by clustering actives by their scaffolds and selecting representatives → ignore bias by scaffolds~~
- ~~Nearest neighbor in pretrained embedding space~~
- ~~pseudo label on ZINC + semi~~
- ~~MOE~~
- ~~IBN, AIN for generalization:~~
- ~~AIS Tokenization + MLP:~~ low indomain performance, may be good for specific protein
- ~~Non-overlap train-test split by scaffold similarity, Tanimoto similarity, Sphere Clustering, SELFIES VAE Embedding, .. to see what correlate best to public/private LB~~
- ~~For new library, just consider input as bb1 + bb2 + bb3 instead of whole mol with core inside → reduce gap between train (triazine) and test (non-triazine)~~
- ~~Train a VAE on BELKA → use latent emebedding as input feature to model~~
- ~~Multi-modality pretraining: Random Modality Dropout, independent head for each modality, same number of modality per batch to reduce pre-training/finetuning time~~
- ~~Use BB features (such as RDKIT210, Mordred) and agg (max, mean, ..) for faster feature computation instead compute for each molecule (~100M) → too slow to compute Mordred~~
- ~~Multiple ksize in Squeezeformer Conv~~
- ~~Determine best starting ksize in Conv by experiments~~
- ~~Attention pooling~~
- ~~sEH per is very high → because of BB1. ignore BB1 specific features could improve generalization. Use bb2 + 3 as input to model~~
- ~~Check if mlp able to predict bb1/2/3~~
- ~~Try to embed bb1/2/3 by a fixed len vector: train a MLP accept onehot bb as input (actually embedding of each bb is weight)~~
- ~~FindMCS + RascalMCES rdkit~~
- ~~SMARTS~~
- ~~Given a set of unlabled data, how to determine which feature is important? Train a model to predict/rank a list of feature in an unsupervised way~~
- ~~Clustering based on trained model embedding → which one close to test → train on these samples only~~
- ~~For triazine core in train → bb2/3 can be swap. But for non-triazine core? Order of bb2/3 is matter or not?~~
- ~~GNN with coordinate normalize to Dy attachment point, like ASL~~
- ~~use trained bb mlp embedding as bb embed → embed similarity to find similar bb as test set~~
- ~~MultiModality Iterative Refinement~~
- ~~For each bb1, train a GBDT model to get top K most important features. For each features, count the number of time it was considered as important feature. Get top P features with highest count for training. Can extend the term bb1 to bb2/3, scaffold,..~~
- ~~clustering → train cbm on each cluster → find important features~~
- ~~Neural Fingerprint model: linear → sigmoid → threshold → linear to simulate a Neural FingerPrint. In test, identify which bits in FP is constant → remove~~
- ~~Integrate the above Neural Fingerprint above to pretraining phase~~
- ~~Oversample per BB1 group, per scaffold~~
- ~~How to identify Domain specific features: Overfit a features predictor on train, then infer on test. High error feature prediction is Out of Distribution features~~
- ~~features selection based on KLD between features distribution in train-test set: discrete, non-continuous~~
- ~~SELFIES/AIS/Deepsmiles Fingerprint → likely real for SELFIES~~
- ~~EDA about the 377-70-448-70 and triazines relationship, related to [https://www.kaggle.com/competitions/leash-BELKA/discussion/493294](https://www.kaggle.com/competitions/leash-BELKA/discussion/493294)~~
- ~~Train on all data with sample/class weight~~
- ~~Norm layers: Ghost BN, IBN, AIN, IAS~~
- ~~Weighted loss by pos ratio: some building blocks is easier to predict, or has pos ratio much higher than other -> small the weight to make model less bias toward these samples~~
- ~~Add more RDKit features/descriptor~~
- ~~Concat all molecule string representation -> Single NLP model~~
- ~~Concat Protein string representation for attention~~
- ~~MolFormer: Linear Attention + RoPE~~
- ~~Pretrained model with frozen arch for non-share bbs: Molformer, Chemformer-2: [https://www.kaggle.com/competitions/leash-BELKA/discussion/498983](https://www.kaggle.com/competitions/leash-BELKA/discussion/498983)~~
- ~~Neural Scaling Law: This behavior suggests the phenomenon of parameter ossification [54] in pre-trained models, suggesting that pre-training can inadvertently “freeze” the model weights in a way that limits their adaptability to the fine-tuning distribution~~
- ~~Data Pruning: Herding [56], Entropy [57], Least Confidence [57], Forgetting [58], GraNd [59], and k-means~~
- ~~from fast_transformers.feature_maps import Favor,GeneralizedRandomFeatures (Molformer)~~
- ~~Warm-starting / piggybacked training: SB first, LB later to improve generalization~~
- ~~Scaffold ID, Block ID as categorical features~~
- ~~The ideal data set would be a data set of uniformly sampled data points from the chemical space without bias for any methodology: -> Plot T-SNE of Molformer/trained foundation model, uniformly select molecules in that embedding space for training~~
- ~~Target Transform: Multiply each bind score of 0/1 to bind fraction corresponding to each building block~~
- ~~10/20 Smoothing~~