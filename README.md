11th place solution (gold medal) of [NeurIPS 2024 - Predict New Medicines with BELKA competition on Kaggle](https://www.kaggle.com/competitions/leash-BELKA)

**Solution writeup:** https://www.kaggle.com/competitions/leash-BELKA/discussion/518993


This repo use [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) and [Hydra](https://github.com/facebookresearch/hydra) for flexible configuration system. One could add more tasks, models, datasets, input representations while keeping all in this single flexible codebase.


If you have any requests/questions, feel free to raise those with me in Github Issues.

## Installation
Recommended to use [this Dockerfile](docker/Dockerfile)

## Some small tricks
- Use container object without reference count for each element (see [this](https://github.com/pytorch/pytorch/issues/13246)) in dataset/dataloader to reduce memory usage. For example, if dataset contain a large list of SMILES strings, it's better to stored it in a Polars dataframe, or a Huggingface's dataset. I found the later Huggingface's dataset show better performance while allow to be on-disk loading, so it's probably the best choice. All you need is about 8GB or disk space and 20 GB of RAM is enough to train a big model on all 100M data
- Use numpy memmap to stored pre-generated features on disk as well
- Make Torch's Dataset's `__get_item__()` accept a list of indices as input, e.g in [src/data/datasets/pretrain_tokenize_mtr_mlm.py](src/data/datasets/pretrain_tokenize_mtr_mlm.py). This allow to use on-the-fly tokenization, which is needed for SMILES Enumeration (`Chem.MolToSmiles(doRandom=True)`). Combine with the above tricks, dataloading work blazy fast with flexible modifications (e.g, you need not to tokenize all data before training)


## Utilities
For this large amount of data, one need to transform dataset to another optimized format for faster dataloader/memory saving, create tokenizers, extract useful features, etc.

The [notebooks](notebooks) directory contains some useful notebooks which is not fully ready, but easy to modify suit for need, e.g just change some paths which are specific to your local environment.

- Pre-generate features (e.g, Fingerprints,..): [src/tools/extract_features.py](src/tools/extract_features.py), e.g `python3 src/tools/extract_features.py --feature ecfp6 --num-chunks 1 --chunk-idx 0 --batch-size 1000 --subset test`
- Simulate LB Cross-Validation split: [full_final_cv_split.ipynb](notebooks/full_final_cv_split.ipynb)
- Random Split: [random_split.ipynb](notebooks/random_split.ipynb)
- Save as HuggingFace's [datasets](https://github.com/huggingface/datasets) for optimized on-disk dataloader: [save_hf_datasets.ipynb](notebooks/save_hf_datasets.ipynb)
- Create and save tokenizers (support SMILES chars, AIS, SELFIES, DeepSMILES): [save_tokenizers.ipynb](notebooks/save_tokenizers.ipynb). Pre-generated tokenizers also included in this repo in [assets/tokenizer_v2](assets/tokenizer_v2/) 
- CatBoost training: [catboost-final.ipynb](notebooks/catboost-final.ipynb)



## Training
Just change the configuration files and start training with multiple Task/Model/Dataset.
**I will update more example training commands to different Tasks, Models and Datasets later**. Many training commands are tracked in [docs/exp_tracking.xlsx](docs/exp_tracking.xlsx) and readily to be tried as well.

Starting by
```
cd src/
```

### Reproduce Roberta + Join MLM+MTR training
#### Prerequisite
Refer to [Utilities](#utilities) for more details
- Saved SMILES char Tokenizer, also included in [assets/tokenizer_v2/smiles_char](assets/tokenizer_v2/smiles_char)
- Train/Test Dataset in Huggingface's dataset format
- Cross validation metadata: `skf20` means `StratifiedKFold(n_splits=20)`


#### Pretraining with MLM + MTR:
```
cd src
python3 train.py exp=pretrain_mtr_mlm exp_name=pretrain-mtr-mlm_roberta-depth8-dim256-abs optim.lr=5e-4 optim.weight_decay=0.0 optim.name='timm@adam' data.tokenizer.name=smiles_char model=roberta model.vocab_size=44 model.encoder.num_hidden_layers=8 model.encoder.hidden_size=256 model.encoder.intermediate_size=512 model.pool_type=concat_attn model.head.dropout=0.1 model.head.type=mtr_mlm loader.train_batch_size=2048 trainer.max_epochs=10 trainer.precision=16-mixed loader.num_workers=16
```

#### Finetune on competition task
replace `ckpt_path` to your pretrained checkpoint path, e.g
```
python3 train.py exp=finetune_tokenize exp_name=finetunemtr-smiles_char_roberta-depth8-dim256-lr5e-4 optim.lr=5e-4 optim.weight_decay=0.0 optim.name='timm@adam' data.tokenizer.name=smiles_char model=roberta model.vocab_size=44 model.encoder.num_hidden_layers=8 model.encoder.hidden_size=256 model.encoder.intermediate_size=512 model.pool_type=concat_attn 'model.head.mlp_chans=[1024,1024]' model.head.dropout=0.3 loader.train_batch_size=2048 trainer.max_epochs=12 trainer.precision=16-mixed loader.num_workers=16 cv.strategy=skf20 cv.fold_idx=0 cv.train_on_all=True 'ckpt_path="outputs/train/run/06-26/08-16-02.947187_pretrain-mtr-mlm_roberta-depth8-dim256-abs/fold_0/ckpts/ep=3_step=215000_train|loss=0.000781.ckpt"'
```

## Submit
Just add flags `train=False predict=True ckpt_path={YOUR CHECKPOINT PATH HERE}` to your training command.


