import polars as pl
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm



class TestDataset(Dataset):
    def __init__(self):
        df = pl.scan_csv('/home/dangnh36/datasets/competitions/leash_belka/processed/train_v2.csv').select(
            pl.col('*')).collect()
        print(df.estimated_size('gb'), 'GB')
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        mol = self.df[index, 'molecule']
        assert len(mol) > 0
        for col in self.df.columns:
            assert self.df[index, col] is not None
        return torch.zeros((456, 123)).float()
    

# class TestDataset(Dataset):
#     def __init__(self):
#         df = pl.scan_csv('/home/dangnh36/datasets/competitions/leash_belka/processed/train_v2.csv').select(
#             pl.col('molecule')).collect()['molecule'].to_list()
#         # print(df.estimated_size('gb'), 'GB')
#         self.df = df

#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, index):
#         mol = self.df[index]
#         assert len(mol) > 0
#         return torch.zeros((456, 123)).float()
    

if __name__ == '__main__':
    dataset = TestDataset()
    loader = DataLoader(dataset, batch_size = 2048, shuffle = True, num_workers=16)
    for batch in tqdm(loader):
        assert batch.shape[0] > 8, f'{batch.shape}'
        assert batch.min() == 0
        