import torch
from torch.utils.data import Dataset
import os

class NER_SA_Dataset(Dataset):
    def __init__(self, directory:str='./data/train'):
        super().__init__()

        self._file_paths = [os.path.join(directory, file) for file in os.listdir(directory)]
        self._batch_size = torch.load(self._file_paths[0], weights_only=True)['embeddings'].shape[0]
        self._batch_in_memory = (-1, None)

    
    def __len__(self):
        last_batch_size = torch.load(self._file_paths[-1], weights_only=True)['embeddings'].shape[0]
        return (len(self._file_paths) -1) * self._batch_size + last_batch_size
    
    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        batch_idx = index // self._batch_size
        if self._batch_in_memory[0] != batch_idx:
            self._cache_batch(batch_idx)

        rel_idx = index % self._batch_size
        batch_data = self._batch_in_memory[1]

        return {field: batch_data[field][rel_idx] for field in ('embeddings', 'labels', 'sentiments')}
    
    def _cache_batch(self, n_batch:int):
        '''
        Stores the desired batch in memory so as not to read it each 
        "__getitem__" if the desired item is in the batch.
        '''
        self._batch_in_memory = (n_batch, torch.load(self._file_paths[n_batch], weights_only=True))



if __name__ == '__main__':
    ds = NER_SA_Dataset('./data/train')
    for i in (0, 1, 2, 3, 4, 5, 99, 100, 101, 102, 103, 104, 105):
        batch_in_memo = ds._batch_in_memory[0]
        print(i)
        ds[i]
        if ds._batch_in_memory[0] != batch_in_memo:
            batch_in_memo = ds._batch_in_memory[0]
            print(f'Loading batch {batch_in_memo} to memory...')
