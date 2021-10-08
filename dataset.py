from torch.utils.data import Dataset
import os
import numpy as np
import torch


class MnistScene(Dataset):
    def __init__(self, path):
        self.path = path
        
    def __len__(self):
        return len(os.listdir(self.path)) // 3
    
    def __getitem__(self, idx):
        with open(os.path.join(self.path, f'mask{idx}.np'), 'rb') as f:
            masks = np.load(f, allow_pickle=True)
        with open(os.path.join(self.path, f'scene{idx}.np'), 'rb') as f:
            scene = np.load(f, allow_pickle=True)
        with open(os.path.join(self.path, f'labels{idx}.np'), 'rb') as f:
            labels = np.load(f, allow_pickle=True)
        return {'scene': torch.from_numpy(scene).float(),
                'masks': torch.from_numpy(masks).float(),
                'labels': torch.from_numpy(labels)}
