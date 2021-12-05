from torch.utils.data import Dataset
import os
import numpy as np
import torch


class MnistScenesGenerator:
    """Generator of scenes with digits from MNIST"""
    
    def __init__(self, data, n_samples=10**5, path='./'):
        self.path = path
        self.data = data
        self.n_samples = n_samples
        self.y_starts = [i * 42 for i in range(3)]
        self.x_starts = [i * 42 for i in range(3)]
        
    def gen_sample(self):
        scene = np.zeros((1, 128, 128))
        masks = []
        labels = []
        for y_start in self.y_starts:
            for x_start in self.x_starts:
                img_num = np.random.randint(len(self.data))
                img = self.data[img_num][0].numpy()
                label = self.data[img_num][1]
                x_shift = np.random.randint(14)
                y_shift = np.random.randint(14)
                x = x_start + x_shift
                y = y_start + y_shift
                if np.random.randint(10) <= 6:
                    scene[:, y:y + 28, x:x + 28] = img
                    labels.append(label)
                    cur_mask = np.zeros_like(scene)
                    cur_mask[:, y:y + 28, x: x + 28] = img != 0
                    masks.append(cur_mask)
                else:
                    cur_mask = np.zeros_like(scene)
                    masks.append(cur_mask)
                    labels.append(-1)
        return scene, np.array(masks), np.array(labels) 
    
    def generate(self):
        for k in range(self.n_samples):
            scene, masks, labels = self.gen_sample()
            labels = np.array(labels)
            with open(os.path.join(self.path, f'mask{k}.np'), 'wb') as f:
                np.save(f, masks.astype('bool'))
            with open(os.path.join(self.path, f'scene{k}.np'), 'wb') as f:
                np.save(f, scene.astype('float16'))
            with open(os.path.join(self.path, f'labels{k}.np'), 'wb') as f:
                np.save(f, labels.astype('float16'))


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
            labels = (labels != -1) * 1. 
        for j in range(len(masks)):
                masks[j] = masks[j] * scene
        return {'scene': torch.from_numpy(scene).float(),
                'masks': torch.from_numpy(masks).float(),
                'labels': torch.from_numpy(labels)}
