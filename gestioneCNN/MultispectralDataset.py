from torch.utils.data import Dataset
import torch
import os
import tifffile
import numpy as np

class MultispectralDataset(Dataset):
    def __init__(self, root_dir,device, transform=None, preload=False, format=".npy"):
        self.samples = []
        self.transform = transform
        self.preload = preload
        self.images = []
        self.device=device
        self.format=format.lower()
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((self.format)):
                    path = os.path.join(cls_path, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))
                    if self.preload:
                        self.images.append((self.load_image(path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            img, label = self.images[idx]
        else:
            path, label = self.samples[idx]
            img = self.load_image(path)
            
            #img = torch.from_numpy(tifffile.imread(path).astype("float32")).permute(2, 0, 1)
            #if self.device:
            #    img = img.to(self.device)
            
        if self.transform:
            img = self.transform(img)
        
        return img, label

    def load_image(self,path):
        if self.format==".tiff" or self.format==".tif":
            img = tifffile.imread(path).astype(np.float32)
        elif self.format==".npy":
            img = np.load(path, mmap_mode='r')  # evita di caricare tutto in RAM
            img = img.copy()  # risolve il warning
        else:
            raise ValueError(f"Formato file non supportato: {self.format}")
        return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()