from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import json
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None, image_size=(512, 512), class_file="classes.txt"):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.image_size = image_size
        self.transform = transform
        self.class_file = class_file

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

        self.classes = self._build_class_list()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.class_to_idx["background"] = 0  # background sempre 0
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.idx_to_class)

    def _build_class_list(self):
        if os.path.exists(self.class_file):
            with open(self.class_file, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
            if "background" not in labels:
                labels = ["background"] + labels
            return labels

        label_set = set()
        for img_file in self.image_files:
            json_path = os.path.join(self.json_dir, img_file.replace('.png', '.json'))
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                    features = data.get("markResult", {}).get("features", [])
                    for feat in features:
                        label = feat.get("properties", {}).get("content", {}).get("label", None)
                        if label:
                            label_set.add(label)
            except Exception as e:
                print(f"Errore nel parsing di {json_path}: {e}")

        labels = ["background"] + sorted(label_set)
        with open(self.class_file, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(label + "\n")
        return labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        gc.collect()
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        json_path = os.path.join(self.json_dir, img_file.replace('.png', '.json'))

        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for feature in data.get("markResult", {}).get("features", []):
                    label = feature.get("properties", {}).get("content", {}).get("label", "background")
                    label_idx = self.class_to_idx.get(label, 0)
                    coords = feature.get("geometry", {}).get("coordinates", [])
                    for polygon in coords:
                        polygon_points = [(x,y) for x, y in polygon]
                        draw.polygon(polygon_points, fill=label_idx)
    
        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask, dtype=np.uint8).squeeze()
        #print("Image shape:", image.shape)
        #print("Mask shape:", mask.shape)
        #mask_np = np.array(mask, dtype=np.uint8)       # H×W uint8
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_np)
            image = augmented['image']
            mask = augmented['mask']
            print("Transformed image:", augmented['image'].shape, augmented['image'].dtype)
            print("Transformed mask:", augmented['mask'].shape, augmented['mask'].dtype)

        else:
            image = np.transpose(image_np,(2, 0, 1)).float()
            mask = mask_np

        img_np = image.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)

        plt.subplot(1,2,1)
        plt.imshow(img_np)
        plt.title("Image")

        plt.subplot(1,2,2)
        plt.imshow(np.transpose(mask.cpu().numpy(), (1, 0)))#mask.cpu().numpy(),cmap='jet')#(np.flip(np.transpose(mask.cpu().numpy(), (1, 0)) , axis=0)), cmap='jet')
        plt.title("Mask")
        plt.show()
        return image, mask
