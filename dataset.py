# dataset.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ZooLakeDataset(Dataset):
    """
    Dataset loader for the ZooLake cropped images.
    Expects split files containing paths relative to the dataset root (as in the uploaded archive).
    """

    def __init__(self, root_dir, split_txt, transform=None, img_size=224):
        self.root = Path(root_dir)

        # read split file
        with open(split_txt, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        # rebuild filepaths relative to data_root
        filepaths = []
        for p in lines:
            p = Path(p)
            if "zooplankton_0p5x" in p.parts:
                idx = p.parts.index("zooplankton_0p5x")
                rel = Path(*p.parts[idx:])  # start from "zooplankton_0p5x"
                filepaths.append(self.root / rel)
            else:
                filepaths.append(self.root / p)
        self.filepaths = filepaths

        # infer classes from directory structure
        classes = sorted(list({p.parent.parent.name for p in self.filepaths if len(p.parts) >= 3}))
        self.classes = classes
        self.class2idx = {c: i for i, c in enumerate(self.classes)}

        # build (path, label) items
        items = []
        for p in self.filepaths:
            parts = p.parts
            if "training_data" in parts:
                idx = parts.index("training_data")
                cls_name = parts[idx - 1]
            else:
                cls_name = p.parent.name
            if cls_name not in self.class2idx:
                self.class2idx[cls_name] = len(self.classes)
                self.classes.append(cls_name)
            label = self.class2idx[cls_name]
            items.append((p, label))
        self.items = items

        # default transform
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, str(path)
