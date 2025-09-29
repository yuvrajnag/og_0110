# eval.py
import argparse, os
import torch
from torch.utils.data import DataLoader
from dataset import ZooLakeDataset
from model import create_student
from utils import compute_metrics

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ZooLakeDataset(args.data_root, args.test_split, img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = create_student(len(ds.classes))
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    ys, ypreds = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            ypreds.extend(preds.tolist())
            ys.extend(labels.numpy().tolist())
    metrics = compute_metrics(ys, ypreds, labels=list(range(len(ds.classes))))
    print("Accuracy:", metrics['accuracy'])
    print("Macro F1:", metrics['macro_f1'])
    for i, cls in enumerate(ds.classes):
        print(f"{cls}: F1={metrics['per_class_f1'][i]:.3f} P={metrics['per_class_precision'][i]:.3f} R={metrics['per_class_recall'][i]:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="C:\\Users\\Ashwith\\OneDrive\\Desktop\\data")
    parser.add_argument('--test_split', default="C:\\Users\\Ashwith\\OneDrive\\Desktop\\data\\zoolake_train_test_val_separated\\test_filenames.txt")
    parser.add_argument('--ckpt', default='./checkpoints/best_student.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()
    evaluate(args)
