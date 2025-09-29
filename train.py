# train.py
import argparse, os
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from dataset import ZooLakeDataset
from model import create_student, create_teacher
from utils import save_checkpoint, compute_metrics
from tqdm import tqdm
import numpy as np

def distillation_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5):
    ce = nn.CrossEntropyLoss()(student_logits, targets)
    p = nn.functional.log_softmax(student_logits / T, dim=1)
    q = nn.functional.softmax(teacher_logits / T, dim=1)
    kl = nn.functional.kl_div(p, q, reduction='batchmean') * (T * T)
    return alpha * ce + (1. - alpha) * kl

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = ZooLakeDataset(args.data_root, args.train_split, transform=train_tf, img_size=args.img_size)
    val_ds = ZooLakeDataset(args.data_root, args.val_split, transform=val_tf, img_size=args.img_size)
    num_classes = len(train_ds.classes)

    # build sampler to mitigate imbalance
    class_counts = [0] * num_classes
    for _, label in train_ds.items:
        class_counts[label] += 1
    # avoid divide by zero
    class_weights = [1.0 / (c + 1e-6) for c in class_counts]
    sample_weights = [class_weights[label] for _, label,in train_ds.items]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    student = create_student(num_classes).to(device)

    teacher = None
    if args.distill:
        teacher = create_teacher(num_classes).to(device)
        if args.teacher_ckpt and os.path.exists(args.teacher_ckpt):
            ck = torch.load(args.teacher_ckpt, map_location=device)
            teacher.load_state_dict(ck['model_state'])
        teacher.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_macro_f1 = 0.0
    for epoch in range(args.epochs):
        student.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = student(imgs)
            if args.distill and teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(imgs)
                loss = distillation_loss(logits, t_logits, labels, T=args.distill_T, alpha=args.distill_alpha)
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        scheduler.step()

        # validation
        student.eval()
        ys, ypreds = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                logits = student(imgs)
                preds = logits.argmax(dim=1).cpu().numpy()
                ypreds.extend(preds.tolist())
                ys.extend(labels.numpy().tolist())
        metrics = compute_metrics(ys, ypreds, labels=list(range(num_classes)))
        print(f"Epoch {epoch+1} Val Macro-F1: {metrics['macro_f1']:.4f} Acc: {metrics['accuracy']:.4f}")
        if metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = metrics['macro_f1']
            save_checkpoint({'model_state': student.state_dict(),
                             'optimizer_state': optimizer.state_dict(),
                             'epoch': epoch,
                             'classes': train_ds.classes},
                            os.path.join(args.output_dir, 'best_student.pth'))

    save_checkpoint({'model_state': student.state_dict(),
                 'optimizer_state': optimizer.state_dict(),
                 'epoch': args.epochs,
                 'classes': train_ds.classes},
                os.path.join(args.output_dir, 'final_student.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="C:\\Users\\Ashwith\\OneDrive\\Desktop\\data")
    parser.add_argument('--train_split', default="C:\\Users\\Ashwith\\OneDrive\\Desktop\\data\\zoolake_train_test_val_separated\\train_filenames.txt")
    parser.add_argument('--val_split', default="C:\\Users\\Ashwith\\OneDrive\\Desktop\\data\\zoolake_train_test_val_separated\\val_filenames.txt")
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', default='./checkpoints')
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--teacher_ckpt', default=None)
    parser.add_argument('--distill_T', type=float, default=4.0)
    parser.add_argument('--distill_alpha', type=float, default=0.5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
