# inference_count.py
import argparse, os
from model import create_student
from dataset import ZooLakeDataset
from torchvision import transforms
from PIL import Image
import torch
from collections import Counter

def load_model(ckpt_path, num_classes, device, img_size):
    model = create_student(num_classes)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model

def count_crops(folder_with_images, model, classes, device, img_size=224):
    tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    counts = Counter()
    for root,_,files in os.walk(folder_with_images):
        for fname in files:
            if not fname.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            p = os.path.join(root, fname)
            img = Image.open(p).convert('RGB')
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1).item()
                counts[classes[pred]] += 1
    return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/mnt/data/dataset_extracted/data')
    parser.add_argument('--ckpt', default='./checkpoints/best_student.pth')
    parser.add_argument('--folder', required=True)
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    ds = ZooLakeDataset(args.data_root, os.path.join(args.data_root, 'zoolake_train_test_val_separated/train_filenames.txt'), img_size=args.img_size)
    classes = ds.classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, len(classes), device, args.img_size)
    counts = count_crops(args.folder, model, classes, device, img_size=args.img_size)
    print("Counts:")
    for cls,c in counts.most_common():
        print(f"{cls}: {c}")
