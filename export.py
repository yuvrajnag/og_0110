# export.py
import argparse, os, torch
from model import create_student

def torchscript_export(ckpt, out_path, img_size=224, device='cpu', num_classes=35):
    model = create_student(num_classes)
    ckpt_data = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_data['model_state'])
    model.eval().to(device)
    example = torch.rand(1,3,img_size,img_size).to(device)
    traced = torch.jit.trace(model, example)
    traced.save(out_path)
    print("Saved TorchScript to", out_path)

def onnx_export(ckpt, out_path, img_size=224, opset=13, num_classes=35):
    model = create_student(num_classes)
    ckpt_data = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt_data['model_state'])
    model.eval()
    dummy = torch.randn(1,3,img_size,img_size)
    torch.onnx.export(model, dummy, out_path, input_names=['input'], output_names=['output'], opset_version=opset)
    print("Exported ONNX:", out_path)

def dynamic_quantize(ckpt, out_path, img_size=224, num_classes=35):
    model = create_student(num_classes)
    ckpt_data = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt_data['model_state'])
    model.eval()
    qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save({'model_state': qmodel.state_dict(), 'classes': ckpt_data.get('classes')}, out_path)
    print("Saved dynamic-quantized model to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=['torchscript','onnx','dynamic_quant'], default='onnx')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, required=True)
    args = parser.parse_args()
    if args.mode == 'torchscript':
        torchscript_export(args.ckpt, args.out, args.img_size, 'cpu', args.num_classes)
    elif args.mode == 'dynamic_quant':
        dynamic_quantize(args.ckpt, args.out, args.img_size, args.num_classes)
    else:
        onnx_export(args.ckpt, args.out, args.img_size, 13, args.num_classes)
