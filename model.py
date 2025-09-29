# model.py
import timm
import torch.nn as nn

def create_student(num_classes, model_name="mobilenetv3_small_100", pretrained=True):
    """
    Create a small, efficient student model via timm.
    If timm model_name isn't available, fallback to efficientnet_b0.
    """
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception:
        model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
    return model

def create_teacher(num_classes, model_name="efficientnet_b4", pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model
