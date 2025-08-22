
import torch
import torch.nn as nn
import torchvision.models as models

def create_model(device):
    """
    Creates and prepares a ResNet50 model for fine-tuning.
    """
    model = models.resnet50(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 2 classes: fracture, normal
    model.to(device)
    
    # Unfreeze all layers for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
        
    return model
