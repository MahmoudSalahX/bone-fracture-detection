
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_dataloaders(data_dir, batch_size):
    """
    Loads and splits the data into training, validation, and test sets.
    """
    transforms_all = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = ImageFolder(data_dir, transform=transforms_all)
    
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Data successfully split:")
    print(f" - Training images:   {len(train_dataset)}")
    print(f" - Validation images: {len(val_dataset)}")
    print(f" - Test images:       {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, full_dataset.classes

def plot_performance_curves(history):
    """
    Plots the performance curves (loss and accuracy) vs. epochs and saves the figure.
    """
    num_epochs = len(history['train_loss'])
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), history['train_loss'], 'o-', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), history['val_loss'], 'o-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), history['train_acc'], 'o-', label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), history['val_acc'], 'o-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    
    plt.tight_layout()
    # NEW: Save the figure to a file
    plt.savefig('performance_curves.png')
    plt.show()
    plt.close() # Close the figure to free up memory

def plot_confusion_matrix(model, loader, dataset_name, class_names, device):
    """
    Calculates and plots the confusion matrix for a given dataset and saves the figure.
    """
    y_pred, y_true = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {dataset_name} Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # NEW: Save the figure to a file (with a unique name)
    filename = f"confusion_matrix_{dataset_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.show()
    plt.close() # Close the figure to free up memory
