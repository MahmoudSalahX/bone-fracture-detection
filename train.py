
import torch
import torch.nn as nn
import torch.optim as optim
import time
import config
from utils import get_dataloaders, plot_performance_curves, plot_confusion_matrix
from model import create_model

def main():
    # Set up the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the data using the function from utils.py
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config.DATA_DIR, config.BATCH_SIZE)

    # Create the model using the function from model.py
    model = create_model(device)

    # Define the optimizer and loss function using settings from config.py
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # --- Start Training ---
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    print("\nStarting model training...")

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_train_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset.indices)
        epoch_train_acc = 100 * correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # Validation phase
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset.indices)
        epoch_val_acc = 100 * correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} [{epoch_time:.2f}s] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    print("\n--- Training Complete ---")

    # Save the trained model
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    # --- Display Analysis ---
    plot_performance_curves(history)
    
    print("\n--- Confusion Matrices ---")
    plot_confusion_matrix(model, train_loader, "Training", class_names, device)
    plot_confusion_matrix(model, val_loader, "Validation", class_names, device)
    plot_confusion_matrix(model, test_loader, "Testing", class_names, device)

if __name__ == '__main__':
    main()
