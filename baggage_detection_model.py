import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import glob
import xml.etree.ElementTree as ET
import time

# Define paths
DATA_ROOT = r"C:\Users\Lenovo\.cache\kagglehub\datasets\orvile\x-ray-baggage-anomaly-detection\versions\1"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LABEL_DIR = os.path.join(DATA_ROOT, "train", "labels")
VALID_IMG_DIR = os.path.join(DATA_ROOT, "valid", "images")
VALID_LABEL_DIR = os.path.join(DATA_ROOT, "valid", "labels")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "images")
TEST_LABEL_DIR = os.path.join(DATA_ROOT, "test", "labels")

# Configuration
BATCH_SIZE = 16
IMAGE_SIZE = 256
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class BaggageDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.classes = ['Knife', 'Gun', 'Wrench', 'Pliers', 'Scissors']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get corresponding XML file
        xml_path = os.path.join(self.label_dir, img_name.split('.')[0] + '.xml')
        
        # Default: no objects (normal baggage)
        has_danger = 0  # Binary classification: 0 = normal, 1 = dangerous
        
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = root.findall("object")
            
            # If any dangerous objects exist in the image
            if len(objects) > 0:
                has_danger = 1
        
        if self.transform:
            image = self.transform(image)
            
        return image, has_danger

# Define transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = BaggageDataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform=transform)
valid_dataset = BaggageDataset(VALID_IMG_DIR, VALID_LABEL_DIR, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Define model using transfer learning with ResNet18
class BaggageDetectionModel(nn.Module):
    def __init__(self):
        super(BaggageDetectionModel, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the final layer for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet(x)

# Initialize model
model = BaggageDetectionModel().to(DEVICE)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

def train_model():
    best_valid_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE).view(-1, 1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * images.size(0)
            predicted = (outputs >= 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Valid]"):
                images = images.to(DEVICE)
                labels = labels.float().to(DEVICE).view(-1, 1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item() * images.size(0)
                predicted = (outputs >= 0.5).float()
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
        
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = valid_correct / valid_total
        
        # Update learning rate based on validation loss
        scheduler.step(valid_loss)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'best_baggage_model.pth')
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")
    
    return history

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['valid_loss'], label='Valid Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['valid_acc'], label='Valid Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model():
    # Load best model
    model.load_state_dict(torch.load('best_baggage_model.pth'))
    model.eval()
    
    test_dataset = BaggageDataset(TEST_IMG_DIR, TEST_LABEL_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    test_correct = 0
    test_total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on test set"):
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE).view(-1, 1)
            
            outputs = model(images)
            predicted = (outputs >= 0.5).float()
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return test_accuracy

def predict_single_image(image_path):
    """Predict if a single image contains dangerous items."""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    preprocess = transform
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    
    # Load model if needed
    if not hasattr(predict_single_image, "model_loaded"):
        model.load_state_dict(torch.load('best_baggage_model.pth'))
        model.eval()
        predict_single_image.model_loaded = True
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output >= 0.5).float().item()
        confidence = output.item()
    
    return prediction, confidence

if __name__ == "__main__":
    # Train the model and get training history
    history = train_model()
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    test_accuracy = evaluate_model()
    
    print("Training complete!")