import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Training configuration"""
    # Paths
    DATA_DIR = 'data/helmet_dataset'  # Download from Kaggle: pkdarabi/helmet
    MODEL_DIR = 'models'
    OUTPUT_DIR = 'outputs'
    
    # ResNet Training
    RESNET_EPOCHS = 25
    RESNET_BATCH_SIZE = 32
    RESNET_LR = 0.001
    RESNET_IMG_SIZE = 224
    
    # LSTM Training
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 64
    LSTM_LR = 0.001
    LSTM_SEQ_LEN = 10
    LSTM_HIDDEN = 128
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DATASET PREPARATION
# =============================================================================

class HelmetDataset(Dataset):
    """Dataset for helmet classification"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def prepare_helmet_data():
    """
    Prepare helmet dataset from directory structure
    Expected structure (from pkdarabi/helmet dataset):
    data/helmet_dataset/
        train/
            images/
            labels/  # YOLO format
        valid/
            images/
            labels/
    
    Or simple structure:
    data/helmet_dataset/
        with_helmet/
        without_helmet/
    """
    print("📂 Preparing helmet dataset...")
    
    image_paths = []
    labels = []
    
    # Try YOLO format first (from pkdarabi/helmet)
    yolo_dirs = ['train', 'valid', 'test']
    yolo_found = False
    
    for split in yolo_dirs:
        img_dir = os.path.join(Config.DATA_DIR, split, 'images')
        lbl_dir = os.path.join(Config.DATA_DIR, split, 'labels')
        
        if os.path.exists(img_dir):
            yolo_found = True
            print(f"Found YOLO format: {split}")
            
            for img_file in os.listdir(img_dir):
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(img_dir, img_file)
                lbl_path = os.path.join(lbl_dir, img_file.rsplit('.', 1)[0] + '.txt')
                
                # Parse YOLO label (class 1 = with helmet, 2 = without)
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) > 0:
                                cls = int(parts[0])
                                if cls == 1:  # With helmet
                                    image_paths.append(img_path)
                                    labels.append(1)
                                    break
                                elif cls == 2:  # Without helmet
                                    image_paths.append(img_path)
                                    labels.append(0)
                                    break
    
    # If not YOLO, try simple folder structure
    if not yolo_found:
        print("Using simple folder structure...")
        classes = ['without_helmet', 'with_helmet']
        
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(Config.DATA_DIR, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_dir, img_name))
                        labels.append(label)
    
    print(f"✅ Found {len(image_paths)} images")
    print(f"   With Helmet: {sum(labels)}")
    print(f"   Without Helmet: {len(labels) - sum(labels)}")
    
    return image_paths, labels

# =============================================================================
# RESNET18 HELMET CLASSIFIER
# =============================================================================

class ResNetHelmetClassifier(nn.Module):
    """ResNet18 for helmet classification"""
    def __init__(self, num_classes=2, dropout=0.5):
        super(ResNetHelmetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze only first 2 layers (less freezing = better for small dataset)
        for name, param in self.resnet.named_parameters():
            if 'layer1' in name or 'conv1' in name or 'bn1' in name:
                param.requires_grad = False
        
        # Replace FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

def train_resnet():
    """Train ResNet18 helmet classifier"""
    print("\\n" + "="*60)
    print("🧠 Training ResNet18 Helmet Classifier")
    print("="*60)
    
    # Prepare data
    image_paths, labels = prepare_helmet_data()
    
    if len(image_paths) == 0:
        print("❌ No images found! Please download dataset from:")
        print("   https://www.kaggle.com/datasets/pkdarabi/helmet")
        print("   Extract to: data/helmet_dataset/")
        return False
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Data transforms - stronger augmentation for small dataset
    train_transform = transforms.Compose([
        transforms.Resize((Config.RESNET_IMG_SIZE, Config.RESNET_IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.RESNET_IMG_SIZE, Config.RESNET_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HelmetDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = HelmetDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.RESNET_BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.RESNET_BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    
    # Initialize model
    model = ResNetHelmetClassifier().to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.RESNET_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    
    # Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\\n🚀 Starting training on {Config.DEVICE}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    for epoch in range(Config.RESNET_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.RESNET_EPOCHS}')
        for images, labels in pbar:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        print(f"\\nEpoch {epoch+1}/{Config.RESNET_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(Config.MODEL_DIR, 'helmet_resnet18.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ Saved best model: {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training & Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(Config.OUTPUT_DIR, 'resnet_training_curves.png')
    plt.savefig(plot_path)
    print(f"\\n📊 Training curves saved: {plot_path}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Helmet', 'Helmet'],
                yticklabels=['No Helmet', 'Helmet'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"📊 Confusion matrix saved: {cm_path}")
    
    # Classification report
    print("\\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['No Helmet', 'Helmet']))
    
    print(f"\\n🎉 ResNet Training Complete! Best Accuracy: {best_acc:.2f}%")
    return True

# =============================================================================
# LSTM TRAJECTORY PREDICTOR
# =============================================================================

class TrajectoryDataset(Dataset):
    """Dataset for LSTM trajectory prediction"""
    def __init__(self, trajectories, seq_length=10):
        self.sequences = []
        self.targets = []
        
        for traj in trajectories:
            if len(traj) > seq_length:
                for i in range(len(traj) - seq_length):
                    self.sequences.append(traj[i:i+seq_length])
                    self.targets.append(traj[i+seq_length])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]), 
                torch.FloatTensor(self.targets[idx]))

class TrajectoryLSTM(nn.Module):
    """LSTM for vehicle trajectory prediction"""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_size)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def generate_synthetic_trajectories(n_samples=1000):
    """Generate synthetic vehicle trajectories for training"""
    print("🎲 Generating synthetic trajectory data...")
    trajectories = []
    
    for _ in range(n_samples):
        length = np.random.randint(20, 50)
        traj = []
        
        # Initial position
        x = np.random.randint(100, 500)
        y = np.random.randint(100, 300)
        vx = np.random.uniform(-3, 3)
        vy = np.random.uniform(2, 8)  # Moving downward
        
        for _ in range(length):
            # Add noise
            vx += np.random.normal(0, 0.5)
            vy += np.random.normal(0, 0.5)
            
            x += vx
            y += vy
            w = 80 + np.random.randint(-10, 10)
            h = 60 + np.random.randint(-10, 10)
            
            traj.append([x, y, x+w, y+h])
        
        trajectories.append(traj)
    
    return trajectories

def train_lstm():
    """Train LSTM trajectory predictor"""
    print("\\n" + "="*60)
    print("🔄 Training LSTM Trajectory Predictor")
    print("="*60)
    
    # Generate data
    trajectories = generate_synthetic_trajectories(2000)
    
    # Split
    train_size = int(0.8 * len(trajectories))
    train_data = trajectories[:train_size]
    val_data = trajectories[train_size:]
    
    # Datasets
    train_dataset = TrajectoryDataset(train_data, Config.LSTM_SEQ_LEN)
    val_dataset = TrajectoryDataset(val_data, Config.LSTM_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.LSTM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.LSTM_BATCH_SIZE)
    
    # Model
    model = TrajectoryLSTM(input_size=4, hidden_size=Config.LSTM_HIDDEN).to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LSTM_LR)
    
    # Training
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\\n🚀 Starting LSTM training on {Config.DEVICE}")
    
    for epoch in range(Config.LSTM_EPOCHS):
        model.train()
        train_loss = 0
        
        for seq, target in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            seq, target = seq.to(Config.DEVICE), target.to(Config.DEVICE)
            
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, target in val_loader:
                seq, target = seq.to(Config.DEVICE), target.to(Config.DEVICE)
                output = model(seq)
                val_loss += criterion(output, target).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{Config.LSTM_EPOCHS}], "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            model_path = os.path.join(Config.MODEL_DIR, 'lstm_tracker.pth')
            torch.save(model.state_dict(), model_path)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('LSTM Training Curves')
    plt.grid(True)
    plot_path = os.path.join(Config.OUTPUT_DIR, 'lstm_training_curves.png')
    plt.savefig(plot_path)
    
    print(f"\\n🎉 LSTM Training Complete! Best Val Loss: {best_loss:.4f}")
    return True

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline"""
    print("🚦 Traffic Violation Detection - Model Training")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model Dir: {Config.MODEL_DIR}")
    print(f"Data Dir: {Config.DATA_DIR}")
    print("="*60)
    
    # Train ResNet
    resnet_success = train_resnet()
    
    # Train LSTM
    lstm_success = train_lstm()
    
    # Summary
    print("\\n" + "="*60)
    print("📊 Training Summary")
    print("="*60)
    
    if resnet_success:
        print("✅ ResNet18 Helmet Classifier: Trained")
        print(f"   Model: {os.path.join(Config.MODEL_DIR, 'helmet_resnet18.pth')}")
    else:
        print("❌ ResNet18: Failed (check dataset)")
    
    if lstm_success:
        print("✅ LSTM Tracker: Trained")
        print(f"   Model: {os.path.join(Config.MODEL_DIR, 'lstm_tracker.pth')}")
    else:
        print("❌ LSTM: Failed")
    
    print("\\n🚀 You can now run: streamlit run app.py")

if __name__ == "__main__":
    main()