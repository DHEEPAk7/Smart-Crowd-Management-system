"""
models.py - Neural Network Models
Contains CSRNet and LSTM architectures
"""

import torch
import torch.nn as nn

class CSRNet(nn.Module):
    """
    CSRNet for Crowd Density Estimation
    Paper: CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
    """
    def __init__(self):
        super(CSRNet, self).__init__()
        
        # Frontend - VGG16 first 10 conv layers
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Backend - Dilated convolutions
        self.backend = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output(x)
        return x


class CrowdLSTM(nn.Module):
    """
    LSTM for Crowd Flow Prediction
    Predicts future crowd statistics based on historical data
    """
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=4):
        super(CrowdLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def load_yolo_model(device):
    """
    Load YOLOv5 model for person detection
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(device)
        model.eval()
        print("✓ YOLOv5 loaded successfully")
        return model
    except Exception as e:
        print(f"Warning: Could not load YOLOv5: {e}")
        print("Using simulated detection mode")
        return None


def initialize_models(device=None):
    """
    Initialize all models and move to device
    
    Args:
        device: torch.device or None (auto-detect)
    
    Returns:
        tuple: (csrnet, lstm, yolo, device)
    """
    if device is None:
        # Check GPU availability
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device('cpu')
            print("⚠️ No GPU detected. Using CPU (slower)")
    
    print(f"Initializing models on {device}...")
    
    # Enable GPU optimizations if using CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("✓ GPU optimizations enabled")
    
    # CSRNet
    csrnet = CSRNet().to(device)
    csrnet.eval()
    print("✓ CSRNet initialized on", device)
    
    # LSTM
    lstm = CrowdLSTM().to(device)
    lstm.eval()
    print("✓ LSTM initialized on", device)
    
    # YOLOv5
    yolo = load_yolo_model(device)
    
    return csrnet, lstm, yolo, device
