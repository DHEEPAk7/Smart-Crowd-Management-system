"""
utils.py - Utility Functions
Contains detection, analysis, and visualization functions
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def find_available_camera(max_index=5):
    """
    Find first available camera
    
    Args:
        max_index: Maximum camera index to check
    
    Returns:
        int: Available camera index or -1 if none found
    """
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return -1


def list_available_cameras(max_index=5):
    """
    List all available cameras
    
    Args:
        max_index: Maximum camera index to check
    
    Returns:
        list: List of available camera indices
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def detect_people_yolo(frame, yolo_model):
        frame: numpy array (H, W, 3)
        yolo_model: YOLOv5 model or None
    
    Returns:
        list: List of detections [x1, y1, x2, y2, confidence]
    """
    if yolo_model is None:
        # Simulated detection for demo
        h, w = frame.shape[:2]
        num_people = np.random.randint(10, 40)
        detections = []
        
        for _ in range(num_people):
            x1 = np.random.randint(0, max(1, w-100))
            y1 = np.random.randint(0, max(1, h-150))
            x2 = x1 + np.random.randint(40, 80)
            y2 = y1 + np.random.randint(120, 180)
            conf = 0.70 + np.random.random() * 0.29
            detections.append([x1, y1, x2, y2, conf])
        
        return detections
    
    # Real YOLOv5 detection
    results = yolo_model(frame)
    detections = []
    
    # Filter for person class (class 0 in COCO)
    for det in results.xyxy[0]:
        if int(det[5]) == 0:  # person class
            detections.append(det[:5].cpu().numpy().tolist())
    
    return detections


def estimate_density_csrnet(frame, csrnet_model, device):
    """
    Estimate crowd density using CSRNet
    
    Args:
        frame: numpy array (H, W, 3)
        csrnet_model: CSRNet model
        device: torch device
    
    Returns:
        tuple: (density_map, estimated_count)
    """
    try:
        # Preprocess
        img = cv2.resize(frame, (640, 480))
        img = img.astype(np.float32) / 255.0
        
        # Normalize using ImageNet stats
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        img = img.to(device)
        
        # Inference with GPU memory optimization
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():  # Mixed precision for faster GPU inference
                    density_map = csrnet_model(img)
            else:
                density_map = csrnet_model(img)
        
        # Convert back to numpy with validation
        density_map = density_map.cpu().numpy()[0, 0]
        
        # Ensure valid output
        density_map = np.nan_to_num(density_map, nan=0.0, posinf=1.0, neginf=0.0)
        density_map = np.clip(density_map, 0, None)  # Ensure non-negative values
        
        estimated_count = float(np.sum(density_map))
        
        # Clean up GPU cache periodically
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return density_map, estimated_count
    
    except Exception as e:
        print(f"Error in CSRNet estimation: {e}")
        # Return fallback empty density map on error
        return np.zeros((480, 640), dtype=np.float32), 0.0


def analyze_zones(detections, width, height):
    """
    Divide frame into 4 zones and count people in each
    
    Args:
        detections: list of detections
        width: frame width
        height: frame height
    
    Returns:
        dict: Zone names to people count
    """
    zones = {
        'Top Left': 0,
        'Top Right': 0,
        'Bottom Left': 0,
        'Bottom Right': 0
    }
    
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if cx < width/2 and cy < height/2:
            zones['Top Left'] += 1
        elif cx >= width/2 and cy < height/2:
            zones['Top Right'] += 1
        elif cx < width/2 and cy >= height/2:
            zones['Bottom Left'] += 1
        else:
            zones['Bottom Right'] += 1
    
    return zones


def predict_crowd_flow_lstm(history, lstm_model, device):
    """
    Predict future crowd flow using LSTM
    
    Args:
        history: deque of historical statistics
        lstm_model: LSTM model
        device: torch device
    
    Returns:
        numpy array or None: Predicted statistics
    """
    if len(history) < 10:
        return None
    
    # Prepare sequence
    sequence = torch.FloatTensor(list(history)).unsqueeze(0)
    sequence = sequence.to(device)
    
    # Predict with GPU acceleration
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():  # Mixed precision for LSTM
                prediction = lstm_model(sequence)
        else:
            prediction = lstm_model(sequence)
    
    return prediction.cpu().numpy()[0]


def generate_alerts(total_count, zones, density_threshold=30):
    """
    Generate alerts based on crowd statistics
    
    Args:
        total_count: total number of people
        zones: dict of zone counts
        density_threshold: threshold for alerts
    
    Returns:
        list: List of alert messages
    """
    alerts = []
    
    # Overall density alerts
    if total_count > density_threshold:
        alerts.append(f"⚠️ HIGH DENSITY: {total_count} people detected")
    elif total_count > density_threshold * 0.7:
        alerts.append(f"⚡ Moderate crowd: {total_count} people")
    
    # Zone-specific alerts
    for zone, count in zones.items():
        if count > density_threshold / 2:
            alerts.append(f"🔴 Congestion in {zone}: {count} people")
    
    return alerts


def visualize_frame(frame, detections, density_map):
    """
    Draw visualizations on frame
    
    Args:
        frame: numpy array (H, W, 3)
        detections: list of detections
        density_map: density map from CSRNet
    
    Returns:
        numpy array: Annotated frame
    """
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]
    
    # Draw zone boundaries
    cv2.line(vis_frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
    cv2.line(vis_frame, (0, h//2), (w, h//2), (255, 255, 255), 2)
    
    # Add zone labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_frame, 'TL', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_frame, 'TR', (w-50, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_frame, 'BL', (10, h-10), font, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_frame, 'BR', (w-50, h-10), font, 0.7, (255, 255, 255), 2)
    
    # Overlay density heatmap with validation
    try:
        # Ensure density_map is valid and 2D
        if density_map is not None and density_map.size > 0:
            # Handle NaN/Inf values
            density_map_clean = np.nan_to_num(density_map, nan=0.0, posinf=255.0, neginf=0.0)
            
            # Ensure proper shape (2D)
            if density_map_clean.ndim == 2:
                # Resize to match frame size
                if density_map_clean.shape != (h, w):
                    density_resized = cv2.resize(density_map_clean, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    density_resized = density_map_clean
                
                # Normalize and apply colormap
                density_normalized = cv2.normalize(density_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                density_colored = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)
                
                # Blend with original frame
                vis_frame = cv2.addWeighted(vis_frame, 0.7, density_colored, 0.3, 0)
    except Exception as e:
        # If density map overlay fails, continue with frame without overlay
        print(f"Warning: Could not overlay density map: {e}")
    
    # Draw detections
    try:
        for det in detections:
            if len(det) >= 5:
                x1, y1, x2, y2, conf = det[:5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Validate coordinates
                if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                    # Bounding box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Confidence score
                    conf_val = float(conf) / 100.0 if float(conf) > 1 else float(conf)
                    cv2.putText(vis_frame, f'{conf_val:.2f}', (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except Exception as e:
        print(f"Warning: Error drawing detections: {e}")
    
    return vis_frame


def create_density_heatmap(density_map):
    """
    Create matplotlib heatmap visualization
    
    Args:
        density_map: 2D numpy array
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(density_map, cmap='jet', cbar=True, ax=ax, 
                cbar_kws={'label': 'Density'})
    ax.set_title("CSRNet Density Map", fontsize=14, fontweight='bold')
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    return fig


def create_zone_chart(zones):
    """
    Create zone distribution bar chart
    
    Args:
        zones: dict of zone counts
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b']
    bars = ax.bar(zones.keys(), zones.values(), color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_title("Zone Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Zone")
    ax.set_ylabel("People Count")
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def create_timeline_chart(history_data):
    """
    Create timeline of crowd count
    
    Args:
        history_data: list of historical statistics
    
    Returns:
        matplotlib figure or None
    """
    if not history_data:
        return None
    
    timestamps = list(range(len(history_data)))
    counts = [h[0] for h in history_data]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, counts, marker='o', linewidth=2, markersize=6, 
            color='#3b82f6', label='Total Count')
    ax.fill_between(timestamps, counts, alpha=0.3, color='#3b82f6')
    
    ax.set_title("Crowd Count Timeline", fontsize=14, fontweight='bold')
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("People Count")
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig


def calculate_statistics(detections, density_count, zones):
    """
    Calculate comprehensive statistics
    
    Args:
        detections: list of detections
        density_count: density estimate
        zones: zone distribution
    
    Returns:
        dict: Statistics dictionary
    """
    return {
        'total_count': len(detections),
        'density_estimate': density_count,
        'zones': zones,
        'avg_confidence': np.mean([d[4] for d in detections]) if detections else 0,
        'max_zone': max(zones.items(), key=lambda x: x[1])[0] if zones else None,
        'min_zone': min(zones.items(), key=lambda x: x[1])[0] if zones else None
    }
