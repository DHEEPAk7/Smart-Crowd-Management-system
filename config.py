"""
config.py - Configuration Settings
Contains all configuration parameters for the system
"""

# ==================== Model Configuration ====================
MODEL_CONFIG = {
    'csrnet': {
        'input_size': (640, 480),
        'channels': 3,
    },
    'lstm': {
        'input_size': 4,
        'hidden_size': 64,
        'num_layers': 2,
        'output_size': 4,
        'sequence_length': 10
    },
    'yolo': {
        'model_name': 'yolov5s',
        'confidence_threshold': 0.5,
        'person_class_id': 0  # COCO dataset person class
    }
}

# ==================== Processing Configuration ====================
PROCESSING_CONFIG = {
    'video_frame_skip': 5,  # Process every Nth frame
    'webcam_frame_delay': 0.1,  # Delay between frames (seconds)
    'max_history_length': 10,  # Maximum frames to keep in history
}

# ==================== Alert Configuration ====================
ALERT_CONFIG = {
    'density_threshold': 30,  # Default density threshold
    'congestion_threshold': 15,  # Per-zone congestion threshold
    'alert_types': {
        'high_density': '⚠️ HIGH DENSITY',
        'moderate_crowd': '⚡ Moderate crowd',
        'zone_congestion': '🔴 Congestion'
    }
}

# ==================== Visualization Configuration ====================
VISUALIZATION_CONFIG = {
    'bbox_color': (0, 255, 0),  # Green for bounding boxes
    'bbox_thickness': 2,
    'zone_line_color': (255, 255, 255),  # White for zone boundaries
    'zone_line_thickness': 2,
    'heatmap_alpha': 0.3,  # Transparency for density overlay
    'colormap': 'jet',  # Colormap for density visualization
    'font': 0,  # cv2.FONT_HERSHEY_SIMPLEX
    'font_scale': 0.5,
    'font_thickness': 2
}

# ==================== Zone Configuration ====================
ZONE_CONFIG = {
    'zones': ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right'],
    'zone_colors': ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b']
}

# ==================== File Configuration ====================
FILE_CONFIG = {
    'allowed_image_types': ['jpg', 'jpeg', 'png'],
    'allowed_video_types': ['mp4', 'avi', 'mov'],
    'temp_video_path': 'temp_video.mp4'
}

# ==================== Device Configuration ====================
DEVICE_CONFIG = {
    'use_cuda': True,  # Use CUDA if available
    'force_cpu': False  # Force CPU even if CUDA is available
}

# ==================== ImageNet Normalization ====================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
