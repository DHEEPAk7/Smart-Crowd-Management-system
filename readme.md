# Smart Crowd Management System

A comprehensive AI-powered crowd management system that integrates **YOLOv5**, **CSRNet**, and **LSTM** for real-time crowd detection, density estimation, and flow prediction.

## 🎯 Features

- **YOLOv5 Person Detection** - Real-time person detection with bounding boxes
- **CSRNet Density Estimation** - Crowd density heatmap generation
- **LSTM Flow Prediction** - Predict future crowd patterns
- **Zone Analysis** - Divide scene into 4 quadrants for detailed analysis
- **Real-time Alerts** - Automatic alerts for high density and congestion
- **Multiple Input Modes** - Support for images, videos, and webcam

## 📁 Project Structure

```
smart-crowd-management/
├── app.py              # Main Streamlit application
├── models.py           # Neural network architectures (CSRNet, LSTM)
├── utils.py            # Utility functions (detection, visualization)
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Installation

### Step 1: Clone or Download the Project

Create a new folder and save all files:
- `app.py`
- `models.py`
- `utils.py`
- `config.py`
- `requirements.txt`

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install streamlit torch torchvision opencv-python numpy pillow matplotlib seaborn pandas
```

## 🎮 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the System

#### 1. Upload Image Mode
- Click "Upload Image" in the sidebar
- Upload a crowd image (JPG, PNG)
- View instant analysis with detections, density map, and statistics

#### 2. Upload Video Mode
- Click "Upload Video" in the sidebar
- Upload a video file (MP4, AVI, MOV)
- Watch frame-by-frame analysis with real-time statistics

#### 3. Webcam Mode
- Click "Use Webcam" in the sidebar
- Click "Start Webcam" checkbox
- View live crowd analysis from your camera

### Settings

- **Density Alert Threshold**: Adjust the threshold for generating alerts (10-100)
- **Show Density Heatmap**: Toggle density visualization
- **Show Zone Analysis**: Toggle zone distribution chart
- **Show Timeline**: Toggle historical trend graph

## 📊 Outputs

### Metrics
- **Total Count**: Number of people detected
- **Density Estimate**: CSRNet crowd density estimation
- **Active Alerts**: Number of current alerts
- **Average Confidence**: Detection confidence score

### Visualizations
- **Annotated Frame**: Original frame with bounding boxes and zones
- **Density Heatmap**: Color-coded crowd density map
- **Zone Distribution**: Bar chart showing people per zone
- **Timeline Graph**: Historical crowd count trends

### Alerts
- **High Density Warning**: When total count exceeds threshold
- **Moderate Crowd Notice**: When approaching threshold
- **Zone Congestion Alert**: When specific zones are overcrowded

## 🔧 Configuration

Edit `config.py` to customize:
- Model parameters
- Alert thresholds
- Visualization settings
- Processing options

## 🤖 Models

### YOLOv5
- **Purpose**: Real-time person detection
- **Source**: Ultralytics YOLOv5s
- **Output**: Bounding boxes with confidence scores

### CSRNet
- **Purpose**: Crowd density estimation
- **Architecture**: VGG16 frontend + Dilated convolutions backend
- **Output**: Density map and estimated count

### LSTM
- **Purpose**: Crowd flow prediction
- **Architecture**: 2-layer LSTM network
- **Output**: Predicted future crowd statistics

## 📝 Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Webcam (for webcam mode)

## 🎨 System Architecture

```
Input (Image/Video/Webcam)
    ↓
YOLOv5 Detection → Bounding Boxes
    ↓
CSRNet Estimation → Density Map
    ↓
Zone Analysis → 4-Quadrant Distribution
    ↓
LSTM Prediction → Future Flow Forecast
    ↓
Alert Generation → Safety Warnings
    ↓
Visualization → Dashboard Display
```

## 💡 Tips

1. **Performance**: Use GPU for faster processing (automatically detected)
2. **Video Processing**: Larger videos will take longer; consider reducing resolution
3. **Webcam**: Ensure good lighting for better detection accuracy
4. **Alerts**: Adjust threshold based on your specific use case

## 🐛 Troubleshooting

### Webcam Not Working
- Check camera permissions
- Ensure no other application is using the camera
- Try refreshing the browser

### Slow Processing
- Enable GPU support
- Reduce video resolution
- Increase frame skip rate in config

### Import Errors
- Verify all dependencies are installed
- Check Python version (3.8+)
- Try reinstalling requirements

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Support

For issues and questions, please create an issue in the repository.

---

**Built with ❤️ using Streamlit, PyTorch, and OpenCV**
