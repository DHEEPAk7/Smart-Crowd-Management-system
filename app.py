"""
app.py - Streamlit Web Application
Smart Crowd Management System Interface

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from collections import deque
import time
from PIL import Image

# Import custom modules
from models import initialize_models
from utils import (
    detect_people_yolo,
    estimate_density_csrnet,
    analyze_zones,
    predict_crowd_flow_lstm,
    generate_alerts,
    visualize_frame,
    create_density_heatmap,
    create_zone_chart,
    create_timeline_chart,
    calculate_statistics
)


# ==================== Page Configuration ====================
st.set_page_config(
    page_title="Smart Crowd Management System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stAlert {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2, h3 {
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Initialize Session State ====================
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=10)

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False


# ==================== Load Models ====================
@st.cache_resource
def load_all_models():
    """Load all AI models"""
    return initialize_models()


# ==================== Main Application ====================
def main():
    # Header
    st.title("🎯 Smart Crowd Management System")
    st.markdown("**YOLOv5 + CSRNet + LSTM Integration**")
    st.divider()
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading AI models... Please wait..."):
            csrnet, lstm, yolo, device = load_all_models()
            st.session_state.csrnet = csrnet
            st.session_state.lstm = lstm
            st.session_state.yolo = yolo
            st.session_state.device = device
            st.session_state.models_loaded = True
        st.success(f"✅ Models loaded successfully! Using: {device}")
    else:
        csrnet = st.session_state.csrnet
        lstm = st.session_state.lstm
        yolo = st.session_state.yolo
        device = st.session_state.device
    
    # ==================== Sidebar ====================
    st.sidebar.header("⚙️ Settings")
    
    input_source = st.sidebar.radio(
        "Input Source",
        ["📸 Upload Image", "🎥 Upload Video", "📹 Use Webcam"],
        index=0
    )
    
    st.sidebar.divider()
    
    density_threshold = st.sidebar.slider(
        "Density Alert Threshold",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Threshold for generating high density alerts"
    )
    
    st.sidebar.divider()
    
    st.sidebar.markdown("### 📊 Visualizations")
    show_heatmap = st.sidebar.checkbox("Show Density Heatmap", value=True)
    show_zones = st.sidebar.checkbox("Show Zone Analysis", value=True)
    show_timeline = st.sidebar.checkbox("Show Timeline", value=True)
    
    st.sidebar.divider()
    
    st.sidebar.markdown("### 🔧 Features")
    st.sidebar.markdown("""
    - ✅ YOLOv5 Person Detection
    - ✅ CSRNet Density Estimation
    - ✅ LSTM Flow Prediction
    - ✅ Zone Analysis (4 Quadrants)
    - ✅ Real-time Alerts
    - ✅ Statistical Dashboard
    """)
    
    st.sidebar.divider()
    
    st.sidebar.markdown("### 📖 How to Use")
    st.sidebar.markdown("""
    1. Select input source
    2. Upload file or start webcam
    3. View real-time analysis
    4. Check alerts and statistics
    """)
    
    # ==================== Main Content ====================
    
    # Upload Image Mode
    if "Upload Image" in input_source:
        process_image_mode(csrnet, lstm, yolo, device, density_threshold, 
                          show_heatmap, show_zones)
    
    # Upload Video Mode
    elif "Upload Video" in input_source:
        process_video_mode(csrnet, lstm, yolo, device, density_threshold,
                          show_heatmap, show_zones, show_timeline)
    
    # Webcam Mode
    else:
        process_webcam_mode(csrnet, lstm, yolo, device, density_threshold,
                           show_heatmap, show_zones, show_timeline)


# ==================== Image Processing Mode ====================
def process_image_mode(csrnet, lstm, yolo, device, density_threshold, 
                       show_heatmap, show_zones):
    """Process uploaded image"""
    st.markdown("### 📸 Upload an Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a crowd image for analysis"
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        with st.spinner("🔍 Analyzing image..."):
            detections = detect_people_yolo(frame, yolo)
            density_map, density_count = estimate_density_csrnet(frame, csrnet, device)
            zones = analyze_zones(detections, frame.shape[1], frame.shape[0])
            alerts = generate_alerts(len(detections), zones, density_threshold)
            vis_frame = visualize_frame(frame, detections, density_map)
            stats = calculate_statistics(detections, density_count, zones)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Total Count", stats['total_count'])
        with col2:
            st.metric("📊 Density Estimate", f"{stats['density_estimate']:.1f}")
        with col3:
            st.metric("⚠️ Active Alerts", len(alerts))
        with col4:
            st.metric("🎯 Avg Confidence", f"{stats['avg_confidence']:.2f}")
        
        # Display annotated image
        st.image(vis_frame, caption="Analysis Result", use_container_width=True)
        
        # Alerts section
        if alerts:
            st.warning("### 🚨 Active Alerts")
            for alert in alerts:
                st.error(alert)
        else:
            st.success("✅ No alerts - Normal crowd levels")
        
        # Additional visualizations
        if show_heatmap or show_zones:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if show_heatmap:
                    st.pyplot(create_density_heatmap(density_map))
            
            with viz_col2:
                if show_zones:
                    st.pyplot(create_zone_chart(zones))
        
        # Zone details
        with st.expander("📍 Zone Details"):
            st.json(zones)


# ==================== Video Processing Mode ====================
def process_video_mode(csrnet, lstm, yolo, device, density_threshold,
                       show_heatmap, show_zones, show_timeline):
    """Process uploaded video"""
    st.markdown("### 🎥 Upload a Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video...", 
        type=['mp4', 'avi', 'mov'],
        help="Upload a crowd video for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open('temp_video.mp4', 'wb') as f:
            f.write(uploaded_file.read())
        
        # Video processing
        cap = cv2.VideoCapture('temp_video.mp4')
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.info(f"📹 Video loaded: {total_frames} frames")
        
        # Placeholders
        stframe = st.empty()
        stats_placeholder = st.empty()
        alerts_placeholder = st.empty()
        charts_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            
            # Update progress
            progress_bar.progress(frame_count / total_frames)
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                detections = detect_people_yolo(frame, yolo)
                density_map, density_count = estimate_density_csrnet(frame, csrnet, device)
                zones = analyze_zones(detections, frame.shape[1], frame.shape[0])
                alerts = generate_alerts(len(detections), zones, density_threshold)
                
                # Update history
                current_stats = [len(detections), density_count, 
                               zones['Top Left'], zones['Bottom Right']]
                st.session_state.history.append(current_stats)
                
                # LSTM prediction
                prediction = predict_crowd_flow_lstm(st.session_state.history, lstm, device)
                
                vis_frame = visualize_frame(frame, detections, density_map)
                
                # Update display
                stframe.image(vis_frame, channels="RGB", use_container_width=True)
                
                # Stats
                with stats_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("👥 Total", len(detections))
                    with col2:
                        st.metric("📊 Density", f"{density_count:.1f}")
                    with col3:
                        st.metric("🎯 Frame", f"{frame_count}/{total_frames}")
                    with col4:
                        if prediction is not None:
                            st.metric("🔮 Predicted", f"{prediction[0]:.0f}")
                
                # Alerts
                if alerts:
                    with alerts_placeholder.container():
                        st.warning("### 🚨 Active Alerts")
                        for alert in alerts:
                            st.error(alert)
                
                # Charts
                if show_timeline and len(st.session_state.history) > 1:
                    with charts_placeholder.container():
                        st.pyplot(create_timeline_chart(list(st.session_state.history)))
            
            time.sleep(0.03)  # Control playback speed
        
        cap.release()
        progress_bar.progress(100)
        st.success("✅ Video processing complete!")


# ==================== Webcam Processing Mode ====================
def process_webcam_mode(csrnet, lstm, yolo, device, density_threshold,
                        show_heatmap, show_zones, show_timeline):
    """Process webcam stream"""
    st.markdown("### 📹 Webcam Mode")
    st.info("Click 'Start Webcam' to begin real-time analysis")
    
    run = st.checkbox('Start Webcam')
    
    FRAME_WINDOW = st.empty()
    stats_placeholder = st.empty()
    alerts_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check permissions.")
            return
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            detections = detect_people_yolo(frame, yolo)
            density_map, density_count = estimate_density_csrnet(frame, csrnet, device)
            zones = analyze_zones(detections, frame.shape[1], frame.shape[0])
            alerts = generate_alerts(len(detections), zones, density_threshold)
            
            # Update history
            current_stats = [len(detections), density_count, 
                           zones['Top Left'], zones['Bottom Right']]
            st.session_state.history.append(current_stats)
            
            # Predict
            prediction = predict_crowd_flow_lstm(st.session_state.history, lstm, device)
            
            vis_frame = visualize_frame(frame, detections, density_map)
            
            # Display
            FRAME_WINDOW.image(vis_frame, channels="RGB", use_container_width=True)
            
            # Stats
            with stats_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("👥 Total Count", len(detections))
                with col2:
                    st.metric("📊 Density", f"{density_count:.1f}")
                with col3:
                    st.metric("⚠️ Alerts", len(alerts))
                with col4:
                    if prediction is not None:
                        st.metric("🔮 Predicted", f"{prediction[0]:.0f}")
            
            # Alerts
            if alerts:
                with alerts_placeholder.container():
                    st.warning("🚨 Active Alerts")
                    for alert in alerts:
                        st.error(alert)
            
            time.sleep(0.1)  # Control frame rate
        
        cap.release()


# ==================== Run Application ====================
if __name__ == "__main__":
    main()
