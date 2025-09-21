import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import queue
from scipy.spatial import distance as dist
import dlib

# Set page configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .camera-container {
        border: 3px solid #1f77b4;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .status-active {
        background-color: #2ecc71;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .status-drowsy {
        background-color: #f1c40f;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .status-warning {
        background-color: #e74c3c;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Global variables for camera processing
frame_queue = queue.Queue(maxsize=10)
processed_frame_queue = queue.Queue(maxsize=10)

# Drowsiness detection parameters
EYE_AR_THRESHOLD = 0.25
DROWSINESS_THRESHOLD = 15

# dlib face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
# The shape_predictor_68_face_landmarks.dat file is required. This is a common
# dependency for dlib-based facial landmark detection.
# You must download this file and place it in the same directory as this script.
# The file can be found at: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except:
    st.warning("`shape_predictor_68_face_landmarks.dat` not found. Drowsiness detection will not work. Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and extract it to the same directory.")
    predictor = None

def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR)"""
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def load_sample_data():
    """Load sample preprocessed data"""
    try:
        X_train = np.load('X_train.npy', allow_pickle=True)
        y_train = np.load('y_train.npy', allow_pickle=True)
        X_test = np.load('X_test.npy', allow_pickle=True)
        y_test = np.load('y_test.npy', allow_pickle=True)
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        return None, None, None, None

def check_gpu_status():
    """Check GPU availability for TensorFlow"""
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    return gpu_available

def preprocess_frame(frame):
    """Preprocess a single frame for drowsiness detection"""
    resized = cv2.resize(frame, (224, 224))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_frame.astype(np.float32) / 255.0
    return normalized

def detect_face_landmarks(frame):
    """Detect face and landmarks for drowsiness analysis"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    return faces, gray

class VideoProcessor:
    def __init__(self):
        self.drowsiness_frames = 0
        self.drowsy_status = "Active"
        self.total_frames = 0
        self.total_drowsy_frames = 0
        self.dlib_predictor = predictor
        self.dlib_detector = detector
        self.facial_landmarks = dlib.get_frontal_face_detector()
        # Indices for the eyes in the 68-point facial landmark model
        self.right_eye_start, self.right_eye_end = 36, 42
        self.left_eye_start, self.left_eye_end = 42, 48

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.total_frames += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.dlib_detector(gray, 0)
        
        drowsiness_detected = False

        for rect in rects:
            shape = self.dlib_predictor(gray, rect)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            left_eye = shape[self.left_eye_start:self.left_eye_end]
            right_eye = shape[self.right_eye_start:self.right_eye_end]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EYE_AR_THRESHOLD:
                self.drowsiness_frames += 1
                if self.drowsiness_frames >= DROWSINESS_THRESHOLD:
                    self.drowsy_status = "Drowsy"
                    drowsiness_detected = True
                    self.total_drowsy_frames += 1
            else:
                self.drowsiness_frames = 0
                self.drowsy_status = "Active"
            
            # Draw facial landmarks and a rectangle around the face
            cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

            # Draw EAR on the image
            cv2.putText(img, "EAR: {:.2f}".format(avg_ear), (rect.left(), rect.top() - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw drowsiness status
        if drowsiness_detected:
            text_color = (0, 0, 255)  # Red
            status_text = "DROWSY!"
        else:
            text_color = (0, 255, 0)  # Green
            status_text = "Active"

        cv2.putText(img, status_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 4)

        # Store processed frame info in session state
        st.session_state['camera_stats'] = {
            'drowsiness_frames': self.drowsiness_frames,
            'drowsy_status': self.drowsy_status,
            'total_drowsy_frames': self.total_drowsy_frames,
            'total_frames': self.total_frames
        }
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def create_class_distribution_plot(y_train, y_test):
    """Create interactive class distribution plot"""
    # Count classes
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training Set', 'Test Set'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    labels = ['Active/Alert', 'Fatigue/Drowsy']
    colors = ['#2ecc71', '#e74c3c']
    
    # Training set
    fig.add_trace(
        go.Bar(x=labels, y=train_counts, name='Training', marker_color=colors),
        row=1, col=1
    )
    
    # Test set
    fig.add_trace(
        go.Bar(x=labels, y=test_counts, name='Testing', marker_color=colors, showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Class Distribution",
        height=400,
        showlegend=False
    )
    
    return fig

def display_sample_images(X_data, y_data, num_samples=8):
    """Display sample images from the dataset"""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    class_names = ['Active/Alert', 'Fatigue/Drowsy']
    
    # Get random sample indices
    indices = np.random.choice(len(X_data), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        axes[i].imshow(X_data[idx])
        axes[i].set_title(f'{class_names[y_data[idx]]}', 
                         color='green' if y_data[idx] == 0 else 'red',
                         fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def analyze_image_properties(X_data):
    """Analyze image properties"""
    # Calculate mean brightness for each image
    brightness = np.mean(X_data.reshape(X_data.shape[0], -1), axis=1)
    
    # Calculate contrast (standard deviation of pixel values)
    contrast = np.std(X_data.reshape(X_data.shape[0], -1), axis=1)
    
    return brightness, contrast

def create_image_properties_plot(X_train, y_train, X_test, y_test):
    """Create plot showing image properties analysis"""
    # Analyze properties
    train_brightness, train_contrast = analyze_image_properties(X_train)
    test_brightness, test_contrast = analyze_image_properties(X_test)
    
    # Create DataFrame for plotting
    train_df = pd.DataFrame({
        'brightness': train_brightness,
        'contrast': train_contrast,
        'label': ['Active' if y == 0 else 'Drowsy' for y in y_train],
        'split': 'Train'
    })
    
    test_df = pd.DataFrame({
        'brightness': test_brightness,
        'contrast': test_contrast,
        'label': ['Active' if y == 0 else 'Drowsy' for y in y_test],
        'split': 'Test'
    })
    
    combined_df = pd.concat([train_df, test_df])
    
    # Create scatter plot
    fig = px.scatter(
        combined_df, 
        x='brightness', 
        y='contrast',
        color='label',
        facet_col='split',
        title='Image Properties: Brightness vs Contrast',
        labels={'brightness': 'Average Brightness', 'contrast': 'Contrast (Std Dev)'},
        color_discrete_map={'Active': '#2ecc71', 'Drowsy': '#e74c3c'}
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Driver Drowsiness Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # System Status
    st.sidebar.subheader("🖥️ System Status")
    
    # Check GPU
    gpu_status = check_gpu_status()
    if gpu_status:
        st.sidebar.markdown('<div class="alert-box success-box">✅ GPU Available</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="alert-box warning-box">⚠️ Using CPU</div>', unsafe_allow_html=True)
    
    # TensorFlow version
    st.sidebar.write(f"TensorFlow: {tf.__version__}")
    st.sidebar.write(f"OpenCV: {cv2.__version__}")
    
    # Load data
    st.sidebar.subheader("📂 Data Loading")
    
    if st.sidebar.button("🔄 Load Sample Data"):
        with st.spinner("Loading preprocessed data..."):
            X_train, y_train, X_test, y_test = load_sample_data()
            
            if X_train is not None:
                st.session_state['data_loaded'] = True
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.sidebar.success("✅ Data loaded successfully!")
            else:
                st.sidebar.error("❌ No preprocessed data found. Please run the main script first.")
    
    # Main content
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Training Images", len(X_train))
        
        with col2:
            st.metric("Total Test Images", len(X_test))
        
        with col3:
            active_count = np.sum(y_train == 0) + np.sum(y_test == 0)
            st.metric("Active/Alert Images", active_count)
        
        with col4:
            drowsy_count = np.sum(y_train == 1) + np.sum(y_test == 1)
            st.metric("Fatigue/Drowsy Images", drowsy_count)
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🖼️ Sample Images", "📈 Image Analysis", "📹 Live Camera Demo"])
        
        with tab1:
            st.subheader("Dataset Distribution")
            
            # Class distribution plot
            fig_dist = create_class_distribution_plot(y_train, y_test)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Dataset statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Set Statistics")
                train_active = np.sum(y_train == 0)
                train_drowsy = np.sum(y_train == 1)
                st.write(f"🟢 Active/Alert: {train_active} ({train_active/len(y_train)*100:.1f}%)")
                st.write(f"🔴 Fatigue/Drowsy: {train_drowsy} ({train_drowsy/len(y_train)*100:.1f}%)")
            
            with col2:
                st.subheader("Test Set Statistics")
                test_active = np.sum(y_test == 0)
                test_drowsy = np.sum(y_test == 1)
                st.write(f"🟢 Active/Alert: {test_active} ({test_active/len(y_test)*100:.1f}%)")
                st.write(f"🔴 Fatigue/Drowsy: {test_drowsy} ({test_drowsy/len(y_test)*100:.1f}%)")
        
        with tab2:
            st.subheader("Sample Images from Dataset")
            
            dataset_choice = st.selectbox("Choose dataset:", ["Training Set", "Test Set"])
            
            if dataset_choice == "Training Set":
                fig_samples = display_sample_images(X_train, y_train)
            else:
                fig_samples = display_sample_images(X_test, y_test)
            
            st.pyplot(fig_samples)
            
            # Individual image viewer
            st.subheader("Individual Image Viewer")
            
            if dataset_choice == "Training Set":
                img_idx = st.slider("Select image index:", 0, len(X_train)-1, 0)
                selected_img = X_train[img_idx]
                selected_label = y_train[img_idx]
            else:
                img_idx = st.slider("Select image index:", 0, len(X_test)-1, 0)
                selected_img = X_test[img_idx]
                selected_label = y_test[img_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(selected_img, caption=f"Label: {'Active/Alert' if selected_label == 0 else 'Fatigue/Drowsy'}")
            
            with col2:
                # Image statistics
                st.subheader("Image Statistics")
                st.write(f"Shape: {selected_img.shape}")
                st.write(f"Min pixel value: {selected_img.min():.3f}")
                st.write(f"Max pixel value: {selected_img.max():.3f}")
                st.write(f"Mean pixel value: {selected_img.mean():.3f}")
                st.write(f"Std pixel value: {selected_img.std():.3f}")
        
        with tab3:
            st.subheader("Image Properties Analysis")
            
            # Create properties plot
            fig_props = create_image_properties_plot(X_train, y_train, X_test, y_test)
            st.plotly_chart(fig_props, use_container_width=True)
            
            # Brightness and contrast histograms
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Brightness Distribution")
                brightness_train, _ = analyze_image_properties(X_train)
                brightness_test, _ = analyze_image_properties(X_test)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(brightness_train, alpha=0.7, label='Training', bins=30)
                ax.hist(brightness_test, alpha=0.7, label='Test', bins=30)
                ax.set_xlabel('Average Brightness')
                ax.set_ylabel('Frequency')
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                st.subheader("Contrast Distribution")
                _, contrast_train = analyze_image_properties(X_train)
                _, contrast_test = analyze_image_properties(X_test)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(contrast_train, alpha=0.7, label='Training', bins=30)
                ax.hist(contrast_test, alpha=0.7, label='Test', bins=30)
                ax.set_xlabel('Contrast (Std Dev)')
                ax.set_ylabel('Frequency')
                ax.legend()
                st.pyplot(fig)
        
        with tab4:
            st.subheader("🎥 Live Camera Preprocessing Demo")
            
            st.markdown("""
            **Real-time drowsiness detection preprocessing using your camera!**
            
            This demo shows:
            - ✅ Face detection and eye landmark tracking in real-time
            - ✅ Eye Aspect Ratio (EAR) calculation
            - ✅ Live drowsiness status and alerts
            """)
            
            # Camera controls
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="camera-container">', unsafe_allow_html=True)
                
                rtc_configuration = RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                })
                
                # Create video processor instance
                webrtc_ctx = webrtc_streamer(
                    key="drowsiness-detection",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_configuration,
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.subheader("📊 Live Statistics")
                
                status_placeholder = st.empty()
                stats_placeholder = st.empty()

                if webrtc_ctx.state.playing:
                    while True:
                        if 'camera_stats' in st.session_state:
                            stats = st.session_state['camera_stats']
                            
                            with status_placeholder:
                                status = stats.get('drowsy_status', 'Active')
                                if status == "Active":
                                    st.markdown(f'<div class="status-active">AWAKE!</div>', unsafe_allow_html=True)
                                elif status == "Drowsy":
                                    st.markdown(f'<div class="status-warning">DROWSY!</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="status-active">AWAKE!</div>', unsafe_allow_html=True)

                            with stats_placeholder.container():
                                st.metric("Frames Since Last Blink", stats.get('drowsiness_frames', 0))
                                st.metric("Drowsy Frames", stats.get('total_drowsy_frames', 0))
                                st.metric("Total Frames Processed", stats.get('total_frames', 0))
                        
                        time.sleep(0.5)
                
                st.subheader("📝 Instructions")
                st.markdown("""
                1. **Click 'START'** to begin camera feed
                2. **Keep your eyes on the screen.** The system will track your eyes.
                3. **If your eyes close for more than a few seconds**, the status will change to `DROWSY!`.
                4. The system needs the `shape_predictor_68_face_landmarks.dat` file to run correctly.
                """)
            
            st.subheader("💡 Tips for Best Results")
            st.markdown("""
            - **Good lighting**: Ensure your face is well-lit.
            - **Stable position**: Keep your face centered in the frame.
            - **Clear background**: Minimize distractions behind you.
            """)
    
    else:
        # Welcome message when no data is loaded
        st.markdown("""
        ## Welcome to Driver Drowsiness Detection System! 👋
        
        This application helps visualize and analyze the drowsiness detection dataset with **live camera integration**.
        
        ### 🚀 Getting Started:
        
        1. **Load sample data** using the sidebar button.
        
        2. **Explore the different tabs** to analyze your dataset.
        
        3. **Try the Live Camera Demo** for real-time drowsiness detection!
        
        ### 🎥 New Live Camera Features:
        - **Real-time drowsiness detection** using your webcam.
        - **Eye Aspect Ratio (EAR)** calculation.
        - **Drowsiness status** and alert display.
        
        ### 📊 Features:
        - **Dataset Overview**: Visualize class distribution and statistics
        - **Sample Images**: Browse through preprocessed images
        - **Image Analysis**: Analyze brightness, contrast, and other properties
        - **Live Camera Demo**: Real-time drowsiness detection
        
        ### 🎯 Dataset Structure:
        - **Active Subjects**: Alert/awake driver images
        - **Fatigue Subjects**: Drowsy/tired driver images
        - **Target**: Binary classification (0: Active, 1: Drowsy)
        """)
        
        # System requirements
        st.subheader("📋 System Requirements")
        st.markdown("""
        **For Camera Features:**
        ```bash
        pip install streamlit-webrtc dlib
        pip install aiortc
        pip install opencv-python numpy scipy
        ```
        
        **Core Requirements:**
        - **Python 3.7+**
        - **TensorFlow 2.x**
        - **OpenCV**
        - **Streamlit**
        - **NumPy, Matplotlib, Seaborn**
        - **Webcam access** (for live demo)
        - **dlib** and the `shape_predictor_68_face_landmarks.dat` file.
        - **GPU support** (recommended for training)
        """)

if __name__ == "__main__":
    main()
