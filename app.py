import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
import time
from pygame import mixer
import threading
import dlib
from scipy.spatial import distance
from collections import deque

# ===========================
# CONFIGURATION
# ===========================
MODEL_PATH = "model.h5"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
IMG_SIZE = 64
ALERT_SOUND_PATH = "alert.mp3"

# Thresholds
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
MAR_THRESHOLD = 0.6   # Mouth Aspect Ratio threshold
DROWSY_THRESHOLD = 0.7  # CNN model threshold
ALERT_TIME_THRESHOLD = 5.0  # Seconds of drowsiness before alert

# Facial landmark indices
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

# ===========================
# INITIALIZE AUDIO
# ===========================
try:
    mixer.init()
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False
    print("Audio mixer not available. Sound alerts disabled.")

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)

# ===========================
# CUSTOM CSS STYLING
# ===========================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .status-alert {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .alert-safe {
        background-color: #d4edda;
        color: #155724;
        border: 3px solid #28a745;
    }
    .alert-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 3px solid #dc3545;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #ffc107;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ===========================
# LOAD MODELS
# ===========================
@st.cache_resource
def load_models():
    """Load CNN model and dlib face detector/predictor."""
    try:
        # Load Keras model
        cnn_model = keras.models.load_model(MODEL_PATH)
        
        # Load dlib face detector and landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        
        return cnn_model, detector, predictor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


# ===========================
# EYE ASPECT RATIO (EAR)
# ===========================
def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio for drowsiness detection.
    
    Args:
        eye_points: Array of (x, y) coordinates for eye landmarks
    
    Returns:
        Eye Aspect Ratio value
    """
    # Compute euclidean distances between vertical eye landmarks
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    
    # Compute euclidean distance between horizontal eye landmarks
    C = distance.euclidean(eye_points[0], eye_points[3])
    
    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear


# ===========================
# MOUTH ASPECT RATIO (MAR)
# ===========================
def calculate_mar(mouth_points):
    """
    Calculate Mouth Aspect Ratio for yawning detection.
    
    Args:
        mouth_points: Array of (x, y) coordinates for mouth landmarks
    
    Returns:
        Mouth Aspect Ratio value
    """
    # Compute vertical distances
    A = distance.euclidean(mouth_points[2], mouth_points[10])  # 51, 59
    B = distance.euclidean(mouth_points[4], mouth_points[8])   # 53, 57
    
    # Compute horizontal distance
    C = distance.euclidean(mouth_points[0], mouth_points[6])   # 49, 55
    
    # Calculate MAR
    mar = (A + B) / (2.0 * C)
    return mar


# ===========================
# EXTRACT LANDMARKS
# ===========================
def shape_to_np(shape):
    """Convert dlib shape object to numpy array."""
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# ===========================
# PREPROCESS FRAME FOR CNN
# ===========================
def preprocess_frame(frame, img_size):
    """Preprocess frame for CNN model prediction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    normalized = resized / 255.0
    preprocessed = normalized.reshape(1, img_size, img_size, 1)
    return preprocessed


# ===========================
# PLAY ALERT SOUND
# ===========================
def play_alert_sound():
    """Play alert sound in a separate thread."""
    def play():
        try:
            if AUDIO_AVAILABLE:
                import os
                if os.path.exists(ALERT_SOUND_PATH):
                    mixer.music.load(ALERT_SOUND_PATH)
                    mixer.music.play()
                else:
                    # If no sound file, create a beep using winsound (Windows) or system beep
                    try:
                        import winsound
                        winsound.Beep(1000, 500)  # 1000 Hz for 500ms
                    except:
                        # For Linux/Mac
                        import os
                        os.system('echo -e "\a"')  # System beep
        except Exception as e:
            print(f"Audio error: {e}")
    
    thread = threading.Thread(target=play)
    thread.daemon = True
    thread.start()


# ===========================
# MAIN APPLICATION
# ===========================
def main():
    # Header
    st.markdown('<p class="main-header">🚗 Advanced Driver Drowsiness Detection System</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
        <div style='text-align: center; color: #555; margin-bottom: 30px;'>
            Multi-modal AI system using facial landmarks (EAR/MAR) + CNN for drowsiness detection
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models and facial landmark detector..."):
        cnn_model, face_detector, landmark_predictor = load_models()
    
    st.success("✅ All models loaded successfully!")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Detection Settings")
    
    st.sidebar.subheader("Thresholds")
    ear_threshold = st.sidebar.slider(
        "EAR Threshold (Eye Closure)",
        min_value=0.15,
        max_value=0.35,
        value=EAR_THRESHOLD,
        step=0.01,
        help="Lower values = more sensitive to eye closure"
    )
    
    mar_threshold = st.sidebar.slider(
        "MAR Threshold (Yawning)",
        min_value=0.4,
        max_value=0.8,
        value=MAR_THRESHOLD,
        step=0.05,
        help="Higher values = less sensitive to yawning"
    )
    
    cnn_threshold = st.sidebar.slider(
        "CNN Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=DROWSY_THRESHOLD,
        step=0.05
    )
    
    alert_time = st.sidebar.slider(
        "Alert Time Threshold (seconds)",
        min_value=2.0,
        max_value=10.0,
        value=ALERT_TIME_THRESHOLD,
        step=0.5,
        help="Duration of drowsiness before triggering alert"
    )
    
    st.sidebar.subheader("Display Options")
    show_landmarks = st.sidebar.checkbox("Show Facial Landmarks", value=True)
    show_metrics = st.sidebar.checkbox("Show Detection Metrics", value=True)
    enable_sound = st.sidebar.checkbox("Enable Sound Alerts", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
        **Detection Methods:**
        - 👁️ **EAR**: Eye closure detection
        - 😮 **MAR**: Yawning detection  
        - 🧠 **CNN**: Deep learning classifier
        
        **Alert triggers when:**
        - Any method detects drowsiness
        - For more than configured time
    """)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Detection Status")
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        warning_placeholder = st.empty()
    
    # Start/Stop buttons
    start_button = st.button("🎥 Start Detection", type="primary", use_container_width=True)
    stop_button = st.button("⏹️ Stop Detection", use_container_width=True)
    
    # Initialize session state
    if 'drowsy_start_time' not in st.session_state:
        st.session_state.drowsy_start_time = None
    if 'alert_triggered' not in st.session_state:
        st.session_state.alert_triggered = False
    if 'ear_history' not in st.session_state:
        st.session_state.ear_history = deque(maxlen=30)
    if 'mar_history' not in st.session_state:
        st.session_state.mar_history = deque(maxlen=30)
    
    # Camera capture
    if start_button:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check your camera connection.")
            return
        
        st.info("🔴 Detection is running... Click 'Stop Detection' to end.")
        
        # Main detection loop
        while True:
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_detector(gray, 0)
            
            # Initialize detection flags
            ear_drowsy = False
            mar_drowsy = False
            cnn_drowsy = False
            avg_ear = 0
            avg_mar = 0
            cnn_prob = 0
            
            # Process each face
            for face in faces:
                # Get facial landmarks
                landmarks = landmark_predictor(gray, face)
                landmarks = shape_to_np(landmarks)
                
                # Extract eye coordinates
                left_eye = landmarks[LEFT_EYE]
                right_eye = landmarks[RIGHT_EYE]
                mouth = landmarks[MOUTH]
                
                # Calculate EAR for both eyes
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Calculate MAR
                avg_mar = calculate_mar(mouth)
                
                # Store history
                st.session_state.ear_history.append(avg_ear)
                st.session_state.mar_history.append(avg_mar)
                
                # Check thresholds
                if avg_ear < ear_threshold:
                    ear_drowsy = True
                
                if avg_mar > mar_threshold:
                    mar_drowsy = True
                
                # Draw landmarks if enabled
                if show_landmarks:
                    for (x, y) in landmarks:
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Draw eye contours
                    cv2.polylines(frame, [left_eye], True, (0, 255, 255), 1)
                    cv2.polylines(frame, [right_eye], True, (0, 255, 255), 1)
                    cv2.polylines(frame, [mouth], True, (0, 255, 255), 1)
                
                # Draw face rectangle
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # CNN prediction
            preprocessed = preprocess_frame(frame, IMG_SIZE)
            prediction = cnn_model.predict(preprocessed, verbose=0)
            alert_prob = prediction[0][0]
            drowsy_prob = prediction[0][1]
            cnn_prob = drowsy_prob
            
            if drowsy_prob >= cnn_threshold:
                cnn_drowsy = True
            
            # Determine overall drowsiness status
            is_drowsy = ear_drowsy or mar_drowsy or cnn_drowsy
            
            # Track drowsiness duration
            if is_drowsy:
                if st.session_state.drowsy_start_time is None:
                    st.session_state.drowsy_start_time = time.time()
                    st.session_state.last_alert_time = 0
                
                drowsy_duration = time.time() - st.session_state.drowsy_start_time
                
                # Trigger alert if duration exceeds threshold
                if drowsy_duration >= alert_time and enable_sound:
                    # Play alert continuously every 1.5 seconds while drowsy
                    current_time = time.time()
                    if current_time - st.session_state.last_alert_time >= 1.5:
                        play_alert_sound()
                        st.session_state.last_alert_time = current_time
                        st.session_state.alert_triggered = True
            else:
                st.session_state.drowsy_start_time = None
                st.session_state.alert_triggered = False
                st.session_state.last_alert_time = 0
                drowsy_duration = 0
            
            # Determine status text
            if is_drowsy:
                status = "DROWSY"
                color = (0, 0, 255)
                
                # Determine cause
                causes = []
                if ear_drowsy:
                    causes.append("Eyes Closed")
                if mar_drowsy:
                    causes.append("Yawning")
                if cnn_drowsy:
                    causes.append("CNN Detected")
                cause_text = " | ".join(causes)
            else:
                status = "ALERT"
                color = (0, 255, 0)
                cause_text = "Driver Awake"
            
            # Draw status on frame
            cv2.putText(frame, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, cause_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if is_drowsy and drowsy_duration > 0:
                cv2.putText(frame, f"Duration: {drowsy_duration:.1f}s", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Display status
            if status == "ALERT":
                status_html = """
                    <div class='status-alert alert-safe'>
                        🚗✅ ALERT<br>Driver is Awake
                    </div>
                """
            else:
                status_html = f"""
                    <div class='status-alert alert-danger'>
                        😴⚠️ DROWSY<br>{cause_text}
                    </div>
                """
            
            status_placeholder.markdown(status_html, unsafe_allow_html=True)
            
            # Display metrics
            if show_metrics:
                metrics_html = f"""
                    <div class='metric-box'>
                        <h4>Detection Metrics</h4>
                        <p><strong>👁️ EAR:</strong> {avg_ear:.3f} (Threshold: {ear_threshold:.3f})</p>
                        <p><strong>😮 MAR:</strong> {avg_mar:.3f} (Threshold: {mar_threshold:.3f})</p>
                        <p><strong>🧠 CNN Drowsy:</strong> {cnn_prob*100:.2f}%</p>
                        <p><strong>🧠 CNN Alert:</strong> {alert_prob*100:.2f}%</p>
                    </div>
                """
                metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
            
            # Display warning if drowsy
            if is_drowsy and drowsy_duration > 0:
                warning_html = f"""
                    <div class='warning-box'>
                        <h4>⚠️ Drowsiness Detected!</h4>
                        <p><strong>Duration:</strong> {drowsy_duration:.1f}s / {alert_time:.1f}s</p>
                        <p><strong>Alert in:</strong> {max(0, alert_time - drowsy_duration):.1f}s</p>
                    </div>
                """
                warning_placeholder.markdown(warning_html, unsafe_allow_html=True)
            else:
                warning_placeholder.empty()
            
            # Check for stop button
            if stop_button:
                break
            
            # Small delay
            time.sleep(0.03)
        
        # Release camera
        cap.release()
        st.success("✅ Detection stopped.")
        
        # Reset session state
        st.session_state.drowsy_start_time = None
        st.session_state.alert_triggered = False


# ===========================
# ENTRY POINT
# ===========================
if __name__ == "__main__":
    main()