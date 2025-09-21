import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
def check_gpu():
    print("=" * 50)
    print("GPU AVAILABILITY CHECK")
    print("=" * 50)
    
    # Check TensorFlow GPU
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU is available and will be used for training!")
        
        # Get GPU details
        for gpu in tf.config.list_physical_devices('GPU'):
            print(f"GPU Device: {gpu}")
            
        # Set memory growth to avoid allocation issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU memory growth enabled")
            except RuntimeError as e:
                print(f"❌ GPU memory growth error: {e}")
    else:
        print("❌ No GPU available. Training will use CPU.")
    
    print("=" * 50)

class DrowsinessDetectionPreprocessor:
    def __init__(self, active_path, fatigue_path, img_size=(224, 224)):
        self.active_path = Path(active_path)
        self.fatigue_path = Path(fatigue_path)
        self.img_size = img_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def detect_faces_and_eyes(self, image):
        """Detect faces and eyes in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_detected = len(faces) > 0
        eyes_detected = False
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                eyes_detected = True
                break
        
        return face_detected, eyes_detected, faces
    
    def preprocess_image(self, image_path):
        """Enhanced image preprocessing pipeline"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: Could not load image {image_path}")
                return None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Face and eye detection
            face_detected, eyes_detected, faces = self.detect_faces_and_eyes(img)
            
            # If face detected, crop to face region
            if face_detected and len(faces) > 0:
                x, y, w, h = faces[0]  # Use the first detected face
                img_rgb = img_rgb[y:y+h, x:x+w]
            
            # Resize image
            img_resized = cv2.resize(img_rgb, self.img_size)
            
            # Normalize pixel values
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Additional preprocessing
            img_enhanced = self.enhance_image(img_normalized)
            
            return img_enhanced, face_detected, eyes_detected
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, False, False
    
    def enhance_image(self, image):
        """Apply image enhancement techniques"""
        # Convert to grayscale for some operations, then back to RGB
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Apply histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        enhanced_rgb = enhanced_rgb.astype(np.float32) / 255.0
        
        # Combine original and enhanced (weighted average)
        final_image = 0.7 * image + 0.3 * enhanced_rgb
        
        return np.clip(final_image, 0, 1)
    
    def load_dataset(self):
        """Load and preprocess the complete dataset"""
        print("Loading dataset...")
        print(f"Active subjects path: {self.active_path}")
        print(f"Fatigue subjects path: {self.fatigue_path}")
        
        images = []
        labels = []
        metadata = []
        
        # Check if paths exist
        if not self.active_path.exists():
            print(f"❌ Active subjects path does not exist: {self.active_path}")
            return None, None, None
            
        if not self.fatigue_path.exists():
            print(f"❌ Fatigue subjects path does not exist: {self.fatigue_path}")
            return None, None, None
        
        # Load Active (Alert) images
        active_files = list(self.active_path.glob("**/*.jpg")) + list(self.active_path.glob("**/*.png")) + list(self.active_path.glob("**/*.jpeg"))
        print(f"Found {len(active_files)} active/alert images")
        
        for i, img_path in enumerate(active_files):
            result = self.preprocess_image(img_path)
            if result[0] is not None:
                images.append(result[0])
                labels.append(0)  # 0 for active/alert
                metadata.append({
                    'path': str(img_path),
                    'label': 'active',
                    'face_detected': result[1],
                    'eyes_detected': result[2]
                })
            
            if i % 100 == 0:
                print(f"Processed {i+1}/{len(active_files)} active images")
        
        # Load Fatigue (Drowsy) images
        fatigue_files = list(self.fatigue_path.glob("**/*.jpg")) + list(self.fatigue_path.glob("**/*.png")) + list(self.fatigue_path.glob("**/*.jpeg"))
        print(f"Found {len(fatigue_files)} fatigue/drowsy images")
        
        for i, img_path in enumerate(fatigue_files):
            result = self.preprocess_image(img_path)
            if result[0] is not None:
                images.append(result[0])
                labels.append(1)  # 1 for fatigue/drowsy
                metadata.append({
                    'path': str(img_path),
                    'label': 'fatigue',
                    'face_detected': result[1],
                    'eyes_detected': result[2]
                })
            
            if i % 100 == 0:
                print(f"Processed {i+1}/{len(fatigue_files)} fatigue images")
        
        print(f"\n📊 Dataset loaded successfully!")
        print(f"Total images: {len(images)}")
        print(f"Active/Alert images: {sum(1 for label in labels if label == 0)}")
        print(f"Fatigue/Drowsy images: {sum(1 for label in labels if label == 1)}")
        
        return np.array(images), np.array(labels), metadata

def visualize_dataset_info(images, labels, metadata):
    """Visualize dataset information"""
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    
    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ['Active/Alert', 'Fatigue/Drowsy']
    
    plt.figure(figsize=(15, 5))
    
    # Class distribution plot
    plt.subplot(1, 3, 1)
    plt.bar(class_names, counts, color=['green', 'red'], alpha=0.7)
    plt.title('Class Distribution')
    plt.ylabel('Number of Images')
    
    # Face detection statistics
    face_detected = sum(1 for meta in metadata if meta['face_detected'])
    eyes_detected = sum(1 for meta in metadata if meta['eyes_detected'])
    
    plt.subplot(1, 3, 2)
    detection_stats = [face_detected, eyes_detected, len(metadata)]
    detection_labels = ['Faces Detected', 'Eyes Detected', 'Total Images']
    plt.bar(detection_labels, detection_stats, color=['blue', 'orange', 'gray'], alpha=0.7)
    plt.title('Detection Statistics')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Sample images
    plt.subplot(1, 3, 3)
    if len(images) > 0:
        sample_idx = np.random.randint(0, len(images))
        plt.imshow(images[sample_idx])
        plt.title(f'Sample Image\nLabel: {class_names[labels[sample_idx]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print(f"Face Detection Rate: {face_detected/len(metadata)*100:.1f}%")
    print(f"Eye Detection Rate: {eyes_detected/len(metadata)*100:.1f}%")
    print(f"Class Balance: {counts[0]/(counts[0]+counts[1])*100:.1f}% Active, {counts[1]/(counts[0]+counts[1])*100:.1f}% Fatigue")

def create_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    """Create a simple CNN model for drowsiness detection"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    # Check GPU availability
    check_gpu()
    
    # Initialize preprocessor
    print("\n🚀 Initializing Driver Drowsiness Detection System...")
    
    active_path = r"D:\my projects\ddd\dataset\Active Subjects"
    fatigue_path = r"D:\my projects\ddd\dataset\Fatigue Subjects"
    
    preprocessor = DrowsinessDetectionPreprocessor(active_path, fatigue_path)
    
    # Load and preprocess dataset
    print("\n📂 Loading and preprocessing dataset...")
    images, labels, metadata = preprocessor.load_dataset()
    
    if images is None:
        print("❌ Failed to load dataset. Please check your paths.")
        return
    
    # Visualize dataset information
    visualize_dataset_info(images, labels, metadata)
    
    # Split dataset
    print("\n✂️ Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Testing set: {len(X_test)} images")
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, 2)
    y_test_cat = keras.utils.to_categorical(y_test, 2)
    
    # Create model
    print("\n🏗️ Creating CNN model...")
    model = create_cnn_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Save preprocessed data for Streamlit app
    print("\n💾 Saving preprocessed data for visualization...")
    np.save('X_train.npy', X_train[:100])  # Save first 100 for demo
    np.save('y_train.npy', y_train[:100])
    np.save('X_test.npy', X_test[:50])  # Save first 50 for demo
    np.save('y_test.npy', y_test[:50])
    

if __name__ == "__main__":
    main()