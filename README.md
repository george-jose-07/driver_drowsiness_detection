```markdown
# Driver Drowsiness Detection System

This repository contains the implementation of a real-time deep learning system designed to monitor driver alertness. The system uses a Convolutional Neural Network (CNN) to analyze facial frames and classify them as "Alert" or "Drowsy". It is specifically optimized for deployment on edge devices like the Raspberry Pi.

## 🚀 Features
* **Custom CNN Architecture**: Features multiple convolutional layers with Batch Normalization and Dropout for robust feature extraction.
* **TFLite Optimization**: Includes Float16 quantization to ensure low-latency performance on resource-constrained hardware.
* **High Efficiency**: Processes 64x64 grayscale frames, making it suitable for high-FPS inference on mobile CPUs.
* **Hardware Support**: Optimized for Raspberry Pi Zero 2 W, 3B+, or 4.

## 📂 Dataset Structure
Organize your training data in the following directory structure:
```text
dataset/
├── Active Subjects/    # Images of alert drivers (Label: 0)
└── Fatigue Subjects/   # Images of drowsy drivers (Label: 1)

```

The training script is configured to load images from these specific folder names.

## ⚙️ Installation & Setup

### Training Environment (PC)

Install the dependencies required to train and evaluate the model:

```bash
pip install -r requirements.txt

```

### Inference Environment (Raspberry Pi)

On the Raspberry Pi, use the lightweight TFLite runtime to save system resources:

```bash
pip install tflite-runtime opencv-python-headless numpy

```

## 📈 Model Training

1. Open the provided Jupyter Notebook (`tflite_train.ipynb`).
2. The script performs the following automatically:
* **Hardware Check**: Detects if an NVIDIA GPU is available for acceleration.
* **Data Loading**: Reads and resizes images to 64x64 in grayscale.
* **Training**: Trains the CNN using an Adam optimizer and categorical cross-entropy loss.
* **Evaluation**: Generates a Confusion Matrix and Classification Report.
* **Export**: Saves the model as `model.h5` and an optimized `model.tflite`.



## 🍓 Raspberry Pi Implementation

The system is optimized for real-time inference on the **Raspberry Pi Zero 2 W**.

1. **Transfer Model**: Copy the generated `model.tflite` to your Raspberry Pi.
2. **Preprocessing**: Camera frames must be captured, converted to grayscale, and resized to 64x64 before inference.
3. **Hardware Deployment**: Use the Pi Camera Module or a USB webcam. The model size is approximately 2.20 MB, ensuring it fits easily within the Pi's memory.

## 📊 Results

* **Test Accuracy**: Achieved ~99.99% accuracy on the validation set.
* **Model Size**: The TFLite model is approximately 2.2 MB.
* **Performance**: Optimized for sub-100ms inference, providing immediate alerting capabilities when fatigue is detected.

```

```