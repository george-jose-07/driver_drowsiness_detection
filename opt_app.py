"""
Driver Drowsiness Detection - Raspberry Pi Zero 2W
- TFLite instead of Keras (3-5x faster, much less RAM)
- OpenCV Haar cascades instead of dlib (no 200MB dependency)
- RPi.GPIO for buzzer (BCM pin 18 by default)
- Skips frames to reduce CPU load
- Runs headless or with display
"""

import cv2
import numpy as np
import time
import threading
import argparse

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[WARN] RPi.GPIO not found — buzzer disabled")

# ── Config ──────────────────────────────────────────────────
MODEL_PATH    = "model.tflite"
IMG_SIZE      = 64
EAR_THRESHOLD = 0.25
CNN_THRESHOLD = 0.70
ALERT_SECONDS = 5.0
FRAME_SKIP    = 2        # run inference every 3rd frame
CAM_W, CAM_H  = 320, 240
BUZZER_PIN    = 18       # BCM GPIO pin connected to buzzer
# ────────────────────────────────────────────────────────────


def setup_gpio():
    if not GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)


def cleanup_gpio():
    if not GPIO_AVAILABLE:
        return
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    GPIO.cleanup()


def buzz(duration=1.0):
    def _b():
        if GPIO_AVAILABLE:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
        else:
            print("\a[ALERT] DROWSY!", flush=True)
    threading.Thread(target=_b, daemon=True).start()


class TFLiteModel:
    def __init__(self, path):
        self.interp = tflite.Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.inp  = self.interp.get_input_details()
        self.out  = self.interp.get_output_details()

    def predict(self, gray_resized):
        """gray_resized: (IMG_SIZE, IMG_SIZE) uint8"""
        x = gray_resized.astype(np.float32) / 255.0
        x = x.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        self.interp.set_tensor(self.inp[0]['index'], x)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out[0]['index'])[0]
        return float(out[0]), float(out[1])  # alert_prob, drowsy_prob


class FaceAnalyzer:
    """Lightweight face+eye analysis using OpenCV built-in cascades."""
    def __init__(self):
        self.face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")

    def analyze(self, gray, frame, draw):
        result = {"face": False, "ear": 1.0, "eyes_open": True}
        faces = self.face.detectMultiScale(
            gray, 1.1, 5, minSize=(80, 80))

        if len(faces) == 0:
            return result

        fx, fy, fw, fh = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        result["face"] = True
        if draw:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)

        # Detect eyes only in upper half of face
        eye_roi = gray[fy:fy+int(fh*0.55), fx:fx+fw]
        eyes = self.eye.detectMultiScale(eye_roi, 1.1, 4, minSize=(20, 20))

        if len(eyes) >= 2:
            ear = float(np.mean([e[3] / max(e[2], 1) for e in eyes[:2]]))
        elif len(eyes) == 1:
            ear = float(eyes[0][3] / max(eyes[0][2], 1))
        else:
            ear = 0.15  # eyes not found → likely closed

        result["ear"] = ear
        result["eyes_open"] = ear >= EAR_THRESHOLD

        if draw:
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(frame,
                    (fx+ex, fy+ey), (fx+ex+ew, fy+ey+eh), (0,255,255), 1)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--cam",        type=int,   default=0)
    parser.add_argument("--alert-time", type=float, default=ALERT_SECONDS)
    parser.add_argument("--cnn",        type=float, default=CNN_THRESHOLD)
    args = parser.parse_args()
    display = not args.no_display

    setup_gpio()

    print("[INFO] Loading TFLite model...")
    model    = TFLiteModel(MODEL_PATH)
    analyzer = FaceAnalyzer()

    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        cleanup_gpio()
        return

    drowsy_start  = None
    alert_fired   = False
    frame_n       = 0
    last_drowsy   = False
    last_ear      = 1.0
    last_cnn      = 0.0
    last_causes   = []

    print("[INFO] Running. Press Ctrl+C or 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_n += 1
            run_inference = (frame_n % (FRAME_SKIP + 1) == 0)

            if run_inference:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

                info = analyzer.analyze(gray, frame, draw=display)
                _, drowsy_prob = model.predict(resized)

                last_ear   = info["ear"]
                last_cnn   = drowsy_prob
                eye_drowsy = not info["eyes_open"]
                cnn_drowsy = drowsy_prob >= args.cnn

                last_causes = []
                if eye_drowsy:  last_causes.append("Eyes Closed")
                if cnn_drowsy:  last_causes.append("CNN")
                last_drowsy = eye_drowsy or cnn_drowsy

            # Timing
            if last_drowsy:
                if drowsy_start is None:
                    drowsy_start = time.time()
                drowsy_dur = time.time() - drowsy_start
                if drowsy_dur >= args.alert_time:
                    if not alert_fired:
                        print(f"[ALERT] Drowsy {drowsy_dur:.1f}s — buzzer!")
                        buzz(1.5)
                        alert_fired = True
                    elif int(drowsy_dur) % 3 == 0:
                        buzz(0.5)
            else:
                drowsy_start = None
                alert_fired  = False
                drowsy_dur   = 0

            status = "DROWSY" if last_drowsy else "ALERT"
            color  = (0, 0, 255) if last_drowsy else (0, 255, 0)

            if display:
                cv2.putText(frame, f"{status}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame,
                    " ".join(last_causes) if last_causes else "Awake",
                    (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame,
                    f"EAR:{last_ear:.2f} CNN:{last_cnn*100:.0f}%",
                    (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
                if last_drowsy and drowsy_dur > 0:
                    cv2.putText(frame,
                        f"Drowsy {drowsy_dur:.1f}s/{args.alert_time:.0f}s",
                        (8, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 1)
                cv2.imshow("Drowsiness - Pi", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if frame_n % 45 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] {status} | "
                          f"EAR={last_ear:.2f} CNN={last_cnn*100:.0f}%"
                          + (f" dur={drowsy_dur:.1f}s" if last_drowsy else ""))

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        cleanup_gpio()


if __name__ == "__main__":
    main()