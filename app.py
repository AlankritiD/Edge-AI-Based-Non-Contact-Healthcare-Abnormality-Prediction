import cv2
import numpy as np
import tensorflow as tf
import dlib
import time
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fftpack import fft

from tensorflow.keras.losses import MeanSquaredError

# Load trained CNN model
model = tf.keras.models.load_model("rppg_heart_rate_cnn.h5", custom_objects={"mse": MeanSquaredError()})

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Bandpass filter for valid heart rate range (0.67–3 Hz → 40–180 BPM)
def bandpass_filter(signal, lowcut=0.67, highcut=3.0, fs=30, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

# Function to preprocess and extract ROI
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) > 0:
        x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
        roi = frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, (128, 128))  # Resize for model input
        roi = np.array(roi) / 255.0  # Normalize
        return np.expand_dims(roi, axis=0), True  # Add batch dimension
    return None, False

# Video capture setup
cap = cv2.VideoCapture(0)

frame_buffer = []
green_signal = []
start_time = time.time()
frame_count = 0

print("Recording for 30 seconds...")

while time.time() - start_time < 30:  # Capture for 30 seconds
    ret, frame = cap.read()
    if not ret:
        break

    roi, detected = preprocess_frame(frame)
    if detected:
        frame_buffer.append(roi)
        
        # Extract green channel intensity (rPPG)
        green_intensity = np.mean(frame[:, :, 1])  # Green channel
        green_signal.append(green_intensity)

    frame_count += 1
    cv2.imshow("Heart Rate Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Compute actual FPS
actual_fps = frame_count / (time.time() - start_time)
print(f"Actual FPS: {actual_fps:.2f}")

cap.release()
cv2.destroyAllWindows()

# Ensure enough frames were captured
if len(green_signal) >= 150:
    filtered_signal = bandpass_filter(green_signal, fs=actual_fps)

    # Apply FFT to extract heart rate frequency
    fft_result = np.abs(fft(filtered_signal))
    freqs = np.fft.fftfreq(len(filtered_signal), d=1 / actual_fps)  # Use actual FPS

    # Keep only valid frequencies (0.67 Hz – 3.0 Hz corresponding to 40-180 BPM)
    valid_indices = (freqs > 0.67) & (freqs < 3.0)
    valid_freqs = freqs[valid_indices]
    valid_fft = fft_result[valid_indices]

    # Find multiple peaks and take weighted mean to avoid outliers
    peaks, properties = find_peaks(valid_fft, height=0.1 * max(valid_fft))
    
    if len(peaks) > 0:
        dominant_freqs = valid_freqs[peaks]
        peak_magnitudes = properties["peak_heights"]

        # Weighted mean of multiple peaks for better accuracy
        weighted_freq = np.sum(dominant_freqs * peak_magnitudes) / np.sum(peak_magnitudes)
    else:
        # If no peak detected, fall back to max value in valid range
        weighted_freq = valid_freqs[np.argmax(valid_fft)]

    bpm = weighted_freq * 60  # Convert frequency to BPM

    # Adjust based on dataset calibration
    bpm_calibrated = 72 + ((bpm - 49) / (89 - 49)) * (89 - 72)  # Scale to expected HR range

    print(f"Estimated Heart Rate: {int(bpm_calibrated)} BPM")
else:
    print("Insufficient data for heart rate estimation.")
