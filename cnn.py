import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ✅ Set dataset path
DATASET_PATH = "E:\\FINAL\\DATASET_2"
IMG_SIZE = 128
FRAMES_PER_VIDEO = 30  # Use first 30 frames per video

X, y = [], []

def extract_hr(gt_path):
    """Extract heart rate from the ground_truth.txt file (second line)."""
    try:
        with open(gt_path, 'r') as file:
            lines = file.readlines()

        if len(lines) < 2:
            print(f"⚠️ Skipping {gt_path} - Insufficient data lines.")
            return None  # Skip this subject
        
        hr_values = lines[1].strip().split()  # Get second line and split values
        hr_values = [float(value) for value in hr_values]  # Convert to floats

        avg_hr = np.mean(hr_values)  # Average HR for consistency
        return avg_hr

    except Exception as e:
        print(f"❌ Error reading {gt_path}: {e}")
        return None


# ✅ Load and preprocess data
print("Extracting frames and HR data...")
for subject in os.listdir(DATASET_PATH):
    subject_path = os.path.join(DATASET_PATH, subject)
    video_path = os.path.join(subject_path, 'vid.avi')
    gt_path = os.path.join(subject_path, 'ground_truth.txt')

    if not os.path.exists(video_path) or not os.path.exists(gt_path):
        print(f"⚠️ Skipping {subject} - Missing video or ground truth file.")
        continue

    # Extract HR data
    hr_value = extract_hr(gt_path)
    if hr_value is None:
        print(f"⚠️ Skipping {subject} - Invalid HR data.")
        continue

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0  # Normalize
        frames.append(frame)
        count += 1
    cap.release()

    # Store valid data
    if len(frames) == FRAMES_PER_VIDEO:
        X.append(np.array(frames))
        y.append(hr_value)
    else:
        print(f"⚠️ Skipping {subject} - Insufficient frames ({len(frames)})")

X = np.array(X)
y = np.array(y)

print(f"✅ Total valid videos loaded: {len(X)}")

# ✅ Handle empty dataset case
if len(X) == 0:
    raise ValueError("❌ No valid data was loaded. Check dataset structure.")

# ✅ Normalize HR values
y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Reshape X to match Conv3D input (batch, time, height, width, channels)
X_train = X_train.reshape(-1, FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)
X_test = X_test.reshape(-1, FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)

# ✅ Build CNN model using Conv3D
model = Sequential([
    Conv3D(32, (3, 3, 3), activation='relu', input_shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling3D((2, 2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)  # Regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ✅ Train the model
print("Training model...")
history = model.fit(X_train, y_train, validation_split=0.1, epochs=25, batch_size=8)

# ✅ Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {mae:.4f}")

# ✅ Save model
model.save("rppg_heart_rate_cnn.h5")
print("✅ Model saved successfully.")
