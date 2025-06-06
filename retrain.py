import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ✅ Set paths and constants
DATASET_PATH = "E:\\FINAL\\DATASET_2"
IMG_SIZE = 128
FRAMES_PER_VIDEO = 30
EPOCHS = 30
BATCH_SIZE = 8

X, y = [], []

def extract_hr(gt_path):
    """Extract heart rate from ground_truth.txt file."""
    try:
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        hr_values = list(map(float, lines[1].strip().split()))
        return np.mean(hr_values)
    except Exception as e:
        print(f"❌ Error reading {gt_path}: {e}")
        return None

# ✅ Load dataset
print("🔍 Loading dataset...")
for subject in os.listdir(DATASET_PATH):
    subject_path = os.path.join(DATASET_PATH, subject)
    video_path = os.path.join(subject_path, 'vid.avi')
    gt_path = os.path.join(subject_path, 'ground_truth.txt')

    if not os.path.exists(video_path) or not os.path.exists(gt_path):
        continue

    hr_value = extract_hr(gt_path)
    if hr_value is None:
        continue

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

    if len(frames) == FRAMES_PER_VIDEO:
        X.append(np.array(frames))
        y.append(hr_value)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples.")

if len(X) == 0:
    raise ValueError("❌ No valid data found!")

# ✅ Preprocess labels
y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)
X_test = X_test.reshape(-1, FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)

# ✅ Build Model
model = Sequential([
    Input(shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)),
    Conv3D(32, (3, 3, 3), activation='relu'),
    MaxPooling3D((2, 2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ✅ Callbacks
checkpoint_cb = ModelCheckpoint('best_rppg_model.h5', save_best_only=True)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ✅ Train
print("🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# ✅ Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"\n✅ Test MAE: {mae:.4f}")

# ✅ Save final model
model.save('rppg_heart_rate_cnn_final.h5')
print("✅ Final model saved as 'rppg_heart_rate_cnn_final.h5'")

# ✅ Plot Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
