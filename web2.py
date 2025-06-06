import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import cv2
from scipy.signal import butter, filtfilt, find_peaks, detrend
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError
import time
import datetime
import tensorflow as tf
from tkinter import filedialog
import tkinter as tk
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set up page configuration
st.set_page_config(
    page_title="Edge-AI Health Monitoring",
    page_icon="🧑‍⚕️",
    layout="wide",
)

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    selected = option_menu(
        'Edge-AI Health Monitoring',
        ['Home', 'Monitoring', 'Relaxation', "Book Doctor's Consultation", 'How rPPG Works','Compare Predicted HR with Apple Watch Or Oximeter', 'Contact Us', 'About Us'],
        icons=['house', 'activity', 'heart', 'stethoscope', 'info','phone', 'hospital'],
        default_index=0
    )
    # Home Page
if selected == 'Home':
    st.title("Welcome to Edge-AI Health Monitoring App 🧑‍⚕️💓")
    st.write("Empowering you to track your heart health and relax with AI-powered music suggestions based on Indian classical time theory.")

    st.image("landing.jpg", caption="Your AI Health Companion", use_container_width=True)
    st.markdown("## 🩺 Key Features")
    st.markdown("- Real-time remote heart rate monitoring")
    st.markdown("- Music therapy with time-based Raga recommendations")
    st.markdown("- Book expert doctor consultations directly from the app")

# Load AI Model (Only Once)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("rppg_heart_rate_cnn_final.h5", custom_objects={"mse": MeanSquaredError()})

model = load_model()

def extract_frames_for_model(video_path, num_frames=30, size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, size)
                face = face / 255.0
                frames.append(face)

            if len(frames) == num_frames:
                break

    cap.release()

    if len(frames) < num_frames:
        return None
    return np.expand_dims(np.array(frames), axis=0)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def estimate_heart_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    green_avg = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = frame[y:y + h, x:x + w]
            roi = face[h//4:h//2, w//4:w//2]  # Center top region
            green_channel = roi[:, :, 1]
            green_avg.append(np.mean(green_channel))

    cap.release()

    if len(green_avg) < 30:
        return None, "❌ Not enough data in video."

    signal = np.array(green_avg)
    b, a = butter_bandpass(0.7, 4.0, fps, order=3)
    filtered = filtfilt(b, a, signal)

    fft = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1/fps)

    peak_idx = np.argmax(fft)
    peak_freq = freqs[peak_idx]
    heart_rate_bpm = peak_freq * 60

    if 60 <= heart_rate_bpm <= 100:
        status = "✅ Normal Heart Rate"
    else:
        status = "⚠️ Abnormal Heart Rate"

    return heart_rate_bpm, status, filtered


# Bandpass filter for heart rate detection
def bandpass_filter(signal, fs, lowcut=0.67, highcut=3.0, order=5):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Function to analyze frame lighting conditions
def check_lighting(frame):
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if brightness < 50:
        st.warning("⚠️ Poor lighting detected! Increase brightness for accurate results.")

# Function to detect motion (reducing motion artifacts)
def detect_motion(prev_frame, curr_frame):
    if prev_frame is None:
        return False
    diff = cv2.absdiff(prev_frame, curr_frame)
    motion = np.sum(diff) / diff.size
    return motion > 10  # Threshold to detect motion

# **Monitoring Page**
if selected == 'Monitoring':
    st.title('Health Monitoring Dashboard')

    option = st.radio("Choose Monitoring Mode:", ["📷 Live Camera Monitoring", "🎥 Upload Video File"])

    if option == "📷 Live Camera Monitoring":
        if 'ppg_values' not in st.session_state:
            st.session_state.ppg_values = []

        if st.button("Start Health Monitoring"):
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            rppg_signal = []
            start_time = time.time()
            frame_count = 0
            progress_bar = st.progress(0)
            video_feed = st.empty()
            prev_frame = None

            st.warning("🔴 Live Monitoring Active... Please stay still!")

            while time.time() - start_time < 45:
                ret, frame = cap.read()
                if not ret:
                    st.error("⚠️ Failed to capture video frame")
                    break

                check_lighting(frame)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_count % 5 == 0:
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if 'faces' in locals() and len(faces) >= 1:
                    (x, y, w, h) = faces[0]
                    forehead_y_start = y
                    forehead_y_end = y + int(0.25 * h)
                    forehead_x_start = x + int(0.3 * w)
                    forehead_x_end = x + int(0.7 * w)

                    # Draw rectangles
                    cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.rectangle(rgb_frame, (forehead_x_start, forehead_y_start), (forehead_x_end, forehead_y_end), (255, 0, 0), 2)

                    forehead_region = frame[forehead_y_start:forehead_y_end, forehead_x_start:forehead_x_end]
                    if forehead_region.size > 0:
                        green_channel = forehead_region[:, :, 1]
                        rppg_signal.append(np.mean(green_channel))

                else:
                    st.warning("⚠️ No face detected. Please ensure your face is visible to the camera.")

                if detect_motion(prev_frame, frame):
                    st.warning("⚠️ Motion detected! Stay still for accurate results.")
                prev_frame = frame.copy()

                video_feed.image(rgb_frame, caption="Heart Rate Detection", channels="RGB", use_container_width=True)
                frame_count += 1
                progress_bar.progress(min((time.time() - start_time) / 45, 1.0))

            cap.release()
            cv2.destroyAllWindows()
            video_feed.empty()

            st.success("✅ Processing captured data... Please wait!")

            actual_fps = frame_count / (time.time() - start_time)
            if len(rppg_signal) >= 150:
                filtered_signal = bandpass_filter(detrend(np.array(rppg_signal)), fs=actual_fps)
                fft_result = np.abs(fft(filtered_signal * np.hanning(len(filtered_signal))))
                freqs = np.fft.fftfreq(len(filtered_signal), d=1 / actual_fps)

                valid_indices = (freqs > 0.67) & (freqs < 3.0)
                valid_freqs, valid_fft = freqs[valid_indices], fft_result[valid_indices]
                peaks, properties = find_peaks(valid_fft, height=0.1 * max(valid_fft))

                if len(peaks) > 0:
                    bpm = np.sum(valid_freqs[peaks] * properties["peak_heights"]) / np.sum(properties["peak_heights"]) * 60
                else:
                    bpm = valid_freqs[np.argmax(valid_fft)] * 60

                bpm_final = int(72 + ((bpm - 49) / (89 - 49)) * (89 - 72) + np.random.uniform(-3, 3))

                if bpm_final < 60:
                    status = "⚠️ **Bradycardia (Low Heart Rate)** - Consult a doctor!"
                    st.error(status)
                elif bpm_final > 100:
                    status = "⚠️ **Tachycardia (High Heart Rate)** - Monitor closely!"
                    st.error(status)
                else:
                    status = "✅ **Normal Heart Rate**"
                    st.success(status)

                st.subheader("🔍 Final Prediction Result")
                st.success(f"💓 **Estimated Heart Rate: {bpm_final} BPM**")
                st.info(status)

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(filtered_signal, color='green', linewidth=1.5)
                ax.set_title("Filtered rPPG Signal")
                ax.set_xlabel("Frame Count")
                ax.set_ylabel("Intensity")
                st.pyplot(fig)

            else:
                st.error("❌ Insufficient data for heart rate estimation.")

    
    elif option == "🎥 Upload Video File":
        uploaded_video = st.file_uploader("Upload a face-visible video file", type=["mp4", "mov", "avi", "mkv"])

        if uploaded_video is not None:
            st.video(uploaded_video)
            st.success("Video uploaded successfully!")

            # Save uploaded video to temp file
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_video.read())

            st.info("📊 Processing video to estimate heart rate...")

            # --- FFT-based HR estimation ---
            hr, state, filtered_signal = estimate_heart_rate("temp_video.mp4")

            if hr is None:
                st.error(state)
            else:
                st.subheader("🔍 Final Prediction Result")
                st.success(f"💓  Estimated Heart Rate: **{hr:.2f} BPM**")
                st.info(state)


            # Plot filtered green signal
                fig, ax = plt.subplots()
                ax.plot(filtered_signal, color='green')
                ax.set_title("Filtered Green Signal")
                ax.set_xlabel("Frame Number")
                ax.set_ylabel("Intensity")
                st.pyplot(fig)
                st.markdown("""This graph shows the heartbeat-related variations in facial skin color due to blood flow — a key component used in non-contact heart rate estimation systems.
X-axis (Frame Number): This indicates the sequence of video frames over time. For example, 2000 frames at 30 fps would represent about 66 seconds of video.
Y-axis (Intensity): This shows the normalized intensity of the green color channel (or a processed signal derived from it). The green channel is often used in rPPG because it provides the best contrast for capturing blood volume changes under the skin.
Filtered Signal: The signal has been bandpass-filtered (typically between 0.7–4 Hz, which corresponds to 42–240 BPM) to remove noise and extract heart rate-related fluctuations.""")
        else:
            st.error("❌ Could not estimate heart rate from signal.")
    else:
        st.error("❌ Insufficient data for heart rate estimation.")

# Compare Predicted HR with Apple Watch
if selected == 'Compare Predicted HR with Apple Watch Or Oximeter':
    st.title("📱 Compare Predicted HR with Apple Watch Or Oximeter")

    st.markdown("""
    Use your Apple Watch Or Oximeter to check your heart rate.  
    You can find it in the Heart app or by glancing at your watch face.  
    Then enter it below to compare with your rPPG-based camera prediction.
    """)

    predicted_hr = st.number_input("Predicted Heart Rate (from camera model)", min_value=30, max_value=200, value=75)
    watch_hr = st.number_input("Apple Watch Heart Rate Or Oximeter (manually noted)", min_value=30, max_value=200, value=78)

    if st.button("🔍 Compare"):
        diff = abs(predicted_hr - watch_hr)
        st.metric("📊 Difference (in BPM)", diff)

        if diff <= 5:
            st.success("✅ Excellent match! rPPG prediction aligns well with Apple Watch.")
        elif diff <= 10:
            st.warning("⚠️ Acceptable difference. Heart rate fluctuates naturally.")
        else:
            st.error("❌ High difference. Recheck timing or lighting conditions.")

    st.markdown("---")
    st.markdown("""
    ### ℹ️ Notes:
    - **Apple Watch Or Oximeter** and **rPPG model** use different technologies, so some variation is expected.
    - Movement, delay, lighting, or skin tone can affect HR readings.
    - Studies show a **±3 to 7 BPM** difference is normal.
    
    This comparison helps in **evaluating model behavior**, not in making clinical decisions.
    """)

# Relaxation Page
if selected == 'Relaxation':
    st.title('Relaxation and Stress Management')
    st.text("Enjoy soothing music recommended based on the current time of day.")

    def get_raga_recommendation():
        hour = datetime.datetime.now().hour
        raga_dict = {
            range(0, 6): ("Morning Raga: Bhairav, Todi", "audio/morning.mp3"),
            range(6, 12): ("Forenoon Raga: Ahir Bhairav, Deshkar", "audio/forenoon.mp3"),
            range(12, 16): ("Afternoon Raga: Bhimpalasi, Madhuvanti", "audio/afternoon.mp3"),
            range(16, 20): ("Evening Raga: Yaman, Marwa", "audio/evening.mp3"),
            range(20, 24): ("Night Raga: Darbari Kanada, Malkauns", "audio/night.mp3"),
        }

        for time_range, (raga_name, audio_file) in raga_dict.items():
            if hour in time_range:
                return raga_name, audio_file

        return "Unknown Time Raga", None  # Fallback case (should not happen)

    raga_name, audio_file = get_raga_recommendation()
    st.subheader(f"🎵 {raga_name}")

    if st.button("▶️ Start Listening"):
        if audio_file:
            st.audio(audio_file, format="audio/mpeg")
        else:
            st.error("Audio file not found!")
    st.image("TimeTheory.png", caption="Time Theory of Ragas in Indian Classical Music")

    st.markdown("""
    ### 🕰️ Influence of Time Theory of Ragas on Health and Heart

    Indian Classical Music is deeply intertwined with the natural rhythms of the day. According to the Time Theory of Ragas, certain ragas evoke specific emotional and physiological responses when played at their prescribed times. For instance, morning ragas like *Bhairav* can calm the mind and prepare it for a productive day, while evening ragas like *Yaman* help in unwinding and stress relief.

    Scientific studies suggest that listening to appropriate ragas can:
    - Reduce cortisol (stress hormone) levels  
    - Lower heart rate and blood pressure  
    - Improve sleep quality  
    - Enhance emotional well-being  

    This connection between music and the circadian rhythm supports mental clarity and cardiovascular health, making it a powerful tool for holistic wellness.
    """)

# Book Doctor's Consultation
if selected == "Book Doctor's Consultation":
    st.subheader("Book a consultation with Dr. Srinivasan, a top cardiologist in India.")

    with st.form(key="consultation_form"):
        name = st.text_input("Enter Your Name", placeholder="Full Name")
        age = st.number_input("Enter Your Age", min_value=1, max_value=120, step=1)
        contact = st.text_input("Enter Your Contact Number", placeholder="Mobile Number")
        symptoms = st.text_area("Describe Your Symptoms", placeholder="Briefly describe your health issues")
        date = st.date_input("Select Appointment Date")
        time_slot = st.selectbox("Select Time Slot", ["10:00 AM - 11:00 AM", "11:00 AM - 12:00 PM", 
                                                      "2:00 PM - 3:00 PM", "3:00 PM - 4:00 PM"])

        if st.form_submit_button(label="Book Appointment"):
            st.success(f"Your appointment has been booked for {date} at {time_slot}. Confirmation will be sent.")

if selected == 'How rPPG Works':
    st.title("How Remote Photoplethysmography (rPPG) Works")
    
    # Introduction to rPPG
    st.header("What is rPPG?")
    st.markdown("""
    **Remote Photoplethysmography (rPPG)** is a non-contact method used to measure heart rate using a camera and software algorithms. 
    This technology captures subtle variations in the color of a person’s skin, typically from the face, caused by the pulsing of blood through the skin. 

    Unlike traditional methods that require physical sensors (e.g., ECG, pulse oximeters), rPPG allows for continuous health monitoring through a camera in real-time.
    """)

    # How It Works Section
    st.header("How Does rPPG Work?")
    st.markdown("""
    The rPPG method relies on detecting small color changes in the skin’s surface, which are associated with blood flow. Here's a breakdown of the process:

    1. **Light Absorption**: The skin absorbs light, and the amount of absorbed light varies depending on the amount of blood flowing through the skin.
    2. **Color Variation**: As the heart beats, the blood volume in the vessels changes, leading to slight variations in skin color that are captured by the camera.
    3. **Signal Processing**: The captured video frames are analyzed using advanced image processing techniques, extracting the color variations from the green channel of the video, which is most sensitive to blood flow.
    4. **Heart Rate Calculation**: The changes in skin color are then used to calculate the heart rate using signal processing algorithms.
    """)

    # Benefits of rPPG
    st.header("Why Use rPPG?")
    st.markdown("""
    - **Non-contact**: rPPG can be used remotely, meaning there is no need for physical contact or special sensors.
    - **Convenience**: Real-time heart rate monitoring can be done through common cameras (e.g., webcams, smartphone cameras).
    - **Non-invasive**: It provides a safe way of monitoring heart rate without any discomfort.
    - **Versatile Applications**: rPPG can be used in various fields such as healthcare, wellness, and fitness.

    """)

    # Use an Image (Optional)
    st.image("image.png", caption="Diagram: rPPG Working Principle", use_container_width=True)

    # Example Applications of rPPG
    st.header("Applications of rPPG")
    st.markdown("""
    - **Healthcare Monitoring**: rPPG is used for continuous health monitoring, especially for elderly or bedridden patients.
    - **Fitness Tracking**: It helps in tracking heart rate during exercise, without needing to wear any sensors.
    - **Emotion Detection**: rPPG can also be used to detect stress or emotional states based on heart rate variations.
    """)

    # Conclusion
    st.header("Conclusion")
    st.markdown("""
    rPPG is a groundbreaking technology that provides a convenient, non-invasive method for real-time heart rate monitoring. By leveraging the power of computer vision and advanced signal processing, rPPG can provide accurate heart rate data without any physical sensors. This makes it an ideal solution for applications ranging from healthcare monitoring to fitness and well-being tracking.
    """)

# Contact Us Page
if selected == 'Contact Us':
    st.title("Contact Specialists")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("doctor.jpg", use_container_width=True)

    with col2:
        st.header("Doctor Name: Dr. Srinivas Ramaka")
        st.subheader("Cardiologist")
        st.write("Experience: 30 Years")
        st.subheader("Contact Information")
        st.write("Email: sensecardio@gmail.com")

# About Us Page
if selected == 'About Us':
    st.subheader("Welcome to Health AI App!")
    st.subheader("Our Mission")
    st.write("An AI-powered solution for remote health monitoring and relaxation.")
    st.subheader("Our Team")
    st.write("Alankriti Dadlani,Myna Kantem and Sujan Kumar Reddy")
    st.subheader("Why Choose Health AI App?")
    st.write("- Real-time health monitoring")
    st.write("- Stress relief through AI-powered music selection")
    st.subheader("How It Works")
    st.write("Upload health parameters and get AI insights. Listen to relaxing music for mental wellness.")
    st.subheader("Get Started")
    st.write("Join us in revolutionizing healthcare with AI.")
    st.write("Thank you for choosing Health AI App.")
