# Edge-AI-Based-Non-Contact-Healthcare-Abnormality-Prediction
**Overview**

This project implements an Edge AI-based non-contact healthcare abnormality prediction system using Remote PPG (rPPG) signal analysis. The system leverages a Jetson Nano to process video input, extract facial images, perform image and signal processing, and predict cardiovascular abnormalities.

**Features**
Non-contact PPG Signal Analysis: Extracts PPG signals from facial video.

Deep Learning-Based Prediction: Uses a CNN-LSTM model to analyze rPPG signals.

Edge AI Acceleration: Optimized for Jetson Nano using CUDA acceleration.

Real-Time Health Monitoring: Provides insights into cardiovascular health abnormalities.

Deployable as a Streamlit Web App: User-friendly interface for real-time monitoring.
Installation & Setup

**Prerequisites**

Jetson Nano Developer Kit

64GB microSD card (EVM Elite recommended)

Python 3.8+

CUDA Toolkit (for Jetson Nano acceleration)

PyTorch with GPU support

Step 1: Set Up Jetson Nano

Flash Jetson Nano SD card using JetPack SDK.

Set up a virtual environment:

sudo apt update && sudo apt upgrade
python3 -m venv edge-ai-env
source edge-ai-env/bin/activate

Step 2: Install Dependencies

pip install -r requirements.txt

Step 3: Run the Model

To test the model on sample data:

python src/inference.py --video sample.mp4

Step 4: Deploy as a Web App

streamlit run src/streamlit_app.py

**Dataset**

UBFC-RPPG Dataset: Used for training and validation.

Preprocessing Steps: Face detection, ROI extraction, signal filtering.

**Model Details**

CNN-LSTM Hybrid Model

Input: Extracted PPG signals from facial regions

Output: Heart rate estimation and abnormality detection
