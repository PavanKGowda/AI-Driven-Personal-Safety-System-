# 🛡️ AI-DRIVEN PERSONAL SAFETY SYSTEM 🛡️ 

The **AI SAFETY SOLUTION** is an innovative solution that leverages the power of Artificial Intelligence (AI) to enhance personal security, with a particular focus on **women’s safety**. This system uses **real-time audio analysis** to detect signs of distress such as **screams or emergency keywords**. By integrating machine learning models on edge devices, the solution ensures fast, on-device processing. Upon detecting a threat, the system automatically **records audio evidence** and **triggers alerts**, making it a powerful tool for real-world safety applications.

.
.
.

📌 **FEATURES**

🎤 Live Audio Monitoring using microphone

🗣️ Emergency Keyword Detection (e.g., help, save me, fire, etc.)

📢 Scream Detection via trained ML model

🔐 Alert Triggering & Logging in real-time

💾 Audio Clipping and Storage with timestamped directories

🔧 Modular Codebase (easy to integrate with mobile/IoT devices)

.
.
.

📊 **MODEL AND DATASET**

📢 **SCREAM DETECTION**

▫️Trained on the Human Screaming Detection Dataset from Kaggle.

▫️Utilizes MFCC feature extraction and a custom neural network.

▫️Converted and deployed using TensorFlow Lite for edge performance.

🔑 **KEYWORD DETECTION**

▫️Built using Google’s Speech Recognition API for real-time transcription.

▫️Emergency keyword list includes: "help", "save me", "emergency", "stop", "fire", etc.

▫️Matches detected speech against this list to trigger alerts.
        
.
.
.

📁 **PROJECT STRUCTURE**

**|── ai_safety_system**: 
        **└── ai_safety_system/keyword_detection.py**       
        **└── ai_safety_system/scream_detection.py**   
**├── data/**
**├── datatsets**
        **├── scream/**                        
        **├── non_scream/**                    
**├── saved_audio/YYYY-MM-DD/**       
**├── train_model_from_wav.py**       
**├── run_alerter.py**               
**├── Run_alert.ipynb**               
**├── requirements.txt** 

.
.
.

🛠️ **SETUP & INSTALLATION**

🔧 **Requirements**

▫️Python 3.8+

▫️TensorFlow, SpeechRecognition, NumPy, Pyaudio, etc.

▫️Install them using:

        pip install -r requirements.txt

▫️Install them using: 

        pip install -r requirements.txt

.
.
.

▶️ **HOW TO RUN**

1. Train the model (optional, already included):

        python train_model_from_wav.py

4. Start the real-time alerter:

        python run_alerter.py

You'll be prompted to press Enter. The script then listens for either a scream or emergency keyword. If detected, the audio is saved and an alert is triggered.

.
.
.

🧱 System Architecture
Here is a simplified flowchart of how the system works:

![image](https://github.com/user-attachments/assets/571e5d52-3389-4fba-a91a-d1b27b870da2)

**FIG : System Architecture of AI Safety Solution System**



.
.
.

🧪 **SAMPLE OUTPUT**

✅ Alerter Activated! Listening in real-time...

🎤 Listening...

🗣️ Detected Speech: 'help me please'
🔑 Emergency keyword detected: 'help'
📢 Reason: Scream
🚨 EMERGENCY TRIGGERED!
💾 Alert clip saved: saved_audio/2025-04-12/13-45-32.wav

📲 Sending alert...

.
.
.

🔒 **FUTURE ENHANCEMENTS**

📱 Mobile App Integration (Flutter + Firebase)

🧭 Safe Route Recommender using Maps API

📡 SOS Alert via WhatsApp/Call/SMS

🤖 Real-time posture or activity monitoring (AI + Camera)

.
.
.

📚 **REFERENCES**

▫️Human Scream Detection Dataset – Kaggle

▫️SpeechRecognition Python Docs

▫️TensorFlow Lite Documentation

