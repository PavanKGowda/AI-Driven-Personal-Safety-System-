import speech_recognition as sr
import tempfile
import os
import time
from datetime import datetime
from ai_safety_system.keyword_detection import EMERGENCY_KEYWORDS

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_FOLDER = "saved_audio"
os.makedirs(BASE_FOLDER, exist_ok=True)

def get_datetime():
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S")

def recognize_keywords_from_audio(audio_data):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data)
        print(f"🗣️ Detected Speech: '{text}'")
        for keyword in EMERGENCY_KEYWORDS:
            if keyword.lower() in text.lower():
                print(f"🔑 Emergency keyword detected: '{keyword}'")
                return True
        return False
    except sr.UnknownValueError:
        print("⚠️ Could not understand audio.")
    except sr.RequestError as e:
        print(f"🔌 API error: {e}")
    return False

def start_alerter():
    print("\n✅ Alerter Activated! Listening in real-time...")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                print("🎤 Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

                # Step 1: Keyword detection
                keyword_detected = recognize_keywords_from_audio(audio)

                # Step 2: Save audio temporarily for scream detection
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    temp_audio_path = temp_audio.name
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio.get_wav_data())

                # Step 3: Scream detection
                scream_detected = predict(temp_audio_path)

                # Step 4: Save only if alert
                if keyword_detected or scream_detected:
                    print("\n🚨 EMERGENCY TRIGGERED!")
                    if keyword_detected:
                        print("🔑 Reason: Keyword")
                    if scream_detected:
                        print("📢 Reason: Scream")

                    # Create folder based on date
                    date_str, time_str = get_datetime()
                    date_folder = os.path.join(BASE_FOLDER, date_str)
                    os.makedirs(date_folder, exist_ok=True)

                    # Save alert clip
                    filename = os.path.join(date_folder, f"{time_str}.wav")
                    os.rename(temp_audio_path, filename)
                    print(f"💾 Alert clip saved: {filename}")
                    # 🚀 Send Alerts

                    time.sleep(3)
                else:
                    os.remove(temp_audio_path)  # Clean up if no alert
                    print("✅ No emergency.\n")

            except KeyboardInterrupt:
                print("\n🛑 Alerter deactivated by user.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

if __name__ == "__main__":
    input("🔘 Press Enter to activate the ALERTER (or Ctrl+C to cancel)...")
    start_alerter()
