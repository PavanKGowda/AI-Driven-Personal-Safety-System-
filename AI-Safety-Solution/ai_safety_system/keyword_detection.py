import speech_recognition as sr

EMERGENCY_KEYWORDS = ["help", "save me", "police", "emergency"]

def recognize_keywords_from_file(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            print(f"🎧 Loading file: {file_path}")
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        print(f"🗣️ Transcription: '{text}'")

        for keyword in EMERGENCY_KEYWORDS:
            if keyword.lower() in text.lower():
                print(f"🚨 Emergency keyword detected: '{keyword}'")
                return True

        print("✅ No emergency keywords detected.")
        return False

    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
    except sr.RequestError as e:
        print(f"⚠️ API error: {e}")
    except Exception as e:
        print(f"❌ Error processing audio file: {e}")

    return False
