from keyword_detection import recognize_keywords_from_file
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if __name__ == "__main__":
    audio_path = "C:/Users/pavan/OneDrive/Documents/AI Safety Solution/datasets/call_help.wav"
    recognize_keywords_from_file(audio_path)