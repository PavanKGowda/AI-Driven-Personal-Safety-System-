import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FAST2SMS_API_KEY")
CONTACTS = os.getenv("ALERT_CONTACTS")

def send_sms_via_fast2sms(message):
    if not API_KEY or not CONTACTS:
        print("❌ Missing FAST2SMS API key or contacts")
        return False

    url = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        "authorization": API_KEY,
        "sender_id": "FSTSMS",
        "message": message,
        "language": "english",
        "route": "q",
        "numbers": CONTACTS
    }

    headers = {
        'cache-control': "no-cache"
    }

    try:
        response = requests.post(url, data=payload, headers=headers)
        print(f"📤 SMS sent! Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")
        return False
