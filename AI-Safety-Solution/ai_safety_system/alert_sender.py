import smtplib
from email.message import EmailMessage
from ai_safety_system.emergency_contact import EMERGENCY_CONTACTS

def send_email_alert(audio_clip_path):
    try:
        msg = EmailMessage()
        msg["Subject"] = "🚨 EMERGENCY ALERT: AI Safety System"
        msg["From"] = "pavankg2803@gmail.com"
        msg["To"] = EMERGENCY_CONTACTS["email"]
        msg.set_content("An emergency has been detected. Audio clip is attached for review.")

        with open(audio_clip_path, "rb") as f:
            audio_data = f.read()
            msg.add_attachment(audio_data, maintype="audio", subtype="wav", filename="alert_clip.wav")

        # Send Email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login("pavankg2803@gmail.com", "Cyperpunk@123")
            smtp.send_message(msg)

        print("📧 Email alert sent successfully!")

    except Exception as e:
        print(f"❌ Failed to send email alert: {e}")
