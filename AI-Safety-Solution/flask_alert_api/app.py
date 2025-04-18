from flask import Flask, request, jsonify
from datetime import datetime
import os
from sms_sender import send_sms_via_fast2sms
app = Flask(__name__)
LOG_FILE = os.path.join(os.path.dirname(__file__), "alerts.log")

@app.route("/send_alert", methods=["POST"])
def send_alert():
    data = request.json

    reason = data.get("reason")
    location = data.get("location")
    timestamp = data.get("timestamp", datetime.now().isoformat())

    if not reason or not location:
        return jsonify({"error": "Missing reason or location"}), 400

    alert_msg = f"[{timestamp}]\n🚨 Emergency: {reason}\n📍 Location: {location}"
    print(alert_msg)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(alert_msg + "\n")

    # 📨 Send SMS here
    sms_sent = send_sms_via_fast2sms(alert_msg)

    return jsonify({
        "status": "alert_received",
        "sms_sent": sms_sent
    }), 200


@app.route("/", methods=["GET"])
def index():
    return "🚨 AI Safety Alert API is running!"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
