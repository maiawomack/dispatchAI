import os
import uuid
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
BASE_URL    = os.getenv("APP_BASE_URL", "https://placeholder.com")


def send_emergency_link(caller_phone: str) -> dict:
    session_id = str(uuid.uuid4())
    link = f"{BASE_URL}/emergency?session={session_id}"

    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    message = client.messages.create(
        to=caller_phone,
        from_=FROM_NUMBER,
        body=(
            "🚨 Emergency Video Link\n"
            f"Tap to start your camera: {link}\n\n"
            "Point your camera at the scene. Help is on the way."
        )
    )

    print(f"✅ SMS sent! SID: {message.sid}")
    print(f"   Session ID: {session_id}")
    print(f"   Link: {link}")

    return {"session_id": session_id, "message_sid": message.sid, "link": link}


if __name__ == "__main__":
    import sys
    phone = sys.argv[1] if len(sys.argv) > 1 else "+16789670048"
    send_emergency_link(phone)