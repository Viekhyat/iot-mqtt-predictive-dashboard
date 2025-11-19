import paho.mqtt.client as mqtt
import json
import os
from datetime import datetime

DATA_FILE = "iot_messages.json"
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "Viekhyat/machine/sensors"

# Safely load existing messages
if os.path.exists(DATA_FILE):
    try:
        with open(DATA_FILE, "r") as f:
            all_messages = json.load(f)
            if not isinstance(all_messages, list):
                all_messages = []
    except json.JSONDecodeError:
        print("⚠️ JSON file corrupted, starting with empty list")
        all_messages = []
else:
    all_messages = []

def normalize_keys(data):
    key_map = {
        "Vibratioon_Level_mms": "Vibration_Level_mms",
        "Vibration_Level_mmms": "Vibration_Level_mms",
    }
    for wrong, correct in key_map.items():
        if wrong in data:
            data[correct] = data.pop(wrong)
    return data

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT Broker")
        client.subscribe(TOPIC)
        print(f"📡 Subscribed to topic: {TOPIC}")
    else:
        print("❌ Connection failed with code", rc)

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    try:
        # Convert JSON string to dict
        data = json.loads(payload)

        # Normalize keys
        data = normalize_keys(data)

        # Add timestamp
        data["timestamp"] = datetime.now().isoformat()

        # Append and save
        all_messages.append(data)
        with open(DATA_FILE, "w") as f:
            json.dump(all_messages, f, indent=4)

        print(f"\n📩 Received message from {msg.topic}:")
        print(json.dumps(data, indent=2))

    except json.JSONDecodeError:
        print(f"⚠️ Failed to parse message: {payload}")

# MQTT Client setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("🚀 Starting MQTT Receiver...")
client.connect(BROKER, PORT, 60)
client.loop_forever()
