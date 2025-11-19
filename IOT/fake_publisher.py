import time
import random
import json
import paho.mqtt.client as mqtt
import sys

# MQTT Broker (Mosquitto Public)
BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "Viekhyat/machine/sensors"
PUBLISH_INTERVAL = 3.0  # seconds

# Define callback functions for MQTT client
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"✅ Connected to MQTT Broker: {BROKER}")
    else:
        print(f"❌ Failed to connect to MQTT Broker. Return code: {rc}")
        sys.exit(1)

def on_disconnect(client, userdata, rc, properties=None):
    if rc != 0:
        print(f"❌ Unexpected disconnection. Return code: {rc}")

# Setup MQTT client with callbacks
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_disconnect = on_disconnect

# Connect with error handling
try:
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()
except Exception as e:
    print(f"❌ Connection error: {e}")
    sys.exit(1)

# Fixed setup
industries = ["Electronics", "Automobile", "Pharmaceutical", "Food Processing", "Steel", "Textile"]
companies_per_sector = 4  # 4 companies * 5 machines = 20 machines per sector
machines_per_company = 5

# Create fixed mapping of companies and machines per industry
sector_map = {}
for industry in industries:
    machines = []
    for c in range(1, companies_per_sector + 1):
        company_id = f"C{industries.index(industry)*companies_per_sector + c:02d}"  # unique company ID
        for m in range(1, machines_per_company + 1):
            machine_id = f"M{(companies_per_sector * machines_per_company) * industries.index(industry) + (c-1)*machines_per_company + m:03d}"
            machines.append({"Company_ID": company_id, "Machine_ID": machine_id})
    sector_map[industry] = machines

regions = ["North", "South", "East", "West", "Central"]

def gen_sensor_data(machine, industry):
    temperature = round(random.uniform(20.0, 90.0), 2)  # Lower max temp
    vibration = round(random.uniform(0.1, 5.0), 3)      # Lower max vibration
    energy_kwh = round(random.uniform(10.0, 2000.0), 2)  # Lower energy
    carbon_kg = round(energy_kwh * random.uniform(0.1, 0.5), 2)  # Lower emission factor
    downtime_hours = round(random.uniform(0.0, 3.0), 3) # Lower downtime
    production_output = int(random.uniform(10, 5000))   # Lower output
    safety_incidents = random.choices([0, 1], weights=[0.995, 0.005])[0]  # Even fewer incidents

    data = {
        "Company_ID": machine["Company_ID"],
        "Industry": industry,
        "Region": random.choice(regions),
        "Machine_ID": machine["Machine_ID"],
        "Temperature_C": temperature,
        "Vibration_Level_mms": vibration,
        "Energy_Consumption_kWh": energy_kwh,
        "Carbon_Emission_kg": carbon_kg,
        "Production_Output": production_output,
        "Downtime_Hours": downtime_hours,
        "Safety_Incidents": safety_incidents
    }
    return data

print(f"🚀 Publishing to topic '{TOPIC}' every {PUBLISH_INTERVAL}s...")

try:
    while True:
        for industry, machines in sector_map.items():
            for machine in machines:
                payload = gen_sensor_data(machine, industry)
                client.publish(TOPIC, json.dumps(payload))
                print("✅ Published:", payload)
                time.sleep(PUBLISH_INTERVAL)
except KeyboardInterrupt:
    print("🛑 Stopping publisher...")
finally:
    client.loop_stop()
    client.disconnect()
