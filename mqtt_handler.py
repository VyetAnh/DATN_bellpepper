import paho.mqtt.client as mqtt

MQTT_BROKER = "ba662f8f.ala.asia-southeast1.emqxsl.com"
MQTT_PORT = 8883

MQTT_USER = "quang"
MQTT_PASS = "12052004"


MQTT_TOPIC_PUMP = "irrigation/device1/control/pump"
MQTT_TOPIC_TIME = "irrigation/device1/control/time"


client = mqtt.Client(client_id="AI_SERVER")

client.username_pw_set(
    MQTT_USER,
    MQTT_PASS
)

client.tls_set()

client.connect(
    MQTT_BROKER,
    MQTT_PORT,
    60
)

client.loop_start()


def send_command_mqtt(result):

    pump_on = int(result["pump_on"])
    relay_ms = int(result["relay_on_ms"])

    if pump_on == 1:

        seconds = max(1, relay_ms // 1000)

        # gửi thời gian
        client.publish(
            MQTT_TOPIC_TIME,
            str(seconds),
            qos=1
        )

        # bật relay
        client.publish(
            MQTT_TOPIC_PUMP,
            "ON",
            qos=1
        )

        print(f"📡 ON | {seconds}s")

    else:

        client.publish(
            MQTT_TOPIC_PUMP,
            "OFF",
            qos=1
        )

        print("📡 OFF")
        
        
if __name__ == "__main__":

    test = {
        "pump_on": 1,
        "relay_on_ms": 10000
    }

    send_command_mqtt(test)

    import time
    time.sleep(2)