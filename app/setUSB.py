import serial
import serial.tools.list_ports
import time

def findNuigurumi():
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        port_name = port.device
        port_desc = port.description.lower()

        if "mindhound_nuigurumi" in port_desc or "mindhound" in port_name.lower():
            try:
                ser = serial.Serial(port_name, 115200, timeout=1)
                time.sleep(1)  # 接続後の安定化のため少し待つ

                # 👉 応答確認を試みる（"ping" 送って "pong" を期待）
                ser.write(b"ping\n")
                response = ser.readline().decode("utf-8").strip()
                if response == "pong":
                    print(f"✅ 実応答あり！ポート接続成功: {port_name}")
                    return ser
                else:
                    print(f"⚠️ 実応答なし: {port_name} → {response}")
                    ser.close()
            except serial.SerialException as e:
                print(f"⚠️ {port_name} に接続できませんでした: {e}")

    print("⚠️ MindHound Nuigurumi が見つかりませんでした")
    return None
