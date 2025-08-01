import serial
import serial.tools.list_ports

def findNuigurumi():
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        port_name = port.device
        port_desc = port.description.lower()

        # 明示的に名前をチェック
        if "mindhound_nuigurumi" in port_desc or "mindhound" in port_name.lower():
            try:
                ser = serial.Serial(port_name, 115200, timeout=1)
                print(f"✅ シリアルポート接続成功: {port_name}")
                return ser
            except serial.SerialException as e:
                print(f"⚠️ {port_name} に接続できませんでした: {e}")

    print("⚠️ MindHound Nuigurumi が見つかりませんでした")
    return None
