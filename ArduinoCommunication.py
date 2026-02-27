import serial
import serial.tools.list_ports
import time


class ArduinoCommunication:

    def __init__(self, baudrate=9600):
        self.arduino = None
        self.baudrate = baudrate

        # Try auto connect
        if not self.auto_connect_arduino():
            self.manual_connect()

    # ----------------------------------

    def auto_connect_arduino(self):
        ports = serial.tools.list_ports.comports()

        for port in ports:
            if ('Arduino' in port.description or
                'USB' in port.description or
                'CH340' in port.description):

                try:
                    self.arduino = serial.Serial(
                        port.device,
                        self.baudrate,
                        timeout=1
                    )
                    time.sleep(2)
                    print(f"Connected to Arduino on {port.device}")
                    return True
                except:
                    continue

        print("Auto-detect failed.")
        return False

    # ----------------------------------

    def manual_connect(self):
        ports = serial.tools.list_ports.comports()

        if not ports:
            print("No serial ports detected. Running in simulation mode.")
            return

        print("\nAvailable ports:")
        for idx, port in enumerate(ports):
            print(f"{idx+1}. {port.device} - {port.description}")

        user_input = input("\nEnter port name or index: ").strip()

        if user_input.isdigit():
            index = int(user_input) - 1
            if 0 <= index < len(ports):
                user_input = ports[index].device
            else:
                print("Invalid selection.")
                return

        self.connect_arduino(user_input)

    # ----------------------------------

    def connect_arduino(self, port):
        try:
            self.arduino = serial.Serial(
                port,
                self.baudrate,
                timeout=1
            )
            time.sleep(2)
            print(f"Connected to Arduino on {port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.arduino = None
            return False

    # ----------------------------------

    def send_to_arduino(self, solution):

        if self.arduino is None:
            print("\nSimulation mode:")
            for move in solution.split():
                print(f"  -> {move}")
            return

        moves = solution.split()

        for idx, move in enumerate(moves, 1):

            print(f"Move {idx}/{len(moves)}: {move}")
            self.arduino.write(f"{move}\n".encode())

            # Handshake wait
            start_time = time.time()
            timeout = 10

            while True:
                if self.arduino.in_waiting:
                    response = self.arduino.readline().decode().strip()
                    if response == "DONE":
                        break

                if time.time() - start_time > timeout:
                    print("Timeout waiting for Arduino.")
                    return

        print("Solution sent successfully.")

    # ----------------------------------

    def send_single_move(self, move):
        for cmd in move:
            self.arduino.write(f"{cmd}\n".encode())
            while True:
                if self.arduino.in_waiting:
                    response = self.arduino.readline().decode().strip()
                    if response == "DONE":
                        break
            

    def is_connected(self):
        return self.arduino is not None and self.arduino.is_open
