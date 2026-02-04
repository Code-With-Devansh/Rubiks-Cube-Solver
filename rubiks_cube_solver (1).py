"""
Rubik's Cube Solver with Webcam and Arduino Integration
This driver captures cube faces using webcam, solves using Kociemba algorithm,
and sends solution to Arduino for execution.
"""

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import kociemba as sv


class RubiksCubeSolver:
    def __init__(self, arduino_port=None, baudrate=9600):
        """
        Initialize the Rubik's Cube Solver
        
        Args:
            arduino_port: Serial port for Arduino (auto-detect if None)
            baudrate: Serial communication speed
        """
        self.solution_text = None
        self.error_text = None  

        self.cap = None
        self.arduino = None
        self.cube_state = [''] * 54 
        self.face_colors = []
        self.current_face = 0
        self.face_names = ['Front', 'Right', 'Back', 'Left', 'Up', 'Down']
        

        self.color_ranges = {
            'F': ([0, 0, 200], [180, 40, 255]),

            'L': ([22, 120, 120], [32, 255, 255]),

            'R1': ([0, 120, 100], [8, 255, 255]),
            'R2': ([170, 120, 100], [180, 255, 255]),

            'U': ([10, 130, 120], [20, 255, 255]),

            'D': ([40, 80, 80], [75, 255, 255]),

            'B': ([95, 80, 80], [125, 255, 255])
        }

        # Initialize Arduino connection
        if arduino_port:
            self.connect_arduino(arduino_port, baudrate)
        else:
            self.auto_connect_arduino(baudrate)
    
    def auto_connect_arduino(self, baudrate=9600):
        """Automatically detect and connect to Arduino"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'Arduino' in port.description or 'USB' in port.description:
                try:
                    self.arduino = serial.Serial(port.device, baudrate, timeout=1)
                    time.sleep(2)  # Wait for Arduino to reset
                    print(f"Connected to Arduino on {port.device}")
                    return True
                except:
                    continue
        print("Warning: Arduino not found. Running in simulation mode.")
        return False
    
    def connect_arduino(self, port, baudrate=9900):
        """Connect to Arduino on specified port"""
        try:
            self.arduino = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            print(f"Connected to Arduino on {port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            return False
    
    def init_webcam(self, camera_index=0):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        print("Webcam initialized")

    def draw_status_overlay(self, frame):
        if not self.solution_text and not self.error_text:
            return frame

        h, w = frame.shape[:2]

        Y_OFFSET = 50   # ← move overlay down by 150 pixels
        BAR_HEIGHT = 60

        cv2.rectangle(
            frame,
            (0, Y_OFFSET),
            (w, Y_OFFSET + BAR_HEIGHT),
            (0, 0, 0),
            -1
        )
        if self.solution_text == '':
                cv2.putText(
                    frame,
                    f"Solution: {"Already Solved!"}",
                    (10, Y_OFFSET + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

        if self.solution_text:
            cv2.putText(
                frame,
                f"Solution: {self.solution_text}",
                (10, Y_OFFSET + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        elif self.error_text:
            cv2.putText(
                frame,
                f"Solver Error: {self.error_text}",
                (10, Y_OFFSET + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return frame

    def draw_cube_net(self, frame):
        """
        Draw captured cube faces as a 2D net on the right side
        """
        if not self.face_colors:
            return frame

        # color to BGR (for drawing only)
        COLOR_MAP = {
            'L': (0, 255, 255),   # yellow
            'U': (0, 165, 255),   # orange
            'F': (255, 255, 255), # white
            'B': (255, 0, 0),     # blue
            'D': (0, 255, 0),     # green
            'R': (0, 0, 255),     # red
        }

        # positions in net (relative)
        net_positions = {
            4: (1, 0),  # Up
            3: (0, 1),  # Left
            0: (1, 1),  # Front
            1: (2, 1),  # Right
            2: (3, 1),  # Back
            5: (1, 2),  # Down
        }

        net_width = 4 * (3 * 34 + 10)   # exact net width
        net_height = 3 * (3 * 34 + 10)

        start_x = 50
        start_y = max(150, (frame.shape[0] - net_height) + 150)

        size = 15
        gap = 2

        for face_idx, face in enumerate(self.face_colors):
            if face_idx not in net_positions:
                continue

            gx, gy = net_positions[face_idx]
            ox = start_x + gx * (3 * (size + gap) + 10)
            oy = start_y + gy * (3 * (size + gap) + 10)

            for i in range(9):
                r = i // 3
                c = i % 3
                x = ox + c * (size + gap)
                y = oy + r * (size + gap)

                color = COLOR_MAP.get(face[i], (50, 50, 50))
                cv2.rectangle(frame, (x, y), (x + size, y + size), color, -1)
                cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 0, 0), 1)

        cv2.putText(frame, "Cube Preview", (start_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def detect_color(self, hsv_roi):
        """
        Detect the color of a sticker from HSV region of interest

        Args:
            hsv_roi: HSV image of the sticker region

        Returns:
            Color character (W, Y, R, O, G, B)
        """

        color_counts = {}
        kernel = np.ones((3, 3), np.uint8)

        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(
                hsv_roi,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8)
            )

            # remove noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            count = cv2.countNonZero(mask)

            # merge red ranges (R1 + R2)
            if color.startswith('R'):
                color_counts['R'] = color_counts.get('R', 0) + count
            else:
                color_counts[color] = count

        # ---- decision logic ----
        if not color_counts:
            return 'F'

        detected_color = max(color_counts, key=color_counts.get)
        max_pixels = color_counts[detected_color]

        # minimum confidence threshold (tune for ROI size)
        MIN_PIXELS = int(0.05 * hsv_roi.shape[0] * hsv_roi.shape[1])

        if max_pixels < MIN_PIXELS:
            # likely white or reflection
            return 'F'

        return detected_color

    def draw_sticker_grid(self, frame, stickers, face_colors=None):
        """
        Draw 3x3 grid for cube face detection
        
        Args:
            frame: Video frame
            stickers: List of sticker rectangles [(x, y, w, h), ...]
            face_colors: List of detected colors for each sticker
        
        Returns:
            Modified frame with grid
        """
        for idx, (x, y, w, h) in enumerate(stickers):
            # Draw rectangle
            color = (0, 255, 0) if face_colors is None else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw sticker number
            cv2.putText(frame, str(idx + 1), (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw detected color if available
            if face_colors and idx < len(face_colors):
                color_text = face_colors[idx]
                cv2.putText(frame, color_text, (x + w//2 - 10, y + h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def get_sticker_positions(self, frame_width, frame_height):
        """
        Calculate positions for 3x3 sticker grid
        
        Returns:
            List of (x, y, width, height) for each sticker
        """
        # Center the grid
        grid_size = 550
        sticker_size = 120
        gap = 70
        
        start_x = (frame_width - grid_size) // 2
        start_y = (frame_height - grid_size) // 2
        
        stickers = []
        for row in range(3):
            for col in range(3):
                x = start_x + col * (sticker_size + gap)
                y = start_y + row * (sticker_size + gap)
                stickers.append((x, y, sticker_size, sticker_size))
        
        return stickers
    
    def scan_face(self, display_window="Rubik's Cube Scanner"):
        """
        Scan one face of the cube
        
        Returns:
            List of 9 color characters for the face
        """
        if self.cap is None:
            self.init_webcam(1)
        
        face_colors = []
        scanning = True
        
        while scanning:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            height, width = frame.shape[:2]
            
            # Get sticker positions
            stickers = self.get_sticker_positions(width, height)
            
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Detect colors if space is pressed
            if len(face_colors) == 0:
                temp_colors = []
                for (x, y, w, h) in stickers:
                    roi = hsv[y:y+h, x:x+w]
                    color = self.detect_color(roi)
                    temp_colors.append(color)
            
            # Draw grid
            display_frame = self.draw_sticker_grid(frame.copy(), stickers,
                                       face_colors if face_colors else temp_colors)

            display_frame = self.draw_cube_net(display_frame)
            display_frame = self.draw_status_overlay(display_frame)
            # Add instructions
            face_name = self.face_names[self.current_face] if self.current_face < 6 else "Done"
            cv2.putText(display_frame, f"Scanning: {face_name} Face ({self.current_face + 1}/6)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, Q to quit",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow(display_window, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                face_colors = temp_colors.copy()
                print(f"{face_name} face captured: {face_colors}")
                scanning = False
            elif key == ord('q'):  # Q to quit
                return None
        
        return face_colors
    
    def scan_all_faces(self):
        """
        Scan all 6 faces of the cube
        
        Returns:
            String representation of cube state in Kociemba format
        """
        print("\n" + "="*50)
        print("RUBIK'S CUBE SCANNING")
        print("="*50)
        print("\nInstructions:")
        print("1. Position the cube so the specified face is clearly visible")
        print("2. Press SPACE when ready to capture the face")
        print("3. Rotate the cube to the next face as instructed")
        print("4. Press Q at any time to quit\n")
        
        all_face_colors = []
        
        for face_idx in range(6):
            self.current_face = face_idx
            print(f"\nPosition {self.face_names[face_idx]} face toward camera...")
            
            face_colors = self.scan_face()
            if face_colors is None:
                print("Scanning cancelled")
                return None
            
            all_face_colors.append(face_colors)
            
            # Store scanned image
            self.face_colors = all_face_colors.copy()
        
        # Convert to Kociemba format (URFDLB order)
        # Assuming scan order is FRBLUD, reorder to URFDLB
        kociemba_order = [4, 1, 0, 5, 3, 2]  # U, R, F, D, L, B
        cube_string = ''
        for face_idx in kociemba_order:
            cube_string += ''.join(all_face_colors[face_idx])
        
        print(f"\nCube colors: {cube_string}")

        cube_kociemba = self.colors_to_kociemba_faces(cube_string, all_face_colors)

        if cube_kociemba is None:
            print("Invalid cube detected")
            return None

        print(f"Kociemba state: {cube_kociemba}")
        return cube_kociemba


    
    def solve_cube(self, cube_string):
        """
        Solve the cube using Kociemba algorithm
        
        Args:
            cube_string: 54-character string representing cube state
        
        Returns:
            Solution string (e.g., "R U R' U'")
        """
        try:
            print("\n" + "="*50)
            print("SOLVING CUBE")
            print("="*50)
            if(cube_string != "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"):
                solution = sv.solve(cube_string)
                print(f"\nSolution found: {solution}")
                print(f"Number of moves: {len(solution.split())}")
                return solution
            else:
                print("Already solved!")
                return ""
            
        except Exception as e:
            self.error_text = str(e)
            print(f"Error solving cube: {e}")
            return None

    def colors_to_kociemba_faces(self, cube_colors, all_face_colors):
        """
        Convert color-based cube string to Kociemba face-based string
        """

        # centers from scan order: Front, Right, Back, Left, Up, Down
        center_map = {
            all_face_colors[4][4]: 'U',  # Up
            all_face_colors[1][4]: 'R',  # Right
            all_face_colors[0][4]: 'F',  # Front
            all_face_colors[5][4]: 'D',  # Down
            all_face_colors[3][4]: 'L',  # Left
            all_face_colors[2][4]: 'B',  # Back
        }

        kociemba = []

        for c in cube_colors:
            if c not in center_map:
                self.error_text = f"Invalid cube: unknown color '{c}'"
                return None
            kociemba.append(center_map[c])

        return ''.join(kociemba)

    def send_to_arduino(self, solution):
        """
        Send solution moves to Arduino
        
        Args:
            solution: Space-separated string of moves
        """
        if self.arduino is None:
            print("\nSimulation mode - moves would be:")
            for move in solution.split():
                print(f"  -> {move}")
            return
        
        print("\n" + "="*50)
        print("SENDING TO ARDUINO")
        print("="*50)
        
        moves = solution.split()
        for idx, move in enumerate(moves, 1):
            print(f"Move {idx}/{len(moves)}: {move}")
            self.arduino.write(f"{move}\n".encode())
            time.sleep(0.5)  # Wait for move to complete
            
            # Read response from Arduino
            if self.arduino.in_waiting:
                response = self.arduino.readline().decode().strip()
                print(f"Arduino: {response}")
        
        print("\nSolution sent to Arduino!")
    

    def run(self):
        """Main execution loop"""
        try:
            self.init_webcam(4)
            solved = False
            while True:
                if not solved:
                     cube_string = self.scan_all_faces()
                if cube_string is None:
                    continue
                try:
                    solution = self.solve_cube(cube_string)
                    if solution:
                        self.solution_text = solution
                    solved = True
                except:
                    self.solution_text = "Not solved"
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = self.draw_cube_net(frame)
                frame = self.draw_status_overlay(frame)
                cv2.imshow("Rubik's Cube Scanner", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            # Solve the cube
            solution = self.solve_cube(cube_string)
            if solution:
                self.solution_text = solution
            # Display solution
            # self.display_solution(solution)
            
            # # Send to Arduino
            # response = input("\nSend solution to Arduino? (y/n): ")
            # if response.lower() == 'L':
            #     self.send_to_arduino(solution)
            
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        if self.arduino is not None:
            self.arduino.close()
        cv2.destroyAllWindows()
        print("\nCleanup complete")


def main():
    """Main entry point"""
    print("="*60)
    print(" RUBIK'S CUBE SOLVER WITH WEBCAM AND ARDUINO")
    print("="*60)
    
    # Create solver instance
    # You can specify Arduino port manually: solver = RubiksCubeSolver(arduino_port='/dev/ttyUSB0')
    solver = RubiksCubeSolver()
    
    # Run the solver
    solver.run()


if __name__ == "__main__":
    main()
