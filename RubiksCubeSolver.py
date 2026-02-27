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
from ArduinoCommunication import ArduinoCommunication

class RubiksCubeSolver:
    def __init__(self, webcam, arduino_port=None, baudrate=9600):
        """
        Initialize the Rubik's Cube Solver
        
        Args:
            arduino_port: Serial port for Arduino (auto-detect if None)
            baudrate: Serial communication speed
        """
        self.camidx = webcam
        self.solution_text = None
        self.error_text = None  

        self.cap = None
        self.cube_state = [''] * 54 
        self.face_colors = []
        self.current_face = 0
        self.arduino = ArduinoCommunication(baudrate)
        self.face_names = ['Front', 'Right', 'Back', 'Left', 'Up', 'Down']
        self.color_ranges = {
            'W': ([0, 0, 200], [180, 40, 255]),

            'Y': ([22, 120, 120], [32, 255, 255]),

            'R1': ([0, 120, 100], [8, 255, 255]),
            'R2': ([170, 120, 100], [180, 255, 255]),

            'O': ([10, 130, 120], [20, 255, 255]),

            'G': ([40, 80, 80], [75, 255, 255]),

            'B': ([95, 80, 80], [125, 255, 255])
        }
    
    
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
                    f"Solution: Already Solved!",
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
            'Y': (0, 255, 255),   # yellow
            'O': (0, 165, 255),   # orange
            'W': (255, 255, 255), # white
            'B': (255, 0, 0),     # blue
            'G': (0, 255, 0),     # green
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
    
    def raw_to_kociemba(self, raw):
        """
        Convert cube string from FRBLDU order
        to Kociemba URFDLB order.

        raw: 54-character string
        returns: 54-character string in URFDLB order
        """

        if len(raw) != 54:
            raise ValueError("Cube string must be 54 characters")

        # Slice faces from FRBLDU
        F = raw[0:9]
        R = raw[9:18]
        B = raw[18:27]
        L = raw[27:36]
        D = raw[36:45]
        U = raw[45:54]

        # Reassemble into URFDLB
        kociemba_string = U + R + F + D + L + B

        return kociemba_string

    def color_to_face_mapping(self, raw):
        """
        Convert color-based string to URFDLB letters
        using center stickers as reference.
        """

        if len(raw) != 54:
            raise ValueError("Cube string must be 54 characters")

        # Extract centers (index 4 of each face block in FRBLDU order)
        centers = {
            raw[4]: 'F',
            raw[13]: 'R',
            raw[22]: 'B',
            raw[31]: 'L',
            raw[40]: 'D',
            raw[49]: 'U',
        }

        converted = ''.join(centers[c] for c in raw)
        return converted
    
    def scan_face(self, expected_done=1, display_window="Rubik's Cube Scanner"):
        """
        Scan one face of the cube.
        Capture occurs automatically after Arduino sends DONE.
        """
        done_count = 0
        if self.cap is None:
            self.init_webcam(1)

        if self.arduino is None:
            print("Error: Arduino not connected.")
            return None

        face_colors = []
        start_time = time.time()
        timeout = 10  # seconds

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            stickers = self.get_sticker_positions(width, height)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Continuously compute temp colors for preview
            temp_colors = []
            for (x, y, w, h) in stickers:
                roi = hsv[y:y+h, x:x+w]
                color = self.detect_color(roi)
                temp_colors.append(color)

            display_frame = self.draw_sticker_grid(frame.copy(), stickers, temp_colors)
            display_frame = self.draw_cube_net(display_frame)
            display_frame = self.draw_status_overlay(display_frame)

            face_name = self.face_names[self.current_face] if self.current_face < 6 else "Done"

            cv2.putText(display_frame,
                        f"Waiting for Arduino... ({face_name} Face {self.current_face + 1}/6)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)

            cv2.imshow(display_window, display_frame)
            cv2.waitKey(1)

            # 🔹 Check Arduino signal
            if self.arduino.arduino.in_waiting:
                response = self.arduino.arduino.readline().decode().strip()
                if response == "DONE":
                    done_count += 1
                    if done_count == expected_done:
                        face_colors = temp_colors.copy()
                        return face_colors
            
            # 🔹 Timeout protection
            if time.time() - start_time > timeout:
                print("Timeout waiting for Arduino")
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
        if not self.arduino:
            self.arduino = ArduinoCommunication()
        move = [
            ['d'],
            ['x'],
            ['x'],
            ['x'],
            ['i', 'x'],
            ['y', 'p']
        ]

        for face_idx in range(6):
            self.current_face = face_idx
            print(f"\nPosition {self.face_names[face_idx]} face toward camera...")

            # Send each command non-blocking
            for cmd in move[face_idx]:
                self.arduino.send_single_move(cmd)

            # Now wait + capture inside scan_face()
            face_colors = self.scan_face(expected_done=len(move[face_idx]))

            if face_colors is None:
                print("Scanning cancelled")
                return None

            all_face_colors.append(face_colors)
            self.face_colors = all_face_colors.copy()
        cube_string = ''
        for face_idx in range(6):
            cube_string += ''.join(all_face_colors[face_idx])
        
        print(f"\nCube colors: {cube_string}")
        mapped = self.color_to_face_mapping(cube_string)
        kociemba_ready = self.raw_to_kociemba(mapped)
        return kociemba_ready
    


    
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

    def run(self):
        """Main execution loop"""
        try:
            self.init_webcam(self.camidx)
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
                
            solution = self.solve_cube(cube_string)
            if solution:
                self.solution_text = solution
            return solution
            
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
        if self.arduino:
            self.arduino.arduino.close()
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
