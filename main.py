import cv2
import numpy as np
import time
import keyboard
import threading

# Import our custom modules
from screen_capture import ScreenCapture
from enemy_detection import EnemyDetector
from mouse_control import MouseController

class AimAssistant:
    def __init__(self):
        # Initialize components
        self.screen_capture = ScreenCapture()
        self.enemy_detector = EnemyDetector(detection_method="color")
        self.mouse_controller = MouseController(smoothing=True, smoothing_factor=0.3)
        
        # Configure parameters
        self.running = False
        self.display_detection = True
        
        # Game-specific settings
        self.screen_center_x = self.screen_capture.monitor['width'] // 2
        self.screen_center_y = self.screen_capture.monitor['height'] // 2
        
        # Set up hotkeys
        keyboard.add_hotkey('f6', self.toggle_running)
        keyboard.add_hotkey('f7', self.mouse_controller.toggle_trigger)
        keyboard.add_hotkey('f8', self.toggle_display)
    
    def toggle_running(self):
        """Toggle the aim assistant on/off"""
        self.running = not self.running
        self.mouse_controller.enabled = self.running
        print(f"Aim assistant {'enabled' if self.running else 'disabled'}")
    
    def toggle_display(self):
        """Toggle detection visualization"""
        self.display_detection = not self.display_detection
        print(f"Detection display {'enabled' if self.display_detection else 'disabled'}")
    
    def process_frame(self, frame):
        """Process a single frame from the screen capture"""
        # Detect enemies
        enemy_boxes, mask = self.enemy_detector.detect_enemies(frame)
        
        # Find closest enemy to crosshair (screen center)
        closest_enemy = None
        closest_distance = float('inf')
        
        for x, y, w, h in enemy_boxes:
            # Calculate enemy "head" position (upper third of bounding box)
            enemy_x = x + w // 2
            enemy_y = y + h // 3
            
            # Calculate distance to crosshair
            distance = np.sqrt((enemy_x - self.screen_center_x)**2 + 
                              (enemy_y - self.screen_center_y)**2)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_enemy = (enemy_x, enemy_y)
        
        # Visualize detection if enabled
        if self.display_detection:
            # Create visualization frame
            vis_frame = self.enemy_detector.highlight_enemies(frame, enemy_boxes)
            
            # Mark screen center (crosshair)
            cv2.circle(vis_frame, 
                      (self.screen_center_x, self.screen_center_y), 
                      10, (255, 0, 0), 2)
            
            # Display closest enemy
            if closest_enemy:
                cv2.circle(vis_frame, closest_enemy, 15, (0, 0, 255), -1)
            
            # Show on screen (scaled down)
            display_frame = cv2.resize(vis_frame, (960, 540))
            cv2.imshow("Aim Assistant Detection", display_frame)
            cv2.waitKey(1)
        elif cv2.getWindowProperty("Aim Assistant Detection", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("Aim Assistant Detection")
        
        # Move mouse to target if running
        if self.running and closest_enemy:
            self.mouse_controller.move_to_target(*closest_enemy)
            self.mouse_controller.fire_if_on_target(*closest_enemy)
    
    def run(self):
        """Main loop for the aim assistant"""
        print("Aim Assistant started")
        print("Press F6 to toggle aim assist")
        print("Press F7 to toggle auto-trigger")
        print("Press F8 to toggle visualization")
        print("Press Ctrl+C to exit")
        
        # Create a separate thread for mouse updates
        def mouse_update_loop():
            while True:
                self.mouse_controller.update()
                time.sleep(0.01)  # 100 updates per second
                
        mouse_thread = threading.Thread(target=mouse_update_loop, daemon=True)
        mouse_thread.start()
        
        # Start screen capture loop
        try:
            self.screen_capture.start_capture_loop(self.process_frame)
        except KeyboardInterrupt:
            print("Exiting...")
            cv2.destroyAllWindows()

if __name__ == "__main__":
    assistant = AimAssistant()
    assistant.run()