import mss
import numpy as np
import cv2
import time

class ScreenCapture:
    def __init__(self, monitor_number=1):
        self.sct = mss.mss()
        # You'll need to adjust these values based on your setup
        # Default to primary monitor
        self.monitor = self.sct.monitors[monitor_number]
        
    def capture_frame(self):
        """Capture a single frame from the screen"""
        sct_img = self.sct.grab(self.monitor)
        # Convert to numpy array for OpenCV processing
        return np.array(sct_img)
    
    def start_capture_loop(self, callback, fps_target=60):
        """Start a continuous capture loop at specified FPS"""
        frame_time = 1/fps_target
        try:
            last_time = time.time()
            while True:
                # Capture screen
                frame = self.capture_frame()
                
                # Process frame through callback
                callback(frame)
                
                # Calculate sleep time to maintain target FPS
                current_time = time.time()
                elapsed = current_time - last_time
                sleep_time = max(0, frame_time - elapsed)
                time.sleep(sleep_time)
                
                # Update FPS tracking
                last_time = current_time
                actual_fps = 1 / (time.time() - current_time + sleep_time)
                if int(time.time()) % 5 == 0:  # Print FPS every 5 seconds
                    print(f"FPS: {actual_fps:.1f}")
                
        except KeyboardInterrupt:
            print("Capture stopped by user")

# Example usage
if __name__ == "__main__":
    # Simple callback that displays the captured frame
    def display_frame(frame):
        # Resize for display (optional)
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Game Capture", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt
    
    # Create and start capture
    capture = ScreenCapture()
    capture.start_capture_loop(display_frame)