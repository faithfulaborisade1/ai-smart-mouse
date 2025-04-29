import pyautogui
import numpy as np
import time
import keyboard

class MouseController:
    def __init__(self, smoothing=True, smoothing_factor=0.5, max_speed=100):
        """
        Initialize the mouse controller
        
        Args:
            smoothing: Whether to use smoothed movement
            smoothing_factor: How smooth the movement should be (0-1, higher = smoother)
            max_speed: Maximum pixels to move per step
        """
        # Disable pyautogui safety
        pyautogui.FAILSAFE = False
        
        self.smoothing = smoothing
        self.smoothing_factor = smoothing_factor
        self.max_speed = max_speed
        
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize target position as current position
        self.current_x, self.current_y = pyautogui.position()
        self.target_x, self.target_y = self.current_x, self.current_y
        
        # Flag to enable/disable control
        self.enabled = False
        self.trigger_enabled = False
        
    def toggle_enabled(self):
        """Toggle whether the controller is active"""
        self.enabled = not self.enabled
        print(f"Mouse control {'enabled' if self.enabled else 'disabled'}")
        
    def toggle_trigger(self):
        """Toggle whether auto-firing is active"""
        self.trigger_enabled = not self.trigger_enabled
        print(f"Auto-trigger {'enabled' if self.trigger_enabled else 'disabled'}")
    
    def move_to_target(self, x, y):
        """Set a new target position for the mouse"""
        if not self.enabled:
            return
            
        # Update target position
        self.target_x, self.target_y = x, y
        
    def update(self):
        """Update mouse position based on current target"""
        if not self.enabled:
            return
            
        # Get current position
        self.current_x, self.current_y = pyautogui.position()
        
        if self.smoothing:
            # Calculate movement vector
            dx = self.target_x - self.current_x
            dy = self.target_y - self.current_y
            
            # Apply smoothing
            move_x = dx * self.smoothing_factor
            move_y = dy * self.smoothing_factor
            
            # Limit max speed
            distance = np.sqrt(move_x**2 + move_y**2)
            if distance > self.max_speed:
                scale = self.max_speed / distance
                move_x *= scale
                move_y *= scale
                
            # Move mouse
            pyautogui.moveRel(move_x, move_y)
        else:
            # Move directly to target
            pyautogui.moveTo(self.target_x, self.target_y)
    
    def fire_if_on_target(self, target_x, target_y, tolerance=10):
        """Fire if mouse is close enough to target"""
        if not (self.enabled and self.trigger_enabled):
            return False
            
        current_x, current_y = pyautogui.position()
        distance = np.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
        
        if distance <= tolerance:
            pyautogui.click()
            return True
        return False

# Example usage
if __name__ == "__main__":
    controller = MouseController(smoothing=True, smoothing_factor=0.3)
    
    # Set up hotkeys
    keyboard.add_hotkey('f6', controller.toggle_enabled)
    keyboard.add_hotkey('f7', controller.toggle_trigger)
    
    print("Press F6 to toggle mouse control")
    print("Press F7 to toggle auto-trigger")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # This would be integrated with the enemy detection
            # For testing, move to center of screen
            if controller.enabled:
                controller.move_to_target(960, 540)
                controller.update()
            time.sleep(0.01)  # 100 updates per second
    except KeyboardInterrupt:
        print("Exiting...")