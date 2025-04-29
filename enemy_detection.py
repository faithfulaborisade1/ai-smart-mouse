import cv2
import numpy as np

class EnemyDetector:
    def __init__(self, detection_method="color"):
        self.detection_method = detection_method
        
        # For color-based detection (customize for your game)
        # These values need to be adjusted based on enemy colors in your specific game
        self.lower_enemy_color = np.array([0, 100, 100])  # Example HSV range for reddish enemies
        self.upper_enemy_color = np.array([10, 255, 255])
        
        # For template matching
        self.templates = []
        
    def load_templates(self, template_paths):
        """Load template images for template matching"""
        self.templates = [cv2.imread(path, cv2.IMREAD_COLOR) for path in template_paths]
        
    def detect_enemies_color(self, frame):
        """Detect enemies based on color thresholding"""
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for enemy colors
        mask = cv2.inRange(hsv, self.lower_enemy_color, self.upper_enemy_color)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (eliminate small noise)
        min_contour_area = 100  # Adjust based on your needs
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        
        # Extract bounding boxes for detected enemies
        enemy_boxes = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            enemy_boxes.append((x, y, w, h))
            
        return enemy_boxes, mask
    
    def detect_enemies_template(self, frame):
        """Detect enemies using template matching"""
        enemy_locations = []
        
        for template in self.templates:
            # Apply template matching
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            
            # Get positions where match exceeds threshold
            threshold = 0.7  # Adjust based on your needs
            locations = np.where(result >= threshold)
            
            # Extract coordinates
            for pt in zip(*locations[::-1]):
                w, h = template.shape[1::-1]
                enemy_locations.append((pt[0], pt[1], w, h))
                
        return enemy_locations, None
    
    def detect_enemies(self, frame):
        """Detect enemies in the given frame"""
        if self.detection_method == "color":
            return self.detect_enemies_color(frame)
        elif self.detection_method == "template":
            return self.detect_enemies_template(frame)
        else:
            raise ValueError(f"Unsupported detection method: {self.detection_method}")
    
    def highlight_enemies(self, frame, enemy_boxes):
        """Draw bounding boxes around detected enemies"""
        result_frame = frame.copy()
        for x, y, w, h in enemy_boxes:
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Mark center (approximately where you'd aim)
            center_x, center_y = x + w // 2, y + h // 3  # Aim for upper third (head area)
            cv2.circle(result_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
        return result_frame

# Example usage
if __name__ == "__main__":
    # Load a test image
    test_image = cv2.imread("test_game_screenshot.jpg")
    
    # Create detector and detect enemies
    detector = EnemyDetector(detection_method="color")
    enemy_boxes, _ = detector.detect_enemies(test_image)
    
    # Highlight enemies in the image
    result = detector.highlight_enemies(test_image, enemy_boxes)
    
    # Display result
    cv2.imshow("Detected Enemies", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()