# Computer Vision Aim Assistant

This project is an educational exploration of computer vision techniques applied to FPS games. It demonstrates how computer vision can be used to analyze game visuals and control mouse movements.

**DISCLAIMER: This project is for EDUCATIONAL PURPOSES ONLY. Using this or similar software in online competitive games may violate terms of service and result in account bans. The creator does not endorse cheating in any form.**

## Project Overview

This aim assistant uses computer vision techniques to:

1. Capture the game screen in real-time
2. Detect enemy players using either color detection or AI-based object detection
3. Calculate optimal aim points (e.g., enemy heads)
4. Control mouse movement to assist with aiming
5. Optionally automate firing when on target

## Features

- Real-time screen capture at high FPS
- Multiple detection methods:
  - Basic color-based detection (faster, less accurate)
  - Advanced YOLO-based AI detection (slower, more accurate)
- Smooth, human-like mouse movement
- Configurable settings for sensitivity and behavior
- Toggle controls via hotkeys
- Visual debugging display

## Project Structure

```
aim-assistant/
├── screen_capture.py     # Screen capture module
├── enemy_detection.py    # Basic enemy detection
├── yolo_detector.py      # Advanced AI-based detection (optional)
├── mouse_control.py      # Mouse movement control
├── main.py               # Main integration script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup and Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```
4. For advanced YOLO detection (optional):
   ```
   pip install -U torch torchvision
   ```

## Usage

1. Start the program:
   ```
   python main.py
   ```

2. Controls:
   - F6: Toggle aim assistance on/off
   - F7: Toggle auto-trigger on/off
   - F8: Toggle visualization display
   - Ctrl+C: Exit program

## Customization

### Game-Specific Settings

You'll need to adjust several parameters based on your specific game:

1. Enemy detection colors in `enemy_detection.py`
2. Screen capture region in `screen_capture.py`
3. Mouse sensitivity in `mouse_control.py`

### Training Custom Detection Models

For better results, you can train a custom YOLO model specific to your game:

1. Collect screenshots from your game
2. Annotate enemies using a tool like [labelImg](https://github.com/tzutalin/labelImg)
3. Train a YOLOv5 model using the provided annotations
4. Replace the default model with your custom one

See the instructions in `yolo_detector.py` for more details.

## Learning Objectives

This project demonstrates several important computer vision and automation concepts:

- Real-time image processing and analysis
- Object detection techniques
- Control systems for mouse movement
- Multi-threaded programming for performance
- Game-specific adaptations of general techniques

## Future Improvements

- Implement recoil compensation
- Add movement prediction for moving targets
- Create a more user-friendly configuration interface
- Optimize for better performance
- Expand to support more games

## Contributing

This is an educational project. If you have improvements or suggestions, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.