# Stare Off Game: Eye Blink Detection Challenge
## Overview
Stare Off is an interactive computer vision game that challenges players to stare at the camera without blinking for as long as possible. Using facial landmark detection and eye aspect ratio calculations, the game tracks eye blinks and records players' staring endurance times. Top scores are maintained in a persistent leaderboard system.

## Key Features
- Real-time eye blink detection using dlib's facial landmarks

- Dynamic leaderboard with top 10 scores persistence

- 5-second countdown before game start

- Mirror-mode camera feed with visual overlays

- Eye Aspect Ratio (EAR) visualization

- Game restart functionality

## Requirements
- Python 3.6+

- Webcam

- Required model file: shape_predictor_68_face_landmarks.dat

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/stareoff-game.git
cd stareoff-game

# Install dependencies
pip install -r requirements.txt

# Download facial landmark model (required)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Alternative Download Methods
1. Visit [shape_predictor_68_face_landmarks.dat](https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat)
2. Download the file
3. Place in project directory

## How to Play
1. Run the game: ```python main.py```

2. Enter your name when prompted

3. Position yourself facing the camera

4. Keep your eyes open during the 5-second countdown

5. Stare continuously at the camera without blinking

6. The game ends when you blink for 3 consecutive frames

7. Press 'R' to restart or ESC to quit

## Game Mechanics
- Eye Aspect Ratio (EAR) Calculation:

```py
def eye_aspect_ratio(eye):
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)
```
- **Blink Detection**: Eyes are considered closed when EAR < 0.25 for 3 consecutive frames

- **Timing**: Game duration is measured from countdown end to first detected blink

- **Leaderboard**: Top 10 scores saved in leaderboard.json

## File Structure
```
stareoff-game/
├── main.py             # Main game logic
├── leaderboard.json     # Score database (auto-generated)
├── requirements.txt     # Dependencies list
└── shape_predictor_68_face_landmarks.dat  # Facial landmark model
```
## Troubleshooting
**Issue**: "Required model file not found"
**Solution**: Ensure ```shape_predictor_68_face_landmarks.dat``` is in the same directory as the script

**Issue**: Camera not detected
**Solution**:

1. Check camera connection

2. Verify other apps aren't using the camera

3. Try changing camera index in ```cv2.VideoCapture(0)``` to 1 or 2

**Issue**: Poor blink detection
**Solution**:

1. Ensure good lighting on your face

2. Position face centrally in camera view

3. Adjust ```EAR_THRESHOLD``` in code (0.15-0.3)

4. Clean camera lens

## Customization Options
Modify these constants in the code:

```py
EAR_THRESHOLD = 0.25       # Lower = more sensitive blink detection
CONSECUTIVE_FRAMES = 3     # Frames needed to register blink
COUNTDOWN_DURATION = 5     # Pre-game countdown in seconds
LEADERBOARD_FILE = "scores.json"  # Leaderboard filename
```
## Leaderboard Format
Scores are saved in JSON format:

```json
[
  {"name": "Player1", "time": 12.34},
  {"name": "Player2", "time": 9.87}
]
```
## Dependencies
- OpenCV (```pip install opencv-python```)

- dlib (```pip install dlib```)

- NumPy (```pip install numpy```)

## Future Enhancements
- Multiple difficulty levels

- Sound effects

- Calibration mode for personalized EAR thresholds

- Networked multiplayer mode

- Integration with social media sharing

## Contributing
Contributions are welcome! Please submit pull requests for:

- Improved documentation

- Bug fixes

- New features

- Performance optimizations
