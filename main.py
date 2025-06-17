import cv2
import dlib
import numpy as np
import time
import json
import os
from collections import deque

# Constants
EAR_THRESHOLD = 0.25  # Eye aspect ratio threshold for closed eyes
CONSECUTIVE_FRAMES = 3  # Number of consecutive frames to confirm eye closure
LEADERBOARD_FILE = "leaderboard.json"
FONT = cv2.FONT_HERSHEY_SIMPLEX
COUNTDOWN_DURATION = 5  # 5-second countdown

def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio (EAR)"""
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    
    # Compute the EAR
    ear = (A + B) / (2.0 * C)
    return ear

def load_leaderboard():
    """Load leaderboard from JSON file or create if not exists"""
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_leaderboard(leaderboard):
    """Save leaderboard to JSON file"""
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(leaderboard, f, indent=2)

def update_leaderboard(name, time_elapsed):
    """Update leaderboard with new score"""
    leaderboard = load_leaderboard()
    
    # Add new entry
    leaderboard.append({"name": name, "time": time_elapsed})
    
    # Sort by time descending (longer times are better)
    leaderboard.sort(key=lambda x: x["time"], reverse=True)
    
    # Keep only top 10
    leaderboard = leaderboard[:10]
    
    save_leaderboard(leaderboard)
    return leaderboard

def display_leaderboard(frame, leaderboard, x=10, y=30, font_scale=0.7, color=(0, 255, 255), show_title=True):
    """Display leaderboard on the video frame"""
    if show_title:
        cv2.putText(frame, "LEADERBOARD:", (x, y), FONT, font_scale, color, 2)
        y += 40
    
    for i, entry in enumerate(leaderboard[:10]):  # Show top 10
        text = f"{i+1}. {entry['name']}: {entry['time']:.2f}s"
        cv2.putText(frame, text, (x, y), FONT, font_scale * 0.9, (0, 255, 0), 1)
        y += 30
    return y  # Return next y position

def draw_centered_text(frame, text, y_offset, font_scale=1, color=(0, 255, 0), thickness=2):
    """Draw text centered horizontally on the frame"""
    text_size = cv2.getTextSize(text, FONT, font_scale, thickness)[0]
    x = (frame.shape[1] - text_size[0]) // 2
    cv2.putText(frame, text, (x, y_offset), FONT, font_scale, color, thickness)
    return y_offset + text_size[1] + 20

def main():
    # Get player name
    player_name = input("Enter your name: ").strip()
    if not player_name:
        player_name = "Player"
    
    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Game variables
    timer_start = None
    time_elapsed = 0
    game_active = False
    game_over = False
    closed_frames = deque(maxlen=CONSECUTIVE_FRAMES)
    leaderboard = load_leaderboard()
    countdown_start = None
    countdown_active = False
    
    # Display instructions
    cv2.namedWindow("Stare Off Game", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stare Off Game", 1000, 700)
    
    print("Starting game... Stare at the camera without blinking!")
    print("Press 'ESC' to quit at any time")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        if len(faces) > 0:
            # Take the first face
            face = faces[0]
            
            # Get facial landmarks
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # Extract left and right eye coordinates
            left_eye = landmarks[42:48]
            right_eye = landmarks[36:42]
            
            # Calculate eye aspect ratio for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Draw eyes on the frame
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
            
            # Display EAR value
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, frame.shape[0] - 10), 
                        FONT, 0.6, (0, 255, 0), 1)
            
            # Check for eye closure
            if ear < EAR_THRESHOLD:
                closed_frames.append(True)
            else:
                closed_frames.append(False)
            
            # Start the countdown when eyes are open
            if not game_active and not countdown_active and not any(closed_frames):
                countdown_active = True
                countdown_start = time.time()
            
            # Handle countdown
            if countdown_active:
                countdown_remaining = COUNTDOWN_DURATION - (time.time() - countdown_start)
                
                if countdown_remaining > 0:
                    # Display large countdown number
                    count_text = str(int(np.ceil(countdown_remaining)))
                    text_size = cv2.getTextSize(count_text, FONT, 4, 8)[0]
                    x = (frame.shape[1] - text_size[0]) // 2
                    y = (frame.shape[0] + text_size[1]) // 2
                    cv2.putText(frame, count_text, (x, y), FONT, 4, (0, 0, 255), 8)
                else:
                    # Countdown finished - start the game
                    countdown_active = False
                    game_active = True
                    timer_start = time.time()
            
            # Update timer if game is active
            if game_active and not game_over:
                time_elapsed = time.time() - timer_start
                
                # Check if eyes have been closed for enough consecutive frames
                if all(closed_frames):
                    game_over = True
                    game_active = False
                    leaderboard = update_leaderboard(player_name, time_elapsed)
        
        # Display game information
        cv2.putText(frame, f"Player: {player_name}", (frame.shape[1] - 200, 30), 
                    FONT, 0.7, (255, 0, 255), 1)
        
        if countdown_active:
            cv2.putText(frame, "Get ready to stare!", (frame.shape[1] // 2 - 120, 100), 
                        FONT, 0.9, (0, 255, 255), 2)
        
        if game_active and not game_over:
            cv2.putText(frame, f"Time: {time_elapsed:.2f}s", (frame.shape[1] - 200, 60), 
                        FONT, 0.7, (255, 0, 255), 1)
            cv2.putText(frame, "STARE!", (frame.shape[1] // 2 - 50, 100), 
                        FONT, 1.2, (0, 0, 255), 2)
        
        if game_over:
            # Display game over message
            cv2.putText(frame, "GAME OVER!", (frame.shape[1] // 2 - 100, 60), 
                        FONT, 1.5, (0, 0, 255), 3)
            
            # Display player's time
            cv2.putText(frame, f"Your Time: {time_elapsed:.2f}s", 
                        (frame.shape[1] // 2 - 120, 120), 
                        FONT, 1, (0, 255, 255), 2)
            
            # Display leaderboard at the end
            y_pos = draw_centered_text(frame, "--- LEADERBOARD ---", 180, 1.2, (0, 255, 255))
            display_leaderboard(frame, leaderboard, x=(frame.shape[1] - 300) // 2, y=y_pos, 
                               font_scale=0.9, show_title=False)
            
            # Display restart instructions
            cv2.putText(frame, "Press 'R' to restart or 'ESC' to quit", 
                        (frame.shape[1] // 2 - 200, frame.shape[0] - 30), 
                        FONT, 0.7, (255, 255, 0), 2)
        
        # Display mini leaderboard during gameplay
        if not game_over:
            display_leaderboard(frame, leaderboard, x=10, y=30, font_scale=0.5)
        
        # Show frame
        cv2.imshow("Stare Off Game", frame)
        
        # Check for keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('r') and game_over:  # Restart game
            game_active = False
            game_over = False
            countdown_active = False
            time_elapsed = 0
            closed_frames.clear()
            leaderboard = load_leaderboard()
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Show final leaderboard in console
    print("\nFinal Leaderboard:")
    leaderboard = load_leaderboard()
    for i, entry in enumerate(leaderboard):
        print(f"{i+1}. {entry['name']}: {entry['time']:.2f} seconds")

if __name__ == "__main__":
    # Check for required model file
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("\nERROR: Required model file not found.")
        print("Please download 'shape_predictor_68_face_landmarks.dat' from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract it and place in the same directory as this script.")
    else:
        main()
