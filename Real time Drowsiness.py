import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time
import platform
import threading

# For Windows beep
if platform.system() == "Windows":
    import winsound

# Constants
EYE_AR_THRESH = 0.25             # EAR below this means the eye is closed
EYE_AR_CONSEC_FRAMES = 48        # Number of consecutive frames (approx. 2 sec at 24 FPS)

# Initialize frame counters and alarm status
COUNTER = 0
ALARM_ON = False

# MediaPipe Face Mesh landmark indexes for eyes
# Left eye: [362, 385, 387, 263, 373, 380]
# Right eye: [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(eye):
    """
    Compute the Eye Aspect Ratio (EAR) for an eye.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        eye (array-like): Array of 6 (x, y) landmark points for one eye

    Returns:
        float: Eye Aspect Ratio
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def sound_alarm():
    """
    Sound an alarm beep. On Windows, uses winsound.Beep.
    On other systems, prints ASCII Bell character.
    """
    if platform.system() == "Windows":
        winsound.Beep(2500, 1000)  # Frequency 2500 Hz, duration 1000 ms
    else:
        print('\a')  # ASCII Bell - may or may not work depending on terminal

def main():
    global COUNTER, ALARM_ON

    print("[INFO] Initializing MediaPipe Face Mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("[INFO] Starting video stream...")
    cap = cv2.VideoCapture(0)  # Default camera index 0
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    time.sleep(1.0)  # Warm-up camera

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        # Get frame dimensions
        h, w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmark coordinates
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))

                # Get eye landmarks
                left_eye = np.array([landmarks[i] for i in LEFT_EYE_IDX])
                right_eye = np.array([landmarks[i] for i in RIGHT_EYE_IDX])

                # Calculate EAR for each eye
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                # Average EAR
                ear = (left_ear + right_ear) / 2.0

                # Draw eye contours
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                # Check if eyes are closed
                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True
                            t = threading.Thread(target=sound_alarm)
                            t.daemon = True
                            t.start()

                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    COUNTER = 0
                    ALARM_ON = False

                # Display EAR
                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Eye Blink Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()