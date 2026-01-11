# Import necessary libraries
import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
import mediapipe as mp

# Check if we are running on a Raspberry Pi to enable GPIO
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    BUZZER_PIN = 18
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    USING_PI = True
    print("[INFO] Raspberry Pi GPIO detected.")
except ImportError:
    USING_PI = False
    print("[INFO] Not running on Raspberry Pi. Using system sound.")
    import platform
    if platform.system() == "Windows":
        import winsound

# Define constants
EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False

# Define the eye aspect ratio function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define the sound alarm function
def sound_alarm(on):
    if USING_PI:
        if on:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
        else:
            GPIO.output(BUZZER_PIN, GPIO.LOW)
    else:
        if on and platform.system() == "Windows":
            winsound.Beep(2500, 50)

# Initialize MediaPipe Face Mesh
print("[INFO] loading facial landmark detector...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Start the video stream
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for speed
    frame = cv2.resize(frame, (500, int(500 * frame.shape[0] / frame.shape[1])))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Extract eye coordinates
            leftEye = np.array([(int(face_landmarks.landmark[i].x * w), 
                                int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE])
            rightEye = np.array([(int(face_landmarks.landmark[i].x * w), 
                                 int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE])
            
            # Calculate the EAR
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            # Visualization
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # Decision logic
            if ear < EAR_THRESH:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        sound_alarm(True)
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                ALARM_ON = False
                sound_alarm(False)
            
            # Display EAR on screen
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()
if USING_PI:
    GPIO.cleanup()