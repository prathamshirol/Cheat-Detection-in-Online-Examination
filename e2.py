import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Constants for eye aspect ratio to indicate blinking
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3

# Define counters for blinks
blink_counter = 0
blink_total = 0

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load YOLOv8 model
model = YOLO("best.pt")  # or the path to your YOLOv8 model

# Gadgets that are not allowed during exams
not_allowed_gadgets = ["cell phone", "laptop", "tablet"]

# Counters for detections
detections_count = {label: 0 for label in not_allowed_gadgets + ["person"]}
total_frames = 0
correct_detections = 0

# Load laptop camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect objects using YOLOv8
    results = model(frame_rgb)  # perform detection
    
    # Initialize lists to store detected objects
    detected_faces = []
    detected_gadgets = []
    detected_bluetooth_headphones = []

    # Process YOLO detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = box.cls
            label = model.names[int(cls)]
            x1, y1, x2, y2 = [int(val) for val in box.xyxy.tolist()[0]]
            coords = (x1, y1, x2, y2)
            if label == "person":
                detected_faces.append(coords)
                detections_count["person"] += 1
            elif label in not_allowed_gadgets:
                detected_gadgets.append(coords)
                detections_count[label] += 1
            elif label == "bluetooth headphone":
                detected_bluetooth_headphones.append(coords)
                detections_count["bluetooth headphone"] += 1

    # Process face mesh for  gaze direction
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Get the eye landmarks
            left_eye_landmarks = np.array([(landmarks[i].x, landmarks[i].y) for i in [33, 160, 158, 133, 153, 144]])
            right_eye_landmarks = np.array([(landmarks[i].x, landmarks[i].y) for i in [362, 385, 387, 263, 373, 380]])

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)

            ear = (left_ear + right_ear) / 2.0

            # Check for blinking
            if ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_total += 1
                    cv2.putText(frame, 'Blink Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                blink_counter = 0

            # Get the nose landmark
            nose = landmarks[1]

            # Calculate the gaze direction
            nose_point = np.array([nose.x, nose.y])
            eye_center = (left_eye_landmarks[0] + right_eye_landmarks[3]) / 2

            dx = nose_point[0] - eye_center[0]
            dy = nose_point[1] - eye_center[1]

            if dx > 0.02:
                gaze_direction = "Looking Left"
            elif dx < -0.02:
                gaze_direction = "Looking Right"
            elif dy > 0.02:
                gaze_direction = "Looking Down"
            elif dy < -0.02:
                gaze_direction = "Looking Up"
            else:
                gaze_direction = "Looking Forward"

            cv2.putText(frame, gaze_direction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw the eye landmarks
            for (x, y) in left_eye_landmarks:
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, (0, 255, 0), -1)
            for (x, y) in right_eye_landmarks:
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 2, (0, 255, 0), -1)

    # Draw bounding boxes around detected objects
    for (x1, y1, x2, y2) in detected_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for (x1, y1, x2, y2) in detected_gadgets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, 'Gadget', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for (x1, y1, x2, y2) in detected_bluetooth_headphones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, 'Bluetooth Headphone', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Check for multiple faces and print alert
    if len(detected_faces) > 1:
        cv2.putText(frame, 'Alert: Multiple faces detected!', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Print detection counts and total frames
print(f"Total frames processed: {total_frames}")
for label, count in detections_count.items():
    print(f"Total {label} detected: {count}")

# Print detection rate (proxy for accuracy)
if total_frames > 0:
    detection_rate = sum(detections_count.values()) / total_frames
    print(f"Detection rate (proxy for accuracy): {detection_rate:.2f}")
