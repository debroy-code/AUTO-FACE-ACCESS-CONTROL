import cv2
import os
import numpy as np

# ========================
# INITIAL SETUP
# ========================
print("[INFO] Loading face recognizer and cascade...")

# Load the trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Check if model file exists
if not os.path.exists('trainer/trainer.yml'):
    print("[ERROR] Trainer file not found! Please train the model first.")
    exit()

recognizer.read('trainer/trainer.yml')

# Load Haar Cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print("[ERROR] Haarcascade file missing!")
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)

# Font for text
font = cv2.FONT_HERSHEY_SIMPLEX

# Names corresponding to training IDs
names = ['None', 'JIT', 'ABHISHEK', 'SNEHA', 'Vasundhara']  # Add more as trained

# ========================
# CAMERA SETUP
# ========================
print("[INFO] Starting camera...")
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[ERROR] Could not access the camera.")
    exit()

cam.set(3, 640)  # Width
cam.set(4, 480)  # Height

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# ========================
# MAIN LOOP
# ========================
print("[INFO] Press 'q' or 'ESC' to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Resize for performance (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve lighting/contrast
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        # Draw a rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 100), 2)

        # Predict face ID and confidence
        id_, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Lower confidence means more accurate
        if confidence < 80:
            id_name = names[id_] if id_ < len(names) else "Unknown"
            confidence_text = f"{round(100 - confidence)}%"
            color = (0, 255, 0)  # Green for known
        else:
            id_name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"
            color = (0, 0, 255)  # Red for unknown

        # Display name and confidence
        cv2.putText(frame, id_name, (x + 5, y - 10), font, 1, color, 2)
        cv2.putText(frame, confidence_text, (x + 5, y + h - 10), font, 0.8, (255, 255, 0), 1)

    # Show the video feed
    cv2.imshow("Face Recognition System", frame)

    # Exit condition
    k = cv2.waitKey(10) & 0xFF
    if k == 27 or k == ord('q'):
        break

# ========================
# CLEANUP
# ========================
print("\n[INFO] Exiting Program and Cleaning Up...")
cam.release()
cv2.destroyAllWindows()
