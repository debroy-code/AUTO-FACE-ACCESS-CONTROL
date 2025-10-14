import cv2
import os
import time

# Create dataset folder if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

print("=== Face Enrollment System ===")
name = input("Enter the name of the person: ").strip()

# Create folder for this person
person_folder = f'dataset/{name}'
if not os.path.exists(person_folder):
    os.makedirs(person_folder)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# CORRECTED LINE: Use CascadeClassifier not Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f"Taking photos of {name}. Look at the camera and move slightly.")
print("Press 'c' to capture manually or wait for auto-capture...")
print("Press 'q' to finish enrollment.")

count = 0

while count < 30:  # Capture exactly 30 images
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Photos: {count}/30', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Enrollment - Press Q to quit, C to capture', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('c') and len(faces) == 1:
        img_name = f"{person_folder}/img_{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        count += 1
    
    # Auto-capture
    elif len(faces) == 1 and count < 30:
        img_name = f"{person_folder}/img_{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Auto-saved: {img_name}")
        count += 1
        time.sleep(0.5)  # Delay between captures

print(f"Enrollment completed! Saved {count} images")
cap.release()
cv2.destroyAllWindows()
