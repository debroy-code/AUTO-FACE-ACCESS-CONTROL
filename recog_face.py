import cv2
import os
import time
import pickle
import json # Import json
from datetime import datetime

# ---------------------------
# Helper: simple fallback logger
# ---------------------------
def simple_log_attendance(student_id, student_name, action, attendance_file='attendance_log.csv'):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    entry = f"{date_str},{time_str},{student_id},{student_name},{action}\n"
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write("Date,Time,Student_ID,Student_Name,Action\n")
    with open(attendance_file, 'a') as f:
        f.write(entry)
    print(f"📝 {action}: {student_name} ({student_id}) at {time_str}")

# ---------------------------
# Try to use AttendanceSystem (preferred)
# ---------------------------
attendance_system = None
face_db = None
try:
    from attendance_logger import AttendanceSystem
    attendance_system = AttendanceSystem()
    face_db = attendance_system.face_data
    print("[INFO] Using attendance_logger.AttendanceSystem for logging.")
except SystemExit:
    print("[WARN] attendance_logger found no face_database.pkl. Falling back to simple logging.")
    attendance_system = None
except Exception as e:
    print(f"[WARN] Could not import AttendanceSystem ({e}). Falling back to simple logging.")
    attendance_system = None

# ---------------------------
# Load LBPH recognizer
# ---------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
trainer_path = 'trainer/trainer.yml'
if not os.path.exists(trainer_path):
    print(f"[ERROR] Trainer file '{trainer_path}' not found. Please run train_model.py first.")
    exit(1)
recognizer.read(trainer_path)
print(f"[INFO] Loaded trainer model from {trainer_path}")

# ---------------------------
# DYNAMICALLY LOAD NAMES (No manual editing)
# ---------------------------
MAP_FILE = 'id_to_name_map.json'
names = []
if os.path.exists(MAP_FILE):
    with open(MAP_FILE, 'r') as f:
        id_to_name_map = json.load(f)
    
    # Rebuild the 'names' list based on the numeric IDs
    # Find the highest ID to create the list size
    max_id = max(int(k) for k in id_to_name_map.keys())
    names = ["Unknown"] * (max_id + 1) # Create list
    
    for numeric_id_str, name in id_to_name_map.items():
        names[int(numeric_id_str)] = name
        
    print(f"[INFO] Dynamically loaded names: {names}")
else:
    print(f"[ERROR] Name map file '{MAP_FILE}' not found. Please run combined_enrollment.py first.")
    exit(1)

# ---------------------------
# Setup cascade, camera, etc.
# ---------------------------
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# ---------------------------
# Build name -> student_id mapping (from face_database.pkl)
# ---------------------------
name_to_id = {}
if face_db:
    try:
        for i, nm in enumerate(face_db['names']):
            name_to_id[nm] = face_db['ids'][i]
        print(f"[INFO] Loaded name -> student_id mapping: {name_to_id}")
    except Exception as e:
        print(f"[WARN] Could not build name->id mapping: {e}")

# ---------------------------
# Main Loop
# ---------------------------
currently_present = set()
print("[INFO] Starting recognition. Press 'q' to quit.")

while True:
    ret, img = cam.read()
    if not ret: break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
    current_frame_detections = set()

    for (x, y, w, h) in faces:
        try:
            id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        except Exception as e:
            continue

        if confidence < 100:
            if id_pred < len(names):
                id_name = names[id_pred] # Get name from our dynamic list
            else:
                id_name = "Unknown" # ID is out of range
            confidence_text = f"{round(100 - confidence)}%"
            color = (0, 255, 0)
        else:
            id_name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"
            color = (0, 0, 255)

        # Get Student ID (from face_db)
        student_id = name_to_id.get(id_name, str(id_pred)) # Fallback to numeric ID

        # Draw box and text
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, str(id_name), (x + 5, y - 5), font, 1, color, 2)
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        if id_name != "Unknown" and id_name != "None":
            current_frame_detections.add(student_id)
            if student_id not in currently_present:
                # Log ENTRY
                log_name = id_name
                log_id = student_id
                
                if attendance_system:
                    attendance_system.log_attendance(log_id, log_name, "ENTRY")
                else:
                    simple_log_attendance(log_id, log_name, "ENTRY")
                currently_present.add(student_id)

    # Detect Exits
    exited = currently_present - current_frame_detections
    for sid in exited:
        sname = "Unknown"
        # Find name for this student_id
        for name, student_id_val in name_to_id.items():
            if student_id_val == sid:
                sname = name
                break
        
        if attendance_system:
            attendance_system.log_attendance(sid, sname, "EXIT")
        else:
            simple_log_attendance(sid, sname, "EXIT")
        currently_present.discard(sid)

    cv2.putText(img, f'Present: {len(currently_present)}', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27 or k == ord('q'):
        break

# --- Cleanup ---
for sid in list(currently_present):
    sname = "Unknown"
    for name, student_id_val in name_to_id.items():
        if student_id_val == sid:
            sname = name
            break
    if attendance_system:
        attendance_system.log_attendance(sid, sname, "EXIT (FORCED)")
    else:
        simple_log_attendance(sid, sname, "EXIT (FORLED)")

print("[INFO] Recognition stopped.")
cam.release()
cv2.destroyAllWindows()
