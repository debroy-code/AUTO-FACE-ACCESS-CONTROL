# reco_face.py
import cv2
import os
import time
import pickle
from datetime import datetime

# ---------------------------
# Helper: simple fallback logger (if attendance_logger.py / face_database.pkl not available)
# ---------------------------
def simple_log_attendance(student_id, student_name, action, attendance_file='attendance_log.csv'):
    """Append an attendance line to CSV (Date,Time,Student_ID,Student_Name,Action)."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    entry = f"{date_str},{time_str},{student_id},{student_name},{action}\n"
    # Create file with header if not exists
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write("Date,Time,Student_ID,Student_Name,Action\n")
    with open(attendance_file, 'a') as f:
        f.write(entry)
    print(f"📝 {action}: {student_name} ({student_id}) at {time_str}")

# ---------------------------
# Try to use AttendanceSystem from your attendance_logger.py (preferred)
# ---------------------------
attendance_system = None
face_db = None
try:
    from attendance_logger import AttendanceSystem
    # Instantiate but do not call run() — we will use its logging function and face database if present.
    attendance_system = AttendanceSystem()
    # attendance_system has loaded face_database.pkl into attendance_system.face_data
    face_db = attendance_system.face_data
    print("[INFO] Using attendance_logger.AttendanceSystem for logging.")
except SystemExit:
    # attendance_logger called exit() because no face_database.pkl — fallback
    print("[WARN] attendance_logger found no face_database.pkl. Falling back to simple logging.")
    attendance_system = None
except Exception as e:
    # Generic fallback (module maybe not present)
    print(f"[WARN] Could not import AttendanceSystem ({e}). Falling back to simple logging.")
    attendance_system = None

# ---------------------------
# Load LBPH recognizer (your trained model)
# ---------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
trainer_path = 'trainer/trainer.yml'
if os.path.exists(trainer_path):
    recognizer.read(trainer_path)
    print(f"[INFO] Loaded trainer model from {trainer_path}")
else:
    print(f"[ERROR] Trainer file '{trainer_path}' not found. Please run train_model.py first.")
    exit(1)

# face cascade
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# camera init
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# If you already have a names list, keep it here.
# You can update this list when you add users (index must match training IDs).
names = ['None','JIT', 'ABHISHEK', 'SNEHA', 'VASUNDHARA']

# If face database present, build a mapping name -> student_id (preferred)
name_to_id = {}
if face_db:
    try:
        for i, nm in enumerate(face_db['names']):
            # face_db['ids'][i] is student's id stored during encoding
            name_to_id[nm] = face_db['ids'][i]
        print("[INFO] Loaded name -> student_id mapping from face_database.pkl")
    except Exception as e:
        print(f"[WARN] Could not build name->id mapping: {e}")

# Session tracking: who is currently present (student_id set)
currently_present = set()

# To detect exits robustly we use detections per-frame
print("[INFO] Starting recognition. Press 'q' to quit.")
while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to capture image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    # Detected in this frame
    current_frame_detections = set()

    for (x, y, w, h) in faces:
        # predict using LBPH
        try:
            id_pred, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        except Exception as e:
            # prediction error fallback
            print(f"[WARN] recognizer.predict failed: {e}")
            continue

        # determine name
        if confidence < 100:
            if id_pred < len(names):
                id_name = names[id_pred]
            else:
                id_name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"
        else:
            id_name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"

        # determine student_id (prefer face_database mapping if available)
        student_id = None
        if id_name != "Unknown" and id_name in name_to_id:
            student_id = name_to_id[id_name]
        else:
            # fallback: use recognizer numeric id as id string
            student_id = str(id_pred)

        # set color: green = known, red = unknown
        if id_name == "Unknown":
            color = (0, 0, 255)   # red (B,G,R)
        else:
            color = (0, 255, 0)   # green

        # draw rectangle and name
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, str(id_name), (x + 5, y - 5), font, 1, color, 2)
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        # If known, add to current detections and log if newly present
        if id_name != "Unknown":
            current_frame_detections.add(student_id)
            if student_id not in currently_present:
                # Log ENTRY (use attendance_system if available)
                if attendance_system:
                    # attendance_system.log_attendance(student_id, id_name, "ENTRY")
                    # attendance_system.log_attendance method exists in attendance_logger.py
                    attendance_system.log_attendance(student_id, id_name, "ENTRY")
                else:
                    simple_log_attendance(student_id, id_name, "ENTRY")
                currently_present.add(student_id)
        else:
            # Unknown: we don't add to currently_present
            pass

    # Detect exits: who was present previously but not in this frame
    exited = set(currently_present) - set(current_frame_detections)
    for sid in exited:
        # find student name if possible
        sname = None
        if face_db:
            # look up name for this sid
            try:
                idx = face_db['ids'].index(sid)
                sname = face_db['names'][idx]
            except Exception:
                # sid in currently_present might be stringified numeric id; try match differently
                for i, _sid in enumerate(face_db['ids']):
                    if str(_sid) == str(sid):
                        sname = face_db['names'][i]
                        break
        if not sname:
            # fallback: try to get from names using numeric id
            try:
                sname = names[int(sid)]
            except Exception:
                sname = "Unknown"

        # log EXIT
        if attendance_system:
            attendance_system.log_attendance(sid, sname, "EXIT")
        else:
            simple_log_attendance(sid, sname, "EXIT")
        currently_present.discard(sid)

    # Update currently_present to only those seen in this frame (we already handled exits)
    # Note: members that were just added remain present
    # Display present count on frame
    cv2.putText(img, f'Present: {len(currently_present)}', (10, 30), font, 1, (255, 255, 255), 2)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27 or k == ord('q'):
        break

# When quitting, mark forced exit for remaining present people
for sid in list(currently_present):
    sname = None
    if face_db:
        try:
            idx = face_db['ids'].index(sid)
            sname = face_db['names'][idx]
        except Exception:
            for i, _sid in enumerate(face_db['ids']):
                if str(_sid) == str(sid):
                    sname = face_db['names'][i]
                    break
    if not sname:
        try:
            sname = names[int(sid)]
        except Exception:
            sname = "Unknown"

    if attendance_system:
        attendance_system.log_attendance(sid, sname, "EXIT (FORCED)")
    else:
        simple_log_attendance(sid, sname, "EXIT (FORCED)")

cam.release()
cv2.destroyAllWindows()
print("[INFO] Recognition stopped. Goodbye.")
