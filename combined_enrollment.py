import cv2
import os
import time
import json

DATASET_PATH = 'dataset'
MAP_FILE = 'id_to_name_map.json'

# --- Create dataset folder if it doesn't exist ---
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# --- Load existing ID-to-Name map or create new one ---
if os.path.exists(MAP_FILE):
    with open(MAP_FILE, 'r') as f:
        id_to_name_map = json.load(f)
else:
    id_to_name_map = {} # Use strings for JSON keys

# Ensure '0' is always 'None' or 'Unknown' for the recognizer
if '0' not in id_to_name_map:
    id_to_name_map['0'] = "None" 

print("=== Combined Enrollment System ===")
print(f"Current mappings: {id_to_name_map}")

# --- Get both inputs from the user ---
name_and_id = input("Enter the Name_ID (e.g., Jit_101): ").strip()
numeric_id = input(f"Enter the Numeric ID for {name_and_id.split('_')[0]} (e.g., 1, 2, 3...): ").strip()
name = name_and_id.split('_')[0]

if numeric_id in id_to_name_map and id_to_name_map[numeric_id] != name:
    print(f"[WARNING] Numeric ID {numeric_id} is already assigned to {id_to_name_map[numeric_id]}.")
    print("Please use a different numeric ID or clear the map file.")
    exit()

# --- Add to map and save ---
id_to_name_map[numeric_id] = name
with open(MAP_FILE, 'w') as f:
    json.dump(id_to_name_map, f, indent=4)
print(f"Updated map: {id_to_name_map}")


# --- Create folder for encode_face.py ---
person_folder = os.path.join(DATASET_PATH, name_and_id)
if not os.path.exists(person_folder):
    os.makedirs(person_folder)
print(f"Saving images for '{name_and_id}' (Numeric ID: {numeric_id})")

# --- Initialize camera ---
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Taking 30 photos. Look at the camera and move slightly...")
count = 0
max_images = 30 

while count < max_images:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    display_frame = frame.copy()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(display_frame, f'Photos: {count+1}/{max_images}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Enrollment', display_frame)
    
    # Auto-capture
    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face_roi_gray = gray[y:y+h, x:x+w]
        
        count += 1
        
        # Save format 1 (for encode_face.py)
        cv2.imwrite(f"{person_folder}/img_{count}.jpg", frame)
        
        # Save format 2 (for train_model.py)
        cv2.imwrite(f"{DATASET_PATH}/User.{numeric_id}.{count}.jpg", face_roi_gray)

        print(f"Saved image {count}/{max_images}")
        time.sleep(0.5) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Enrollment completed! Saved {count} images.")
cap.release()
cv2.destroyAllWindows()