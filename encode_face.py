import cv2
import os
import pickle
import numpy as np
from datetime import datetime

print("=== Enhanced Face Encoding System ===")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Enhanced data structure
known_encodings = []
known_names = []
known_ids = []
registration_dates = []

dataset_path = 'dataset'

if not os.path.exists(dataset_path):
    print("No dataset folder found! Run enroll_user.py first.")
    exit()

print("Processing student faces...")

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    
    if os.path.isdir(folder_path):
        # Extract name and ID from folder name (format: Name_ID)
        if '_' in folder_name:
            name, student_id = folder_name.split('_', 1)
        else:
            name = folder_name
            student_id = "Unknown"
        
        print(f"Processing {name} (ID: {student_id})...")
        
        for image_name in os.listdir(folder_path):
            if image_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(folder_path, image_name)
                
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    # Temporary encoding (will be replaced with proper face encoding)
                    face_encoding = [x, y, w, h]
                    
                    known_encodings.append(face_encoding)
                    known_names.append(name)
                    known_ids.append(student_id)
                    registration_dates.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Save enhanced database
encoding_data = {
    'encodings': known_encodings,
    'names': known_names,
    'ids': known_ids,
    'registration_dates': registration_dates
}

with open('face_database.pkl', 'wb') as f:
    pickle.dump(encoding_data, f)

print(f"Encoding completed! Saved {len(known_names)} face encodings.")
