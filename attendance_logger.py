import cv2
import pickle
import numpy as np
from datetime import datetime
import time
import os

class AttendanceSystem:
    def __init__(self):
        # Load face database
        if os.path.exists('face_database.pkl'):
            with open('face_database.pkl', 'rb') as f:
                self.face_data = pickle.load(f)
        else:
            print("No face database found! Run encode_faces.py first.")
            exit()
        
        # Initialize attendance log
        self.attendance_file = 'attendance_log.csv'
        self.initialize_attendance_log()
        
        # Track currently present students (to avoid duplicate entries)
        self.currently_present = set()
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("Attendance System Initialized!")
        print(f"Registered students: {set(self.face_data['names'])}")
    
    def initialize_attendance_log(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w') as f:
                f.write("Date,Time,Student_ID,Student_Name,Action\n")
    
    def log_attendance(self, student_id, student_name, action):
        """Log entry or exit with timestamp"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        log_entry = f"{date_str},{time_str},{student_id},{student_name},{action}\n"
        
        with open(self.attendance_file, 'a') as f:
            f.write(log_entry)
        
        print(f"📝 LOGGED: {student_name} ({student_id}) - {action} at {time_str}")
    
    def recognize_and_log(self, frame):
        """Detect faces and log attendance"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        current_frame_detections = set()
        
        for (x, y, w, h) in faces:
            # Simple recognition logic (temporary)
            # In Phase 3, we'll replace this with proper face recognition
            for i, encoding in enumerate(self.face_data['encodings']):
                # Simple distance calculation (temporary)
                ex, ey, ew, eh = encoding
                distance = abs(x - ex) + abs(y - ey) + abs(w - ew) + abs(h - eh)
                
                if distance < 100:  # Threshold for "match"
                    student_id = self.face_data['ids'][i]
                    student_name = self.face_data['names'][i]
                    
                    current_frame_detections.add(student_id)
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'{student_name} ({student_id})', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Log entry if not already present
                    if student_id not in self.currently_present:
                        self.log_attendance(student_id, student_name, "ENTRY")
                        self.currently_present.add(student_id)
                    
                    break
        
        # Check for exits (students who were present but not detected now)
        exited_students = self.currently_present - current_frame_detections
        for student_id in exited_students:
            # Find student name
            for i, sid in enumerate(self.face_data['ids']):
                if sid == student_id:
                    student_name = self.face_data['names'][i]
                    self.log_attendance(student_id, student_name, "EXIT")
                    break
        
        self.currently_present = current_frame_detections
        
        return frame
    
    def run(self):
        """Main loop for attendance system"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("Attendance system running...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Process frame for attendance
            processed_frame = self.recognize_and_log(frame)
            
            # Display currently present count
            cv2.putText(processed_frame, f'Present: {len(self.currently_present)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Attendance System - Press Q to quit', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Log exits for all remaining present students
        for student_id in self.currently_present:
            for i, sid in enumerate(self.face_data['ids']):
                if sid == student_id:
                    student_name = self.face_data['names'][i]
                    self.log_attendance(student_id, student_name, "EXIT (FORCED)")
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Attendance system stopped.")

# Run the system
if __name__ == "__main__":
    system = AttendanceSystem()
    system.run()
