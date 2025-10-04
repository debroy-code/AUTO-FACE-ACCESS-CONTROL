import cv2
import os

# Create a directory to save the dataset if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize the camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Use OpenCV's built-in face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prompt the user for a numeric user ID
face_id = input('\n Enter user id and press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
# Initialize face sample count
count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to capture image")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured face to the dataset folder
        # The image is saved in grayscale
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])

        # Display the video frame
        cv2.imshow('image', img)

    # Wait for 30ms or until 'q' is pressed
    k = cv2.waitKey(100) & 0xff
    if k == 27 or k == ord('q'):
        break
    # Exit if 30 face samples have been taken
    elif count >= 30:
        break

# Cleanup
print("\n [INFO] Exiting Program and cleaning up.")
cam.release()
cv2.destroyAllWindows()
