import cv2
import os

# Create and load the trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Define a list of names corresponding to the user IDs
# IMPORTANT: The order matters. ID 1 corresponds to the first name, ID 2 to the second, etc.
# 'None' is a placeholder for index 0.
names = ['None','sharukh khan', 'JIT', 'DEEP', 'VASUNDHARA'] # Add more names as you add more users

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

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # The predict method returns the ID and a confidence value (lower is better)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is less than 100 ==> "0" is perfect match
        if confidence < 100:
            # Check if the ID is within the range of our names list
            if id < len(names):
                id_name = names[id]
            else:
                id_name = "Unknown" # Handle case where ID is not in the list
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            id_name = "unknown"
            confidence_text = f"  {round(100 - confidence)}%"

        # Display the name and confidence level on the screen
        cv2.putText(
            img,
            str(id_name),
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(confidence_text),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' or 'q' for exiting video
    if k == 27 or k == ord('q'):
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleaning up.")
cam.release()
cv2.destroyAllWindows()
