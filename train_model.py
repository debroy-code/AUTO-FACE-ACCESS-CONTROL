import cv2
import numpy as np
from PIL import Image
import os
import sys # Import the sys module to exit the script

# Path for face image database
path = 'dataset'

# Create the LBPH (Local Binary Patterns Histograms) face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to get the images and label data
def getImagesAndLabels(path):
    # Get all file paths in the dataset folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        try:
            # Open the image and convert it to grayscale
            PIL_img = Image.open(imagePath).convert('L') # 'L' converts to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            # Get the ID from the image filename (e.g., User.1.5.jpg -> ID is 1)
            id = int(os.path.split(imagePath)[-1].split(".")[1])

            # Detect the face in the image
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
        except Exception as e:
            print(f"Skipping file with error: {imagePath} - {e}")


    return faceSamples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)

# ---- NEW: Check if we have found any faces before training ----
if len(faces) == 0:
    print("\n [ERROR] No faces found in the dataset. Please create a dataset first.")
    print("         Make sure the images are clear and well-lit.")
    sys.exit() # Exit the program if no faces were found
# ----------------------------------------------------------------

# Train the recognizer with the faces and their corresponding IDs
recognizer.train(faces, np.array(ids))

# Create a 'trainer' directory if it doesn't exist
if not os.path.exists('trainer'):
    os.makedirs('trainer')

# Save the trained model into the trainer/trainer.yml file
recognizer.write('trainer/trainer.yml')

# Print the number of faces trained and end the program
print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")

