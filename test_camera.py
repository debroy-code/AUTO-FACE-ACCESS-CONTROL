import cv2

# Test camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
else:
    print("Camera working! Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
            
        cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Test completed!")
