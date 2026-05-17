import cv2
import os

# Create directories if they don't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_face():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 's' to save the image or 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Face Detection - Press s to save, q to quit', frame)
        
        # Break the loop on 'q' press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(faces) > 0:
            # Save the captured face
            face_img = frame[y:y+h, x:x+w]
            img_name = f"dataset/face_{len(os.listdir('dataset')) + 1}.jpg"
            cv2.imwrite(img_name, face_img)
            print(f"Saved {img_name}")
    
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_face()
