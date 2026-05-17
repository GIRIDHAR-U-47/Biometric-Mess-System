import cv2
import os
import numpy as np
from PIL import Image
import pickle

# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        # Convert to grayscale
        pil_image = Image.open(image_path).convert('L')
        img_numpy = np.array(pil_image, 'uint8')
        
        # Get the ID from the image name (assuming format: face_1.jpg, face_2.jpg, etc.)
        id = int(os.path.split(image_path)[-1].split("_")[1].split(".")[0])
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    
    return face_samples, ids

def train_model():
    print("Training the model...")
    faces, ids = get_images_and_labels(path)
    
    if len(faces) == 0:
        print("No faces found in the dataset. Please capture some faces first.")
        return False
    
    recognizer.train(faces, np.array(ids))
    
    # Save the model
    recognizer.save('trainer.yml')
    print("Model trained and saved as 'trainer.yml'")
    return True

def recognize_faces():
    if not os.path.exists('trainer.yml'):
        print("Please train the model first by capturing some faces.")
        return

    recognizer.read('trainer.yml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Recognize the face
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            # Draw rectangle around the face
            color = (0, 255, 0)  # Green for recognized face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display the ID
            cv2.putText(frame, f"ID: {id}", (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x+5, y+h-5), font, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Face Recognition - Press q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # First, train the model with captured faces
    if train_model():
        # Then, start recognition
        recognize_faces()
