
import cv2
import os
import numpy as np
from PIL import Image
import json

# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# File to store name mappings
names_file = 'names.json'

def load_names():
    if os.path.exists(names_file):
        with open(names_file, 'r') as f:
            return json.load(f)
    return {}

def save_names(names):
    with open(names_file, 'w') as f:
        json.dump(names, f)

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        # Get the ID from the image name
        try:
            id = int(os.path.split(image_path)[-1].split("_")[1].split(".")[0])
        except:
            print(f"Skipping invalid filename: {image_path}")
            continue
            
        # Convert to grayscale
        pil_image = Image.open(image_path).convert('L')
        img_numpy = np.array(pil_image, 'uint8')
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    
    return face_samples, ids

def register_face():
    name = input("Enter the name for this person: ").strip()
    if not name:
        print("Name cannot be empty!")
        return
    
    # Get the next available ID
    names = load_names()
    if not names:  # If no names exist yet
        next_id = 1
    else:
        next_id = max(int(k) for k in names.keys()) + 1
    
    # Start capturing
    cap = cv2.VideoCapture(0)
    print(f"Capturing face for {name} (ID: {next_id})")
    print("Press 's' to save or 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Register Face - s:save, q:quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(faces) > 0:
            # Save the face image
            face_img = gray[y:y+h, x:x+w]
            img_name = f"dataset/face_{next_id}.jpg"
            cv2.imwrite(img_name, face_img)
            
            # Update names dictionary
            names[str(next_id)] = name
            save_names(names)
            print(f"Saved {name} (ID: {next_id})")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def train_model():
    print("Training the model...")
    faces, ids = get_images_and_labels(path)
    
    if len(faces) == 0:
        print("No faces found in the dataset. Please register some faces first.")
        return False
    
    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')
    print("Model trained and saved as 'trainer.yml'")
    return True

def recognize_faces():
    if not os.path.exists('trainer.yml'):
        print("Please train the model first by registering some faces.")
        return

    recognizer.read('trainer.yml')
    names = load_names()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
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
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            # If confidence is less than 100, it's considered a match
            if confidence < 100:
                name = names.get(str(id), f"Unknown (ID: {id})")
                confidence_text = f"  {round(100 - confidence)}%"
                color = (0, 255, 0)  # Green for recognized face
            else:
                name = "Unknown"
                confidence_text = "  {0}%".format(round(100 - confidence))
                color = (0, 0, 255)  # Red for unknown face
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, confidence_text, (x+5, y+h-5), font, 1, (255, 255, 0), 1)
        
        cv2.imshow('Face Recognition - Press q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create dataset directory if it doesn't exist
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    while True:
        print("\nFace Recognition System")
        print("1. Register a new face")
        print("2. Train the model")
        print("3. Start face recognition")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            register_face()
        elif choice == '2':
            if train_model():
                input("Model trained successfully! Press Enter to continue...")
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
