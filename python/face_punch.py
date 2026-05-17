import cv2
import face_recognition
import os
import subprocess
import time

# Load known faces
known_encodings = []
known_ids = []

for file in os.listdir("faces"):
    img = face_recognition.load_image_file(f"faces/{file}")
    enc = face_recognition.face_encodings(img)
    if enc:
        known_encodings.append(enc[0])
        known_ids.append(file.split("_")[0])

cap = cv2.VideoCapture(0)

last_id = None
last_message = ""
popup_time = 0

while True:
    ret, frame = cap.read()
    rgb = frame[:, :, ::-1]

    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for (top, right, bottom, left), enc in zip(faces, encodings):

        matches = face_recognition.compare_faces(known_encodings, enc)
        name = "Unknown"

        if True in matches:
            idx = matches.index(True)
            user_id = known_ids[idx]

            # Write ID for Java
            with open("recognized_id.txt", "w") as f:
                f.write(user_id)

            # Call Java backend
            result = subprocess.run(
                ["java", "MessPunchSystem"],
                capture_output=True,
                text=True
            )

            message = result.stdout.strip()

            # First detection popup
            if user_id != last_id:
                popup_time = time.time()
                last_message = message
                last_id = user_id

            # Draw face box
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)

    # ONE-TIME BIG POPUP (3 sec)
    if time.time() - popup_time < 3:
        cv2.rectangle(frame, (50, 50), (600, 160), (0, 0, 0), -1)
        cv2.putText(frame, last_message, (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

    # CONTINUOUS SMALL BOTTOM DISPLAY
    if last_message:
        cv2.rectangle(frame, (0, 430), (640, 480), (0, 0, 0), -1)
        cv2.putText(frame, last_message, (10, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    cv2.imshow("Mess Biometric System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
