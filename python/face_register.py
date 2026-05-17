import cv2
import os

os.makedirs("faces", exist_ok=True)

user_id = input("Enter User ID: ")
name = input("Enter Name: ")

cap = cv2.VideoCapture(0)

print("Press 's' to save face")

while True:
    ret, frame = cap.read()
    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        filename = f"faces/{user_id}_{name}.jpg"
        cv2.imwrite(filename, frame)
        print("Face registered:", filename)
        break

cap.release()
cv2.destroyAllWindows()
