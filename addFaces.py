print("Importing libraries...")
import cv2
import pickle
import os
from pathlib import Path
import face_recognition
import csv
facedetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces_data=[]
i=0

name=input("Enter Your Name: ")
email=input("Enter your email: ")
directory = f"training/{name}"
if not os.path.exists(directory):
    os.makedirs(directory)

print("Starting capture...")

video=cv2.VideoCapture(0)

while True:
    ret,frame=video.read()
    cv2.imshow("Frame",frame)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        if len(faces_data)<=10 and i%10==0:
            faces_data.append(crop_img)
            cv2.imwrite(f"{directory}/{len(faces_data)}.jpg",crop_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==10:
        break
video.release()
cv2.destroyAllWindows()

print("Encoding the face data...")

DEFAULT_ENCODINGS_PATH = Path("data/faces_data.pkl")

def encode_known_faces(model= "hog", encodings_location= DEFAULT_ENCODINGS_PATH):
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

encode_known_faces()

print("Writing data to data.csv")

if os.path.isfile("Attendance/data.csv"):
    with open("Attendance/data.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        found = False
        for row in rows:
            if row[0] == name:
                row[1]=email
                found = True
                break
        if not found:
            rows.append([name,email])
    csvfile.close()
    with open("Attendance/data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    csvfile.close()
else:
    with open("Attendance/data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['NAME','EMAIL'])
        writer.writerow([name,email])
    csvfile.close()

