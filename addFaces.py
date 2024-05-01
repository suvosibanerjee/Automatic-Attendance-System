import cv2
import pickle
import os
from pathlib import Path
import face_recognition

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data=[]

i=0

name=input("Enter Your Name: ")
directory = f"training/{name}"
if not os.path.exists(directory):
    os.makedirs(directory)

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        #resized_img=cv2.resize(crop_img, (50,50))
        if len(faces_data)<=10 and i%10==0:
            faces_data.append(crop_img)
            cv2.imwrite(f"{directory}/{len(faces_data)}.jpg",crop_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==10:
        break
video.release()
cv2.destroyAllWindows()\

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

