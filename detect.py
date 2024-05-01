from pathlib import Path
import cv2
import face_recognition
import time
from datetime import datetime
import csv
import os

DEFAULT_ENCODINGS_PATH = Path("data/faces_data.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

from collections import Counter
from PIL import Image, ImageDraw
import pickle


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )

def recognize_faces(image_location,model= "hog",encodings_location= DEFAULT_ENCODINGS_PATH):
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )
    name="Unknown"
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        print(name, bounding_box)
        #_display_face(draw, bounding_box, name)

    #del draw
    #pillow_image.show()  
    return name   
 
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

COL_NAMES = ['NAME', 'TIME']

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        cv2.imwrite(f"test.jpg",crop_img)
        output = recognize_faces("test.jpg")
        if output=="Unknown":
            print("No face detected")
            continue
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        attendance = [str(output), str(timestamp)]
        
        if os.path.isfile("Attendance/Attendance.csv"):
            with open("Attendance/Attendance.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                found = False
                for row in rows:
                    if row[0] == output:
                        row[1] = timestamp
                        found = True
                        break
                if not found:
                    rows.append(attendance)
            csvfile.close()
            with open("Attendance/Attendance.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(rows)
            csvfile.close()
        else:
            with open("Attendance/Attendance.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()