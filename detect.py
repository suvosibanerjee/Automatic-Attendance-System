print("Importing libraries ...")
from pathlib import Path
import cv2
import face_recognition
import time
from datetime import datetime
import csv
import os
import smtplib
from email.message import EmailMessage
from collections import Counter
import pickle
from dotenv import load_dotenv
facedetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

load_dotenv()

DEFAULT_ENCODINGS_PATH = Path("data/faces_data.pkl")
Path("output").mkdir(exist_ok=True)


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
    return name   

print("Defining functions ...")
 
video=cv2.VideoCapture(0)

COL_NAMES = ['NAME', 'EMAIL','FIRST','LAST']

print("Starting capture ...")

while True:
    ret,frame=video.read()
    cv2.imshow("Frame",frame)
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
        attendance = [str(output), str(timestamp),""]
        
        if os.path.isfile("Attendance/Attendance.csv"):
            with open("Attendance/Attendance.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                found = False
                for row in rows:
                    if row[0] == output:
                        if row[1]=="":
                            row[1] = timestamp
                        else:
                            row[2]=timestamp
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
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

server = smtplib.SMTP_SSL("smtp.fastmail.com",465)
server.login(os.getenv("EMAIL"),os.getenv("PASSWORD"))

def sendEmail(emails,success=False):
    
    if success:
        
        for email in emails:
            message = EmailMessage()
            message["from"]="attendance@fastmail.com"
            message.set_content("Your attendance has been marked successfully")
            message["subject"] = "Attedance Marked"
            message["to"]=email
            server.send_message(message)
            print(f"Sent email to {email}")

    else:
        for email in emails:
            message = EmailMessage()
            message["from"]="attendance@fastmail.com"
            message.set_content("Your attendance has not been marked. Approach your faculty")
            message["subject"] = "Attedance not marked"
            message["to"]=email
            server.send_message(message)
            print(f"Sent email to {email}") 

def convert_to_datetime(timestamp):
    return datetime.strptime(timestamp, "%H:%M-%S")

def time_diff_in_minutes(start_time, end_time):
    diff = end_time - start_time
    return diff.total_seconds() / 60

def retrieve_rows_with_time_difference_above_threshold(csv_file, threshold_minutes):
    rows_to_retrieve = []
    rows_not_to_retrieve = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row[0] =="":
                rows_not_to_retrieve.append(row)
                continue


            timestamp1 = convert_to_datetime(row[1])
            if row[1]=="":
                ts = time.time()
                timestamp2 = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            else:

                timestamp2 = convert_to_datetime(row[2])
            time_difference = time_diff_in_minutes(timestamp1, timestamp2)
            if time_difference > threshold_minutes:
                rows_to_retrieve.append(row)
            else:
                rows_not_to_retrieve.append(row)
    return [rows_to_retrieve,rows_not_to_retrieve]

def retrieve_second_column_value(data_csv, target_column_value):
    with open(data_csv, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == target_column_value:
                return row[1]
    return None

threshold_minutes = 1

rows=retrieve_rows_with_time_difference_above_threshold("Attendance/Attendance.csv", threshold_minutes)
rows_to_process = rows[0]
rows_not_to_process =rows[1]

success=[]
fail=[]
for row in rows_to_process:
    first_column_value = row[0]
    second_column_value = retrieve_second_column_value("Attendance/data.csv", first_column_value)
    if second_column_value is not None:
        success.append(second_column_value)
    else:
        print("No match found for", first_column_value)

for row in rows_not_to_process:
    first_column_value = row[0]
    second_column_value = retrieve_second_column_value("Attendance/data.csv", first_column_value)
    if second_column_value is not None:
        fail.append(second_column_value)
    else:
        print("No match found for", first_column_value)

print("Sending emails")
sendEmail(success,True)
sendEmail(fail,False)
server.quit()



