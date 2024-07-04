1.	Image Acquisition: The system captures an image or video stream from the classroom using a camera.

2.	 Face Detection: The image is analysed to detect the faces present within it. This might involve algorithms like Haar Cascades or deep learning models.

3.	  Face Cropping: Detected faces are cropped from the image, isolating them for further processing.

4.	 Create Dataset: This step involves creating a dataset of labelled images with student faces for training the facial recognition model.

5.	 Training the Dataset and Saving it in XML File: The facial recognition model is trained using the dataset to learn to identify specific students. The trained model is then saved in an XML file for future use.

6.	 Testing (or Start Capture): This step signifies the system is ready for live operation. It might skip directly to "Capture Face to be Detected" if training is complete.

7.	 Capture Face to be Detected: The system captures a new image or extracts a face from the video stream to identify the person.

8.	 Output Confidence: The trained model analyses the captured face and outputs a confidence score indicating its certainty about the identified person.

9.	 Greater than Threshold? (Yes/No): This step checks if the confidence score from the model is higher than a predefined threshold. This threshold determines the minimum acceptable certainty for recognizing someone.

10.	Yes: Mark Present, Store in Excel, Cloud, and Send Email: If the confidence score is high enough:
a.	The person is marked as present in the attendance record.
b.	The attendance data is stored in an Excel file on the local system.
c.	The data is also uploaded to a cloud storage for backup and accessibility.
d.	An email notification is sent to the student confirming their attendance.

11.	No: Continue Capturing Face: If the confidence score is low, the system understands it cannot identify the person with certainty. It continues capturing faces from the video stream or image sequence, hoping to get a clearer view for better identification.

12.	People Absent: Send Email, Store in Excel and Cloud: For students who haven't been identified throughout the session:
a.	They are marked as absent in the attendance record.
b.	An email notification is sent to them informing them about their absence.
c.	The absence data is also stored in the Excel file and uploaded to the cloud.
