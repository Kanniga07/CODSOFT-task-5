import cv2

class FaceDetector:
    def __init__(self, cascade_path):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.video_cap = cv2.VideoCapture(0)

    def check_webcam(self):
        if not self.video_cap.isOpened():
            print("Error: Unable to access the webcam.")
            return False
        return True

    def detect_faces(self):
        while True:
            ret, video_data = self.video_cap.read()
            if not ret:
                print("Error: unable to get a frame.")
                break

            col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(col, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("video_live", video_data)

            if cv2.waitKey(10) == ord('a'):
                break

    def release_resources(self):
        self.video_cap.release()
        cv2.destroyAllWindows()

cascade_path = "C:/Users/kanni/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_detector = FaceDetector(cascade_path)

if face_detector.check_webcam():
    face_detector.detect_faces()
    face_detector.release_resources()
