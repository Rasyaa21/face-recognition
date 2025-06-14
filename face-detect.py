import mediapipe as mp
import cv2 
from PIL import Image
import json
import os
import mysql.connector
import numpy as np
from dotenv import load_dotenv
import face_recognition

# Load .env
load_dotenv()

# MySQL Connection
db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)
cursor = db.cursor()

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

def load_users():
    cursor.execute("SELECT name, embedding FROM users")
    data = cursor.fetchall()

    names = []
    encodings = []

    for name, embedding_json in data:
        names.append(name)
        embedding = np.array(json.loads(embedding_json))
        encodings.append(embedding)

    return names, encodings

while cap.isOpened():
    names, encoding = load_users()
    success, frame = cap.read()
    if not success: 
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    face_count = 0
    
    if results.detections: 
        face_count = len(results.detections)
        for detection in results.detections:
            boxFace = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x = int(boxFace.xmin * w)
            y = int(boxFace.ymin * h)
            w_box = int(boxFace.width * w)
            h_box = int(boxFace.height * h)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            print(detection)
            
            face_img = frame_rgb[y:y+h_box, x:x+w_box]
            face_resized = cv2.resize(face_img, (150,150))
            face_encoding = face_recognition.face_encodings(face_resized)
            
            name_detected = "unknown"
            
            if face_encoding:
                face_encoding = face_encoding[0]
                
                distances = face_recognition.face_distance(encoding, face_encoding)
                min_distance = np.min(distances)
                idx = np.argmin(distances)
                
                if min_distance < 0.5 :
                    name_detected = names[idx]
                else: 
                    name_detected = "Unknown"

            cv2.putText(frame, f"Nama : {name_detected}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
    cv2.putText(frame, f"faces detected: {face_count}", (30, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()