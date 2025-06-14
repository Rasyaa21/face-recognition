import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import mysql.connector
import json
import numpy as np 
import os
from dotenv import load_dotenv

load_dotenv()

db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)
cursor = db.cursor()

def save_user(name, embedding):
    embedding_json = json.dumps(embedding.tolist())
    sql = "INSERT INTO users (name, embedding) VALUES (%s, %s)"
    cursor.execute(sql, (name, embedding_json))
    db.commit()

def capture_and_register(name):
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(frame_rgb)
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
            
            if face_encodings:
                embedding = face_encodings[0]
                save_user(name, embedding)
                messagebox.showinfo("Success", f"User {name} registered successfully.")
                cap.release()
                cv2.destroyAllWindows()
                return
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Register Face - Press Q to quit", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showwarning("Warning", "No face detected for registration.")

# TKINTER FORM
def register():
    name = entry_name.get()
    if not name:
        messagebox.showerror("Error", "Name is required")
        return
    capture_and_register(name)

# Build Form
root = tk.Tk()
root.title("Face Registration")

tk.Label(root, text="Enter Name:").pack(pady=10)
entry_name = tk.Entry(root, width=30)
entry_name.pack(pady=5)

tk.Button(root, text="Register Face", command=register, bg="blue", fg="white", width=20).pack(pady=10)

root.mainloop()