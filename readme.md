# 🧠 Face Recognition 

Real-time face detection and recognition system using:

- MediaPipe (for fast face detection)
- face_recognition (for embedding and recognition)
- MySQL database (for user & embedding storage)
- OpenCV (for video streaming & real-time capture)
- Tkinter (for user registration GUI)
- Python 3.11.8 (tested version)

---

## 🚀 Features

- Real-time face detection from webcam.
- Face registration form with name input via Tkinter.
- Save face embeddings into MySQL database.
- Face recognition by comparing real-time capture with database.
- Clean separation between registration & recognition pipeline.
- Can be extended into full SaaS Face Recognition system.

---

## 🛠️ Tech Stack

| Library | Function |
| -------- | -------- |
| `mediapipe` | Real-time face detection |
| `face_recognition` | Generate embeddings (128D) |
| `OpenCV` | Webcam capture & drawing bounding box |
| `mysql-connector-python` | MySQL database connection |
| `dotenv` | Manage database credentials securely |
| `tkinter` | GUI Registration interface |
| `Python 3.11.8` | Core environment |

---

## 📦 Installation

### 1️⃣ Clone this repo

```bash
git clone <your-repo-url>
cd face-detection
