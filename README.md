# ai-attendance-system


🚀 Overview

The Smart Attendance System is an AI-powered application that automates attendance tracking using facial recognition. It leverages computer vision techniques to detect and recognize individuals in real-time and logs attendance efficiently without manual intervention.

This system eliminates traditional attendance methods by providing a contactless, accurate, and scalable solution.


🛠 Tech Stack

Language: Python

Libraries:
OpenCV,face_recognition,MediaPipe,NumPy, Pandas


⚙️ System Architecture

🔹 1. Dataset Creation
Captures user images via webcam
Stores images in structured folders (/photos/<person_name>)

🔹 2. Face Detection & Recognition
Uses MediaPipe for face detection
Uses face_recognition for encoding & matching
Compares faces using Euclidean distance

🔹 3. Attendance Logging
Recognized users are logged into a CSV file
Stores:
Name
Timestamp

🔹 4. Report Generation
Aggregates multiple attendance records
Generates consolidated reports using Pandas

📂 Project Structure

smart-attendance-system-cv/

│── attendance_system.py          # Main recognition system

│── generate_attendance_report.py # Report generation

│── photos/                       # Stored face images

│── attendance/                   # CSV attendance logs

│── README.md
