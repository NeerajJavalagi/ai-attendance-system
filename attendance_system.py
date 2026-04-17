import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import os
import csv
from datetime import datetime

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Path to photos folder
photos_path = '../photos'

def capture_new_person_photos():
    # Ask for the person's name
    person_name = input("Enter the name of the new person : ").strip()
    if not person_name:
        print("Invalid name. Please try again.")
        return

    # Create a folder for the new person
    person_folder = os.path.join(photos_path, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    # Capture 5 photos using webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Capturing 5 photos. Please look at the camera.")
    photo_count = 0

    while photo_count < 5:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            continue

        # Display the frame
        cv2.imshow("Capturing Photos - Press 'c' to capture", frame)

        # Wait for 'c' key to be pressed to capture photo
        if cv2.waitKey(1) & 0xFF == ord('c'):
            photo_filename = os.path.join(person_folder, f"{person_name}_{photo_count + 1}.jpg")
            cv2.imwrite(photo_filename, frame)
            print(f"Photo {photo_count + 1} captured as {photo_filename}")
            photo_count += 1

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture canceled.")
            break

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Main Functionality
def main():
    # Ask the user if they want to add a new person or start attendance recognition
    while True:
        choice = input("Enter 'a' to add a new person, 'r' to start recognition, or 'q' to quit: ").strip().lower()
        if choice == 'a':
            capture_new_person_photos()
        elif choice == 'r':
            break
        elif choice == 'q':
            return
        else:
            print("Invalid choice. Please enter 'a', 'r', or 'q'.")

    # Load known faces and names from the photos folder
    known_face_encodings = []
    known_face_names = []

    # Load and encode images from each subfolder in the photos folder
    for person_name in os.listdir(photos_path):
        person_folder = os.path.join(photos_path, person_name)

        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(person_folder, filename)
                    try:
                        # Load the image
                        img = face_recognition.load_image_file(img_path)
                        encodings = face_recognition.face_encodings(img)
                        if len(encodings) > 0:
                            known_face_encodings.append(encodings[0])
                            # Use the folder name as the person's name
                            known_face_names.append(person_name)
                        else:
                            print(f"No face found in {filename}, skipping.")
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")

    # Initialize attendance CSV
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    attendance_file = f'../attendance/{current_date}.csv'

    with open(attendance_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Name', 'Time'])

    # Start video capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Press 'q' to quit the application.")

    # Initialize attendance tracking
    attendance_logged = set()

    # Use MediaPipe for face detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame, retrying...")
                continue

            # Convert the frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces using MediaPipe
            results = face_detection.process(rgb_frame)

            # If faces are detected, proceed with recognition
            if results.detections:
                face_locations = []
                for detection in results.detections:
                    # Extract bounding box information
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                    int(bboxC.width * iw), int(bboxC.height * ih))
                    face_locations.append((y, x + w, y + h, x))

                # Convert face locations to the format used by face_recognition
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # Use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

                    # Log attendance if a known face is recognized and hasn't been logged yet
                    if name != "Unknown" and name not in attendance_logged:
                        attendance_logged.add(name)
                        current_time = datetime.now().strftime("%H:%M:%S")
                        with open(attendance_file, 'a', newline='') as f:
                            csv_writer = csv.writer(f)
                            csv_writer.writerow([name, current_time])
                        print(f"{name} marked present at {current_time}")

                # Draw bounding boxes and labels
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            # Display the resulting frame
            cv2.imshow('Smart Attendance System', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()