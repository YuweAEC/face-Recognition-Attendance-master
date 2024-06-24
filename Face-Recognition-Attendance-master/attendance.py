import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the directory containing person images
IMAGE_PATH = 'Face-Recognition-Attendance-master/Person_Images'
# Path to the attendance file
ATTENDANCE_FILE = 'Face-Recognition-Attendance-master/Attendance.csv'

# Load images and names
def load_images(image_path):
    images = []
    names = []
    for file in os.listdir(image_path):
        img = cv2.imread(f'{image_path}/{file}')
        images.append(img)
        names.append(os.path.splitext(file)[0])
    return images, names

# Encode faces
def encode_faces(images):
    encodings = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings.append(face_recognition.face_encodings(img_rgb)[0])
    return encodings

# Mark attendance
def mark_attendance(name):
    with open(ATTENDANCE_FILE, 'r+') as file:
        data = file.readlines()
        names_list = [line.split(',')[0] for line in data]
        if name not in names_list:
            now = datetime.now()
            time_string = now.strftime('%H:%M:%S')
            date_string = now.strftime('%Y-%m-%d')
            file.write(f'\n{name},{time_string},{date_string}')

# Main function
def main():
    images, names = load_images(IMAGE_PATH)
    encodings = encode_faces(images)

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        img_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        faces_cur_frame = face_recognition.face_locations(img_rgb)
        encodings_cur_frame = face_recognition.face_encodings(img_rgb, faces_cur_frame)

        for encoding, face_loc in zip(encodings_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(encodings, encoding)
            face_distances = face_recognition.face_distance(encodings, encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = names[best_match_index].upper()
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
