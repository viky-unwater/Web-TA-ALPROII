from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from cvzone.ClassificationModule import Classifier
import string

app = Flask(_name_)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1, max_num_hands=2)
classifier = Classifier("data/1_Model/keras_model.h5", "data/1_Model/labels.txt")
offset = 20
imgSize = 500
labels = list(string.ascii_uppercase)

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        all_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                all_landmarks.extend(hand_landmarks.landmark)

                # Draw landmarks and connections for each hand
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # Extract bounding box for all detected hands
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
            for landmark in all_landmarks:
                x_min = min(x_min, landmark.x)
                y_min = min(y_min, landmark.y)
                x_max = max(x_max, landmark.x)
                y_max = max(y_max, landmark.y)

            x_min, y_min, x_max, y_max = (
                max(0, int(x_min * frame.shape[1]) - offset),
                max(0, int(y_min * frame.shape[0]) - offset),
                min(frame.shape[1], int(x_max * frame.shape[1]) + offset),
                min(frame.shape[0], int(y_max * frame.shape[0]) + offset),
            )

            # Calculate aspect ratio of hands image
            aspect_ratio = (x_max - x_min) / (y_max - y_min)

            # Create a white image with the correct data type
            imgWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 200

            # Crop the hands_img and resize it to fit inside imgWhite
            hands_img = frame[y_min:y_max, x_min:x_max]
            h, w, _ = hands_img.shape

            if aspect_ratio > 1:
                wCal = int(imgSize / aspect_ratio)
                imgResize = cv2.resize(hands_img, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                hCal = int(imgSize * aspect_ratio)
                imgResize = cv2.resize(hands_img, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Perform prediction on imgWhite
            prediction, index = classifier.getPrediction(imgWhite)
            print("Prediction for hands:", prediction)

            # Display the prediction on the frame
            cv2.putText(frame, f"Prediction: {labels[index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Convert the frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if _name_ == "_main_":
    app.run(debug=True)