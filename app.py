from flask import Flask, render_template, Response, jsonify
import cv2
import string
import numpy as np
from cvzone.ClassificationModule import Classifier
import mediapipe as mp
import time

app = Flask(__name__)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1, max_num_hands=2)
classifier = Classifier("keras_model.h5", "labels.txt")
offset = 20
imgSize = 500
labels = list(string.ascii_uppercase)

cap = cv2.VideoCapture(0)

last_prediction = ""  # Variable to store the last prediction
prediction = ""

@app.route('/get_prediction')
def get_prediction():
    global last_prediction
    return jsonify({'prediction': last_prediction})

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Preserve the original BGR color format
            frame_rgb = frame.copy()

            results = hands.process(frame_rgb)
            all_landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    all_landmarks.extend(hand_landmarks.landmark)

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

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

                aspect_ratio = (x_max - x_min) / (y_max - y_min)
                imgWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 200

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

                prediction, index = classifier.getPrediction(imgWhite)
                last_prediction = labels[index]
                cv2.putText(frame, f"Prediction: {labels[index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def beranda():
    return render_template('Beranda.html')


@app.route('/interpretasi')
def interpretasi():
    return render_template('Interpretasi.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/tentang')
def tentang():
    return render_template('Tentang.html')


@app.route('/lainnya')
def lainnya():
    return render_template('Lainnya.html')


if __name__ == '__main__':
    app.run(debug=True)