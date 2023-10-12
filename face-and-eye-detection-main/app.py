from flask import Flask, render_template, Response
import cv2
import skimage as ski
import numpy as np

app = Flask(__name__)

def generate_frames():
    vid = cv2.VideoCapture(0)  # Open the camera (0 represents default camera)
    while True:
        ack ,img = vid.read()
        if not ack:
            break
        else:
            fd=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier( cv2.data.haarcascades + 'haarcascade_eye.xml')  

            vid=cv2.VideoCapture(0)
            counter=0
            while True:
                ack,img=vid.read()
                if ack:
                    faces=fd.detectMultiScale(img,1.2,2)
                    if len(faces)==1:
                        #counter+=1
                        for x,y,w,h in faces:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)

                
                        eye_rect = eye_cascade.detectMultiScale(img)  
                        for (x1, y1, w1, h1) in eye_rect:  
                                cv2.rectangle(img, (x1, y1),  
                                (x1 + w1, y1 + h1), (255, 0, 0), 5)      

                        ##
                        ack, buffer = cv2.imencode('.jpg', img)
                        img = buffer.tobytes()
                        yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
