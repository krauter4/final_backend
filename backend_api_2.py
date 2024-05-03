from flask import Flask, Response, jsonify
from flask_cors import CORS, cross_origin
import base64
import cv2
import imutils
import face_recognition
import pickle
import time
from imutils.video import VideoStream, FPS

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load known faces and embeddings
encodingsP = "encodings.pickle"
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

# Global variables
global latest_names, latest_frame
latest_names = 'Unknown'
latest_frame = None

def start_video_stream():
    global latest_names, latest_frame
    latest_names = 'Unknown'
    vs = VideoStream(src=0, framerate=10).start()
    time.sleep(2.0)
    fps = FPS().start()

    while True:
        latest_names = 'Empty'
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        boxes = face_recognition.face_locations(frame)
        print(len(boxes))
        if len(boxes) == 0:
            latest_names = 'Empty'
        else:
            print('Human detected!')
            encodings = face_recognition.face_encodings(frame, boxes)
            names = []

            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                latest_names = name
                print(latest_names)

                names.append(name)

            for ((top, right, bottom, left), name) in zip(boxes, names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            latest_frame = buffer.tobytes()

        # cv2.imshow("Facial Recognition is Running", frame)
        time.sleep(1.5)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        fps.update()

    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()

@app.route('/video')
def video_feed():
    global latest_frame, latest_names
    if latest_names == 'Unknown':
        response = {
            'frame': base64.b64encode(latest_frame).decode('utf-8'),
            'name': latest_names
        }
        return jsonify(response)
    else:
        response = {
            'frame': '',
            'name': latest_names
        }
        return jsonify(response)

if __name__ == '__main__':
    from threading import Thread
    print("Starting video stream...")
    thread = Thread(target=start_video_stream)
    thread.start()
    print("Starting backend server...")
    time.sleep(2)
    app.run(host='0.0.0.0', debug=True, use_reloader=False)

