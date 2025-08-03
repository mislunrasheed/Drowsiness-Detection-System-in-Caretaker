
from flask import Flask, render_template, Response, request, jsonify
import cv2
import time
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

app = Flask(__name__)

# Global variables for Twilio credentials (set via form)
TWILIO_SID = None
TWILIO_AUTH_TOKEN = None
TWILIO_PHONE_NUMBER = None
PARENT_PHONE_NUMBER = None

# Load Haar cascade models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Camera setup
cap = cv2.VideoCapture(0)
drowsy_start_time = None
DROWSY_THRESHOLD = 15  # 15 seconds
sms_sent = False
no_face_start_time = None  # Timer for no-face detection
NO_FACE_THRESHOLD = 10  # 10 seconds for no-face alert
sms_sent_no_face = False  # Flag for no-face SMS

# Function to send SMS for drowsiness
def send_sms():
    global TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, PARENT_PHONE_NUMBER
    if not all([TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, PARENT_PHONE_NUMBER]):
        print("[✖] Error: Twilio credentials not set.")
        return
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="[ALERT] The babysitter appears drowsy.",
            from_=TWILIO_PHONE_NUMBER,
            to=PARENT_PHONE_NUMBER
        )
        print("[✔] SMS Sent Successfully! Message SID:", message.sid)
    except TwilioRestException as e:
        print("[✖] Twilio Error:", str(e))
    except Exception as e:
        print("[✖] General Error sending SMS:", str(e))

# Function to send SMS for no face detected
def send_no_face_sms():
    global TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, PARENT_PHONE_NUMBER
    if not all([TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, PARENT_PHONE_NUMBER]):
        print("[✖] Error: Twilio credentials not set.")
        return
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body="[ALERT] The babysitter is not facing the camera.",
            from_=TWILIO_PHONE_NUMBER,
            to=PARENT_PHONE_NUMBER
        )
        print("[✔] No-Face SMS Sent Successfully! Message SID:", message.sid)
    except TwilioRestException as e:
        print("[✖] Twilio Error:", str(e))
    except Exception as e:
        print("[✖] General Error sending no-face SMS:", str(e))

# Video feed generator with improved face detection for tilted heads
def generate_frames():
    global drowsy_start_time, sms_sent, no_face_start_time, sms_sent_no_face
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[✖] Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces with more sensitive parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # Reduced from 1.3 for better tilt detection
            minNeighbors=3,   # Reduced from 5 for less strict detection
            minSize=(100, 100)  # Unchanged to filter smaller faces
        )

        # If no faces detected, try rotated frames
        if len(faces) == 0:
            for angle in [-15, 15]:  # Check ±15° rotations
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_gray = cv2.warpAffine(gray, M, (w, h))
                faces = face_cascade.detectMultiScale(
                    rotated_gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(100, 100)
                )
                if len(faces) > 0:
                    # Use detected faces from rotated frame (simplified coordinates)
                    break  # Exit rotation loop if faces found

        print(f"[DEBUG] Faces detected: {len(faces)}")

        # Check if no faces are detected
        if len(faces) == 0:
            if no_face_start_time is None:
                no_face_start_time = time.time()
                print("[TIMER] No face detected - timer started")
            else:
                no_face_elapsed_time = time.time() - no_face_start_time
                print(f"[TIMER] No face for {int(no_face_elapsed_time)}s")
                if no_face_elapsed_time >= NO_FACE_THRESHOLD and not sms_sent_no_face:
                    print("[ALERT] Sending no-face SMS")
                    send_no_face_sms()
                    sms_sent_no_face = True
            cv2.putText(frame, "NO FACE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Reset no-face timer and flag when a face is detected
            no_face_start_time = None
            sms_sent_no_face = False

            # Original drowsiness detection logic (unchanged)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.2,
                    minNeighbors=10,
                    minSize=(20, 20)
                )
                print(f"[DEBUG] Eyes detected: {len(eyes)}")

                if len(eyes) >= 2:
                    print("[STATUS] Awake")
                    drowsy_start_time = None
                    sms_sent = False
                    cv2.putText(frame, "AWAKE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    if drowsy_start_time is None:
                        drowsy_start_time = time.time()
                        print("[TIMER] Started")
                    else:
                        elapsed_time = time.time() - drowsy_start_time
                        print(f"[TIMER] Drowsy for {int(elapsed_time)}s")
                        if elapsed_time >= DROWSY_THRESHOLD and not sms_sent:
                            print("[ALERT] Sending SMS")
                            send_sms()
                            sms_sent = True
                        cv2.putText(frame, f"DROWSY ({int(elapsed_time)}s)", (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('pro.html')

@app.route('/start-detection', methods=['POST'])
def start_detection():
    global TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, PARENT_PHONE_NUMBER
    data = request.get_json()
    TWILIO_SID = data['twilio_sid']
    TWILIO_AUTH_TOKEN = data['twilio_token']
    TWILIO_PHONE_NUMBER = data['twilio_number']
    PARENT_PHONE_NUMBER = data['parent_number']
    return jsonify({"message": "Detection started with provided credentials!"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("[INFO] Starting application...")
    if not cap.isOpened():
        print("[✖] Error: Could not access the camera. Check if it’s connected or in use by another application.")
        exit(1)
    print("[✔] Success! Camera is on.")
    print("[INFO] Starting Flask server on http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"[✖] Error running Flask server: {str(e)}")

