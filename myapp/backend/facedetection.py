# Imports
from deepface import DeepFace
import requests
import os
from datetime import datetime
import cv2
import time
import traceback

# Global variables
current_mood = 'neutral'
detection_active = False

# Utility functions
def get_current_mood():
    global current_mood
    return current_mood

def start_detection():
    global detection_active
    detection_active = True

def stop_detection():
    global detection_active
    detection_active = False

# Main class
class EmotionDetector:
    """
    A class to handle face detection and emotion analysis.
    """
    def __init__(self):
        haarcascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)

    def detect_emotions(self, frame):
        global current_mood
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        print(f"[DEBUG] Number of faces detected: {len(faces)}")
        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                print(f"[DEBUG] face_rgb shape: {face_rgb.shape}, dtype: {face_rgb.dtype}")
                analysis = DeepFace.analyze(
                    img_path=face_rgb,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True
                )
                print(f"[DEBUG] DeepFace analysis result: {analysis}")
                if analysis:
                    detected_mood = analysis[0]['dominant_emotion']
                    current_mood = detected_mood
                    results.append({
                        'box': (x, y, w, h),
                        'emotions': analysis[0]['emotion'],
                        'dominant_emotion': detected_mood
                    })
            except Exception as e:
                print(f"Could not analyze face: {e}")
                traceback.print_exc()
        if not results:
            current_mood = 'none'
        return results

# Integration function
def send_to_n8n_init(user_id, detected_emotion, confidence_score):
    """Send emotion data to n8n and get empathetic response"""
    try:
        # Respect feature flag
        if not (os.getenv('ENABLE_N8N_EMOTION', 'false').lower() in ('1', 'true', 'yes')):
            return {"success": False, "response": "n8n emotion integration disabled"}

        n8n_url = os.getenv('N8N_EMOTION_URL') or os.getenv('N8N_INIT_URL')
        if not n8n_url:
            return {"success": False, "response": "Configuration missing"}

        payload = {
            "user_input": "",
            "emotion": detected_emotion,
            "confidence": confidence_score,
            "user_id": str(user_id),
            "face_coordinates": {},
            "timestamp": datetime.now().isoformat()
        }

        headers = {'Content-Type': 'application/json'}
        secret = os.getenv('N8N_WEBHOOK_SECRET')
        if secret:
            import hmac, hashlib, json
            sig = hmac.new(secret.encode(), json.dumps(payload).encode(), hashlib.sha256).hexdigest()
            headers['X-N8N-SIGNATURE'] = sig

        response = requests.post(n8n_url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "response": "Service temporarily unavailable"}
    except Exception as e:
        print(f"n8n integration error: {e}")
        return {"success": False, "response": "I'm here to help. How are you feeling today?"}

# Drawing function
def draw_results(frame, results):
    """Draws emotion detection results on a frame."""
    if not results:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return
    res = results[0]
    x, y, w, h = res['box']
    dominant_emotion = res['dominant_emotion']
    emotions = res['emotions']
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    y_offset = 60
    for emotion, score in emotions.items():
        y_offset += 20

# Main loop
def real_time_emotion_detection():
    """Demonstrates real-time emotion detection using a webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = EmotionDetector()
    last_check_time = 0
    check_interval = 0.3
    last_results = []
    global detection_active
    detection_active = True
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        now = time.time()
        if detection_active and (now - last_check_time >= check_interval):
            last_results = detector.detect_emotions(frame)
            last_check_time = now
            # --- Response generation using n8n ---
            if last_results:
                # Use first detected face for response
                detected_emotion = last_results[0]['dominant_emotion']
                confidence_score = max(last_results[0]['emotions'].values())
                user_id = "user_123"  # Replace with actual user_id if available
                n8n_result = send_to_n8n_init(user_id, detected_emotion, confidence_score)
                if n8n_result.get("success"):
                    final_response = n8n_result.get("response")
                    support_resources = n8n_result.get("support_resources", [])
                else:
                    final_response = "Your existing fallback response here"
                    support_resources = []
                print(f"[RESPONSE] {final_response}")
                if support_resources:
                    print(f"[SUPPORT RESOURCES] {support_resources}")
        draw_results(frame, last_results)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Script entry point
if __name__ == "__main__":
    real_time_emotion_detection()