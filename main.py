import cv2
import time
from datetime import datetime
import os
from person_detector import PersonDetector

MIN_AREA = 500
WEBCAM_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
SNAPSHOT_COOLDOWN = 5

def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    time.sleep(0.5)
    return cap

def main():
    try:
        detector = PersonDetector()
    except Exception as e:
        print(f"Model load error: {e}")
        return

    print("Security Camera Started. Press 'q' to quit.")
    os.makedirs("detections", exist_ok=True)

    cap = open_camera()
    time.sleep(1)

    if not cap.isOpened():
        print("❌ Camera could not be opened.")
        return

    last_snapshot_time = 0
    first_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠️ Camera disconnected. Reinitializing...")
            cap.release()
            time.sleep(2)
            cap = open_camera()
            first_frame = None
            continue

        frame = cv2.resize(frame, (640, 480))
        original_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = gray
            continue

        delta_frame = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = any(cv2.contourArea(c) > MIN_AREA for c in contours)

        status_text = "Status: Monitoring"
        color = (0, 255, 0)

        if motion_detected:
            persons = detector.detect_people(frame, CONFIDENCE_THRESHOLD)

            if persons:
                status_text = f"WARNING: Person Detected ({len(persons)})"
                color = (0, 0, 255)

                for (x1, y1, x2, y2) in persons:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, "Person",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

                now = time.time()
                if now - last_snapshot_time > SNAPSHOT_COOLDOWN:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detections/person_{ts}.jpg"
                    cv2.imwrite(filename, original_frame)
                    print(f"📸 Snapshot saved: {filename}")
                    last_snapshot_time = now

        cv2.putText(frame, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Security Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()