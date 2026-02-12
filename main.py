import cv2
import time
from datetime import datetime
import platform
from person_detector import PersonDetector

MIN_AREA = 500
WEBCAM_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
SNAPSHOT_COOLDOWN = 5

def open_camera():
    system = platform.system()
    if system == 'Windows':
        cap = cv2.VideoCapture(0) # Reverting to default (likely MSMF) as DSHOW is hanging
    elif system == 'Darwin':
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(0)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    time.sleep(1.0)
    return cap

def enhance_for_low_light(frame):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    to enhance details in low light conditions.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge and convert back to BGR
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

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

    print("Starting camera... (If this hangs, please check camera connection)")
    cap = open_camera()
    time.sleep(1)

    if not cap.isOpened():
        print("❌ Camera could not be opened. Please check if another app is using it.")
        return

    print("✅ Camera started successfully.")
    first_frame = None

    while True:
        try:
            ret, frame = cap.read()

            if not ret:
                print("⚠️ Camera disconnected or empty frame. Reinitializing...")
                cap.release()
                time.sleep(2)
                cap = open_camera()
                first_frame = None
                continue
            
            frame = cv2.resize(frame, (640, 480))
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            break
        original_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (0-255)
        brightness = cv2.mean(gray)[0]
        is_low_light = brightness < 60
        
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if first_frame is None:
            first_frame = gray_blur
            continue

        delta_frame = cv2.absdiff(first_frame, gray_blur)
        thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = any(cv2.contourArea(c) > MIN_AREA for c in contours)

        status_text = "Status: Monitoring"
        color = (0, 255, 0)
        
        # Process frame for detection/display
        display_frame = frame
        if is_low_light:
            display_frame = enhance_for_low_light(frame)
            status_text += " (Low Light)"

        if motion_detected:
            # Detect on the best available frame (enhanced if dark)
            persons = detector.detect_people(display_frame, CONFIDENCE_THRESHOLD)

            if persons:
                status_text = f"WARNING: Person Detected ({len(persons)})"
                if is_low_light:
                     status_text += " (Low Light)"
                color = (0, 0, 255)

                for (x1, y1, x2, y2) in persons:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, "Person",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

                now = time.time()
                if now - last_snapshot_time > SNAPSHOT_COOLDOWN:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detections/person_{ts}.jpg"
                    # Save the enhanced frame if in low light, or original? 
                    # User probably wants to see WHO it is, so enhanced is better.
                    cv2.imwrite(filename, display_frame) 
                    print(f"📸 Snapshot saved: {filename}")
                    last_snapshot_time = now

        cv2.putText(display_frame, status_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Security Feed", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()