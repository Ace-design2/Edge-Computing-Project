import cv2
import time
from datetime import datetime
import os
from person_detector import PersonDetector

# CONFIGURATION
MIN_AREA = 500
WEBCAM_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
SNAPSHOT_COOLDOWN = 5  # Seconds between snapshots

def main():
    # Setup Detector
    try:
        detector = PersonDetector()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Security Camera Started. Press 'q' to quit.")

    last_snapshot_time = 0

    while True:
        print("Initializing Camera...")
        # Revert to default backend (Auto/MSMF) but with reconnection logic
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        
        print("Checking if camera is opened...")
        if not cap.isOpened():
            print("Error: Could not open video device. Retrying in 5 seconds...")
            time.sleep(5)
            continue
        print("Camera is opened.")
        
        # Warmup camera to allow auto-focus/exposure and initialization
        print("Warming up camera for 2 seconds...")
        time.sleep(2)
        
        
        # Optional: Force resolution to ensure consistency
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Camera initialized successfully.")
        first_frame = None

        while True:
            # print("Reading frame...") 
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed. Reconnecting...")
                break

            # Resize
            frame = cv2.resize(frame, (640, 480))
            original_frame = frame.copy() # Keep clean copy for saving
            
            # Motion Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if first_frame is None:
                first_frame = gray
                continue

            delta_frame = cv2.absdiff(first_frame, gray)
            thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) > MIN_AREA:
                    motion_detected = True
                    break # Optimization: One motion blob is enough to trigger check
            
            status_text = "Status: Monitoring"
            color = (0, 255, 0) # Green

            if motion_detected:
                # Run Person Detection
                persons = detector.detect_people(frame, conf_threshold=CONFIDENCE_THRESHOLD)
                
                if persons:
                    status_text = f"WARNING: Person Detected ({len(persons)})"
                    color = (0, 0, 255) # Red
                    
                    # Draw boxes
                    for (x1, y1, x2, y2) in persons:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Save Snapshot
                    current_time = time.time()
                    if current_time - last_snapshot_time > SNAPSHOT_COOLDOWN:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"detections/person_{timestamp}.jpg"
                        # Ensure directory exists
                        os.makedirs("detections", exist_ok=True)
                        cv2.imwrite(filename, original_frame)
                        print(f"Snapshot saved: {filename}")
                        last_snapshot_time = current_time

            # Draw UI
            cv2.putText(frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Security Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
