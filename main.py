import cv2
import time
from datetime import datetime
import os
from person_detector import PersonDetector

# CONFIGURATION
MIN_AREA = 500
WEBCAM_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
SNAPSHOT_COOLDOWN = 15  # Seconds between snapshots (Streetlight mode)

def main():
    # Setup camera with MSMF (Default Windows) but forcing resolution
    print("Initializing Camera with MSMF...")
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_MSMF)
    
    # Force resolution to standard VGA
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Camera initialized successfully.")
    # Setup Detector
    try:
        detector = PersonDetector()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Security Camera Started. Press 'q' to quit.")

    first_frame = None
    last_snapshot_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame (ret=False). Exiting loop.")
            break
        
        # Debug: Print frame shape once
        if first_frame is None: 
            print(f"First frame received. Shape: {frame.shape}")

        # Resize
        frame = cv2.resize(frame, (640, 480))
        original_frame = frame.copy() # Keep clean copy for saving
        
        # Cooldown Logic (Streetlight style)
        current_time = time.time()
        time_since_last = current_time - last_snapshot_time
        
        if time_since_last < SNAPSHOT_COOLDOWN:
            remaining = int(SNAPSHOT_COOLDOWN - time_since_last)
            cv2.putText(frame, f"Wait: {remaining}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Security Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

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
            
            trigger_label = "motion"
            status_text = "Motion Detected"
            color = (255, 0, 0) # Blue for general motion

            if persons:
                trigger_label = "person"
                status_text = f"WARNING: Person Detected ({len(persons)})"
                color = (0, 0, 255) # Red
                
                # Draw boxes for people
                for (x1, y1, x2, y2) in persons:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Save Snapshot (For any motion)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections/{trigger_label}_{timestamp}.jpg"
            cv2.imwrite(filename, original_frame)
            print(f"Snapshot saved: {filename}")
            last_snapshot_time = current_time # Reset cooldown timer

        # Draw UI
        cv2.putText(frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Security Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
