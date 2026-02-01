import cv2
import time

def test_camera(index, backend_name, backend_id):
    print(f"Testing camera index {index} with backend {backend_name}...")
    if backend_id is not None:
        cap = cv2.VideoCapture(index, backend_id)
    else:
        cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Failed to open camera with {backend_name}")
        return False
    
    print(f"Camera opened with {backend_name}. Reading frames...")
    
    # Warmup
    time.sleep(2)
    
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i} captured successfully. Shape: {frame.shape}")
        else:
            print(f"Frame {i} capture failed.")
            
    cap.release()
    print("Test complete.\n")

print("OpenCV Version:", cv2.__version__)

# Test default
test_camera(0, "Default", None)

# Test AVFoundation (standard for macOS)
test_camera(0, "AVFoundation", cv2.CAP_AVFOUNDATION)

# Test index 1 (sometimes implied if 0 fails)
test_camera(1, "Default (Index 1)", None)
