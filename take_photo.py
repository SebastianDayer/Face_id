import cv2
import os
import numpy as np
from datetime import datetime
import time
import pathlib

PERSON_NAME = "Sebastian"  

def capture_photos(name):
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Allow camera to warm up
    time.sleep(.5)

    photo_count = 0
    
    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")
    
    while True:
        # Capture frame webcam
        success, frame = cap.read()

        if not success:
            print('Not able to read frame. End.')
            break
        
        frame2 = cv2.GaussianBlur(frame, (31, 31), 0)

        combinedImages = np.hstack((frame, frame2))

        # Display the frame
        cv2.imshow('Capture', combinedImages)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{name}_{timestamp}.png"
            filepath = os.path.join('/home/dayseb/.keras/datasets/face_detection_photos/photos/sebastian', file_name)
            cv2.imwrite(filepath, frame2)
            print(f"Photo {photo_count} saved: {filepath}")
        
        elif key == ord('q'):  # Q key
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")

if __name__ == "__main__":
    capture_photos(PERSON_NAME)

