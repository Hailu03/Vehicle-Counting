import cv2
import torch
import numpy as np 
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

video_path = 'datasets/sample_video.mp4'
conf_thresh = 0.5
tracking_class = [2, 5, 7]  # Example classes for cars

# Initialize DeepSORT Tracker and YOLO Model
tracker = DeepSort(max_age=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights='weights/yolov9-c-converted.pt', device=device, fuse=True)
model = AutoShape(model)

# Load class names and set random colors for each class
with open('datasets/classes.names', 'r') as f:
    classes = f.read().strip().split('\n')
color = np.random.randint(0, 255, (len(classes), 3))

# Video capture and frame rate settings
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  
frame_delay = int(1000 / fps)

# Define line positions for counting
line_y_up = 420  # Y-coordinate of the upward counting line
line_y_down = 600  # Y-coordinate of the downward counting line

# Initialize counters and dictionary to store previous vehicle positions
num_car_upwards = 0
num_car_downwards = 0
previous_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)
    detect = []
    for detect_object in results.pred[0]:
        label, conf, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if class_id in tracking_class and conf >= conf_thresh:
            detect.append([[x1, y1, x2 - x1, y2 - y1], conf, class_id])

    # Update tracker with detection results
    tracks = tracker.update_tracks(detect, frame=frame)

    # Draw lines for counting
    cv2.line(frame, (0, line_y_up), (frame.shape[1], line_y_up), (0, 255, 0), 2)
    cv2.line(frame, (0, line_y_down), (frame.shape[1], line_y_down), (0, 0, 255), 2)

    for track in tracks:
        if track.is_confirmed() and track.time_since_update == 0:
            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = bbox
            class_id = track.get_det_class()

            # Calculate the center of the bounding box
            center_y = (y1 + y2) // 2

            # Check if the track_id exists in previous positions
            if track_id in previous_positions:
                prev_center_y = previous_positions[track_id]

                # Check if the vehicle crossed the upward line
                if prev_center_y > line_y_up and center_y <= line_y_up:
                    num_car_upwards += 1
                # Check if the vehicle crossed the downward line
                elif prev_center_y < line_y_down and center_y >= line_y_down:
                    num_car_downwards += 1

            # Update the position of the vehicle in the dictionary
            previous_positions[track_id] = center_y

            # Set color and label for bounding box
            color_id = int(class_id)
            color_bbox = color[color_id]
            B, G, R = map(int, color_bbox)

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (B, G, R), 2)
            cv2.rectangle(frame, (int(x1), int(y1) - 20), (int(x1) + len(classes[class_id]) * 12, int(y1)), (B, G, R), -1)
            cv2.putText(frame, f"{classes[class_id]} {track_id}", (int(x1) + 5, int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display vehicle counts on the frame
    cv2.putText(frame, 'Car Upwards: ' + str(num_car_upwards), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'Car Downwards: ' + str(num_car_downwards), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow('Tracking', frame)

    # Exit condition
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
