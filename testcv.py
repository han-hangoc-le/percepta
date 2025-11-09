# IMPORTANT: This code is for testing the CV logic on laptop camera (segmentation + rough info overlay + toggleable contour mode). Please ensure the model yolov8n-seg.pt is in the same directory as this script.

import cv2
from ultralytics import YOLO
import base64
import numpy as np

# ================================
# 1. Load YOLOv8n-seg model
# ================================
model = YOLO("yolov8n-seg.pt")

# ================================
# 2. Initialize laptop camera
# ================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# ================================
# 3. Helper function to encode cropped objects to base64
# ================================
def crop_and_encode(frame, box):
    x1, y1, x2, y2 = map(int, box)
    cropped = frame[y1:y2, x1:x2]
    _, buffer = cv2.imencode('.jpg', cropped)
    return base64.b64encode(buffer).decode('utf-8')

# ================================
# 4. Example object info dictionary
# ================================
object_info = {
    "person": "Humans are social beings with reasoning ability.",
    "dog": "Dogs are domesticated mammals known for loyalty.",
    "bottle": "A bottle is a container for liquids.",
    "cell phone": "A mobile device for communication.",
    "laptop": "Portable computer for work or study.",
    # Add more as needed...
}

# ================================
# 5. Main loop
# ================================
outline_mode = False  # press 'o' to toggle

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results, 'boxes') else []
    scores = results.boxes.conf.cpu().numpy() if hasattr(results, 'boxes') else []
    labels = results.boxes.cls.cpu().numpy() if hasattr(results, 'boxes') else []

    detected_objects = []

    # Choose base frame
    if outline_mode and results.masks is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        outlined_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        masks = results.masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            if scores[i] < 0.35:
                continue

            label_name = results.names[int(labels[i])]
            color = (0, 255, 0)

            mask_uint8 = (mask * 255).astype(np.uint8)
            obj_region = cv2.bitwise_and(frame, frame, mask=mask_uint8)
            gray_obj = cv2.cvtColor(obj_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_obj, 80, 150)
            edges = cv2.bitwise_and(edges, mask_uint8)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_colored[np.where((edges_colored != [0, 0, 0]).all(axis=2))] = color
            outlined_frame = cv2.addWeighted(outlined_frame, 1.0, edges_colored, 1.0, 0)

            # Outer contour
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(outlined_frame, contours, -1, color, 2)

            x1, y1, x2, y2 = map(int, boxes[i])
            cropped_base64 = crop_and_encode(frame, boxes[i])
            detected_objects.append({
                'label': label_name,
                'confidence': float(scores[i]),
                'box': (x1, y1, x2, y2),
                'cropped_base64': cropped_base64
            })

    else:
        outlined_frame = results.plot()
        for i, box in enumerate(boxes):
            label_name = results.names[int(labels[i])]
            conf = float(scores[i])
            if conf < 0.35:
                continue
            cropped_base64 = crop_and_encode(frame, box)
            x1, y1, x2, y2 = map(int, box)
            detected_objects.append({
                'label': label_name,
                'confidence': conf,
                'box': (x1, y1, x2, y2),
                'cropped_base64': cropped_base64
            })

    # ================================
    # 6. Draw info boxes + arrows
    # ================================
    for obj in detected_objects:
        label = obj['label']
        x1, y1, x2, y2 = obj['box']
        info_text = object_info.get(label, "No info available.")
        color = (255, 200, 50)

        # Compute object center
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Info box position (top-left corner)
        box_x, box_y = x1 - 10, max(30, y1 - 60)
        box_w, box_h = 240, 50

        # Draw box background
        cv2.rectangle(outlined_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 40, 40), -1)
        cv2.rectangle(outlined_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)

        # Text
        cv2.putText(outlined_frame, f"{label.title()}", (box_x + 10, box_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(outlined_frame, info_text, (box_x + 10, box_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Arrow from info box to object center
        cv2.arrowedLine(outlined_frame, (box_x + box_w // 2, box_y + box_h), (center_x, center_y),
                        color, 2, tipLength=0.05)

    # ================================
    # 7. Display info
    # ================================
    if detected_objects:
        labels_found = ", ".join(list({d['label'] for d in detected_objects})[:3])
        status_message = f"Found: {labels_found}{'...' if len(detected_objects) > 3 else ''}"
    else:
        status_message = "No objects detected"

    mode_text = "Mode: Info + Contour" if outline_mode else "Mode: Info + Normal"
    cv2.putText(outlined_frame, status_message, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(outlined_frame, mode_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    cv2.imshow("YOLOv8 Smart Overlay", outlined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('o'):
        outline_mode = not outline_mode
        print(f"Toggled mode: {'Internal Contour' if outline_mode else 'Normal'}")

# ================================
# 8. Cleanup
# ================================
cap.release()
cv2.destroyAllWindows()
