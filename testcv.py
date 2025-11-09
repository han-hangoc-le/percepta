# IMPORTANT: This code is for testing the CV logic on laptop camera (segmentation + rough info overlay + toggleable contour mode). Please ensure the model yolov8n-seg.pt is in the same directory as this script.

import cv2
from ultralytics import YOLO
import base64
import numpy as np
import textwrap

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
    "person": (
        "From a biologist's view, a human is a complex multicellular organism composed of "
        "trillions of cells working together. Our physiology, behavior, and cognition evolved "
        "to adapt and communicate in social environments, balancing survival and cooperation."
    ),
    "dog": (
        "Dogs (Canis lupus familiaris) are domesticated descendants of wolves. "
        "Biologically, they exhibit an exceptional sense of smell and emotional intelligence, "
        "shaped through thousands of years of human companionship."
    ),
    "bottle": (
        "Though man-made, a bottle represents how humans manipulate organic and inorganic materials "
        "to create containers — mimicking natural structures like fruit skins that protect liquids."
    ),
    "cell phone": (
        "A cell phone may not be biological, but it's an extension of human behavior — "
        "enhancing communication, a key evolutionary advantage in our species."
    ),
    "laptop": (
        "From a biologist's lens, a laptop is an artificial neural extension of the human brain, "
        "designed to store, process, and communicate information — echoing the very functions "
        "of our nervous system."
    ),
    "chair": (
        "A chair supports the human skeletal system by maintaining posture and reducing "
        "muscle strain. While not living, it interacts closely with human biomechanics."
    ),
    "mouse": (
        "In biology, a mouse refers to a small mammal used widely in genetic and behavioral studies. "
        "However, this device mimics natural hand-eye coordination, converting muscle movement into control."
    ),
    "orange": (
        "Oranges are fruits produced by flowering plants in the genus Citrus. Biologically, "
        "they are rich in vitamin C and evolved their bright color and sweet taste to attract animals "
        "for seed dispersal — a beautiful example of coevolution."
    ),
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
            glow = cv2.GaussianBlur(outlined_frame, (9, 9), 0)
            outlined_frame = cv2.addWeighted(outlined_frame, 1.0, glow, 0.4, 0)

            x1, y1, x2, y2 = map(int, boxes[i])
            cropped_base64 = crop_and_encode(frame, boxes[i])
            detected_objects.append({
                'label': label_name,
                'confidence': float(scores[i]),
                'box': (x1, y1, x2, y2),
                'cropped_base64': cropped_base64
            })

    else:
        outlined_frame = frame.copy()
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                if scores[i] < 0.35:
                    continue
                label_name = results.names[int(labels[i])]
                color = (0, 255, 255)
                mask_uint8 = (mask * 255).astype(np.uint8)

                # Draw only contour outline
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

    # ================================
    # 6. Draw info boxes + arrows
    # ================================
    for obj in detected_objects:
        label = obj['label']
        x1, y1, x2, y2 = obj['box']
        info_text = object_info.get(label, "No info available from biological perspective.")

        # === Color Scheme (soft neon tone) ===
        color = (0, 215, 255)  # golden-teal neon

        # === Text Wrapping ===
        max_chars_per_line = 45
        wrapped_lines = textwrap.wrap(info_text, width=max_chars_per_line)

        # === Measure text size to resize box dynamically ===
        font_scale = 0.45
        font_thickness = 1
        line_height = 18
        box_padding = 10

        max_line_width = 0
        for line in wrapped_lines:
            (line_width_px, line_height_px), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            max_line_width = max(max_line_width, line_width_px)

        text_height = line_height * (len(wrapped_lines) + 2)
        box_width = max(240, min(max_line_width + 20, 400))  # keep within reasonable range
        box_height = text_height + box_padding

        # === Box position (above the object, with margin) ===
        box_x = x1 - 10
        box_y = max(30, y1 - box_height - 20)
        box_w = box_width
        box_h = box_height

        # === Semi-transparent info box background ===
        overlay = outlined_frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, outlined_frame, 0.4, 0, outlined_frame)

        # === Neon border and title ===
        cv2.rectangle(outlined_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2)
        cv2.putText(outlined_frame, f"{label.title()}", (box_x + 10, box_y + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)

        # === Wrapped info text ===
        y_text = box_y + 45
        for line in wrapped_lines:
            cv2.putText(outlined_frame, line, (box_x + 10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            y_text += line_height

        # === Arrow from info box → nearest edge of object bbox ===
        box_center_x = box_x + box_w // 2
        box_bottom_y = box_y + box_h
        target_x = int((x1 + x2) / 2)
        target_y = y1
        cv2.arrowedLine(outlined_frame, (box_center_x, box_bottom_y), (target_x, target_y),
                        color, 2, tipLength=0.05)
    
    cv2.imshow("Biology Lens", outlined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('o'):
        outline_mode = not outline_mode
        print(f"Outline mode: {'ON' if outline_mode else 'OFF'}")

# ================================
# 8. Cleanup
# ================================
cap.release()
cv2.destroyAllWindows()
