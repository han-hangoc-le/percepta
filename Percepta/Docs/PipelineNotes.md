## YOLOv8 + AR Pipeline Notes

### 1. Converting YOLOv8 to Core ML
1. Install `ultralytics` and `coremltools` in a Python environment.
2. Export a Core ML package (choose the `n` or `s` model size for on-device use):
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.export(format="coreml", imgsz=640, half=True)
   ```
3. Drag the generated `.mlmodel` into Xcode; it will compile into `YOLOv8n.mlmodelc` at build time.
4. Ensure the filename matches the `modelFileName` argument in `DetectionCoordinator` (default is `YOLOv8n`).

### 2. Backend contract for `/objects`
```jsonc
POST /api/objects
{
  "clientObjectId": "uuid-from-ios",
  "lensMode": "mathematician",
  "label": "cup",
  "confidence": 0.92,
  "boundingBox": { "x": 0.41, "y": 0.33, "width": 0.12, "height": 0.18 },
  "imageBase64": "<jpeg>"
}
```
Response:
```jsonc
{
  "clientObjectId": "uuid-from-ios",
  "annotatedImageBase64": "<jpeg>",
  "message": "Applied mathematician style."
}
```
Use the same normalized bounding box space (origin bottom-left, values 0–1). The server should debounce requests per `clientObjectId` so uploads remain idempotent.

#### Running the local Flask pipeline
1. `cd server && python -m venv .venv && source .venv/bin/activate`
2. `pip install flask flask-cors python-dotenv cerebras-cloud-sdk google-genai rembg pillow`
3. Export the required API keys (`CEREBRAS_API_KEY`, `GOOGLE_GENAI_API_KEY`).
4. `python app.py` to start the server (defaults to port **5050**, override with `PORT` env var if needed).
5. On the iPhone, set the app base URL to `http://<your-mac-ip>:5050/api` (already hard-coded to `http://10.25.19.251:5050/api` in `APIService`). Each detected object will POST its cropped image to `/api/objects`, so the pipeline runs once per object automatically.

### 3. Object lifecycle
1. `YOLODetector` emits detections → `ObjectTracker` keeps stable IDs by IoU.
2. New or significantly-moved tracks trigger a crop + upload through `ObjectProcessingService`.
3. Responses carry back a stylized texture; `DetectionCoordinator` anchors them via RealityKit billboards at raycasted world positions.
4. When tracks expire (object leaves the frame), anchors and local state are cleaned automatically.

### 4. RealityKit alignment tips
- The billboard plane is 25 cm wide; adjust `width` in `DetectionCoordinator.makeBillboard` if you need larger assets.
- Because ARCore-style depth may be noisy indoors, you can lower `frameThrottle` or add plane detection hints for more stable anchors.

### 5. Testing checklist
1. Launch on a physical device (ARKit + camera unavailable in Simulator).
2. Confirm camera permission prompt appears and session starts.
3. Verify bounding boxes track objects smoothly and only one upload fires per object.
4. Inspect backend logs to ensure the `/objects` payload includes base64 data + normalized box coordinates.
5. Validate anchored images stay attached to objects after light movement; tweak IoU thresholds or anchor sizing if jitter occurs.
