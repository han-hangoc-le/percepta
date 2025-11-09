from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from phase2_vertex import generate_vertex_overlay
import os, json, re, html, base64, binascii
from uuid import uuid4
from typing import Tuple
from cerebras.cloud.sdk import Cerebras

# -------------------- SETUP --------------------
load_dotenv()
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("outputs", exist_ok=True)
API_PREFIX = "/api"

client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

# -------------------- LENS PROMPTS --------------------
LENS_PROMPTS = {
    "math": (
        "You are a mathematics lens assistant. Explain the object's measurable "
        "geometry, dimensions, and related equations such as area, volume, or perimeter."
    ),
    "physics": (
        "You are a physics lens assistant. Describe the object's physical properties such as "
        "motion, energy, or forces, and include relevant equations like F = ma or E = mc²."
    ),
}

# -------------------- PHASE 1 PROMPT TEMPLATE --------------------
PHASE1_PROMPT = """
You are generating structured data for a mathematical lens AI that will create holographic,
blueprint-style skeleton diagrams of real-world objects.

Given the following information:
Object: {object}
Image (optional): {image_url}

Return a JSON object with exactly four keys:
1. "object" – the object’s name.
2. "equation" – the most relevant mathematical formula describing its measurable property
   (area, volume, surface, or geometric dimension).
3. "explanation" – a short explanation of what the formula means and what each variable represents.
4. "guide" – a complete, detailed rendering prompt describing how to visualize the object
   as a neon-blueprint mathematical schematic.

In "guide", describe:
- The **subject**, referring directly to the object: e.g., “A laptop”, “A cup”, “A bottle”.
- The **style**: glowing neon (electric cyan or blue) on a pure black background.
- The **geometry focus**: explain how to depict its measurable geometry (length, width, radius, height, etc.)
  and its key equation.
- The **mathematical annotation**: label variables such as r, h, w, L, A, V as appropriate.
- The **composition**: describe the view (front, side, top, or isometric) that best shows the measurable dimensions.
- The **objective**: emphasize that the final output should look like a technical or mathematical wireframe,
  NOT a photo or realistic rendering.

Example output:
{{
  "object": "bottle",
  "equation": "V = πr²h",
  "explanation": "Volume of a cylinder where r is radius and h is height.",
  "guide": "Create a highly detailed, neon-blueprint rendering of a cylindrical bottle with a transparent body.
            Use glowing cyan lines on a pure black background. Label radius r at the top and height h along
            the vertical axis. Add the formula V = πr²h in bright blue text below. Include faint dashed lines
            showing the circular base area (πr²)."
}}
"""


PHASE1_PROMPT_PHYSICS = """
Given the following information:

Object: {object}
Image (optional): {image_url}

Return a JSON output with exactly four keys:
1. "object" — repeat the object's name.
2. "equation" — the most relevant PHYSICS equation describing how the object behaves or interacts
   (e.g., Newton’s laws, energy, motion, force, pressure, etc.).
3. "explanation" — a short explanation of what the equation means and how it relates to the object.
4. "guide" — a detailed, visual step-by-step instruction for overlaying AR annotations on the photo.

In "guide", include these details clearly:
- Describe the object's physical context (motion, orientation, forces, or energy aspects).
- Specify physical parameters or variables to label (F, m, a, v, E, P, etc.).
- Describe how to visualize vectors, arrows, or fields (e.g., direction of force, gravity, or velocity).
- Indicate the color and style for each overlay (e.g., red arrows for forces, blue text for equations).
- Show where to place the main equation (above or beside the object) and how to align the arrows or labels.
- Emphasize: keep the real image visible — no redrawing, no object replacement.
- Style: AR physics overlay on real-world photo, visually clear and conceptually accurate.

Example output:
{{
  "object": "ball",
  "equation": "F = m · a",
  "explanation": "Newton's Second Law — the force on an object equals its mass times its acceleration.",
  "guide": "Show a glowing red arrow in the direction of the ball’s motion to represent the force F. "
           "Label 'm' near the ball to indicate its mass, and draw a smaller arrow labeled 'a' to show acceleration. "
           "Place the equation F = m·a above the ball in bright blue text. Keep the ball photo visible."
}}
"""

# -------------------- CLEANER --------------------
def clean_text_for_prompt(text: str) -> str:
    """Normalize special math or formatting symbols for model safety."""
    if not text:
        return ""

    replacements = {
        "·": "×",  
        "x": "×",  
        "X": "×",
        "÷": "÷",
        "^": "^",
        "√": "√",
        "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "•": "-"
    }


    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = re.sub(r"\s+", " ", text).strip()
    text = text.encode("ascii", "ignore").decode("ascii")
    return text

def clean_explanation(text: str) -> str:
    """Cleans and normalizes physics explanations for visual rendering."""
    if not text:
        return ""

    import html, re

    # Decode HTML entities
    text = html.unescape(text)

    # Replace problematic physics/math symbols
    replacements = {
        "²": "^2",
        "³": "^3",
        "·": "*",
        "×": "*",
        "÷": "/",
        "–": "-",
        "—": "-",
        "√": "sqrt",
        "θ": "theta",
        "°": " degrees",
        "±": "+/-",
        "≈": "~",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # Clean residual formatting characters and collapse whitespace
    text = re.sub(r"[^a-zA-Z0-9.,:;!?()'\"*/^~+= _-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to avoid overflow
    if len(text) > 250:
        text = text[:250] + "..."

    return text


def sanitize_slug(value: str, fallback: str = "object") -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value or fallback).strip("-")
    return sanitized or fallback


def persist_uploaded_image(image_base64: str, client_object_id: str, label: str) -> Tuple[str, str]:
    """Decode base64 image bytes, save to uploads/, and return (path, public_url)."""
    if not image_base64:
        raise ValueError("Missing imageBase64 payload.")

    header, encoded = (image_base64.split(",", 1) + [""])[:2] if "," in image_base64 else ("", image_base64)
    try:
        binary = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 image data.") from exc

    extension = "png"
    if "image/" in header:
        match = re.search(r"image/([a-zA-Z0-9.+-]+)", header)
        if match:
            extension = match.group(1)
    extension = extension.replace("jpeg", "jpg")

    safe_label = sanitize_slug(label)
    filename = f"{sanitize_slug(client_object_id)[:20]}_{safe_label}.{extension}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    with open(path, "wb") as output:
        output.write(binary)

    host_url = request.host_url.rstrip("/")
    return path, f"{host_url}/uploads/{filename}"


def encode_file_to_base64(image_path: str) -> str:
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError("Generated overlay image missing.")
    with open(image_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


# -------------------- PHASE 1 FUNCTION --------------------
def generate_equation_facts(obj):
    # --------------- TESTING HARD CODE : CHANGE OBJECT AND MODE --------------------
    object_name = obj.get("object", "bottle")
    lens_mode = obj.get("lens_mode", "math").lower()
    image_url = obj.get("image_url", "none")


    # Pick system prompt based on lens
    system_prompt = LENS_PROMPTS.get(lens_mode, LENS_PROMPTS["math"])

    if lens_mode == "physics":
        prompt = PHASE1_PROMPT_PHYSICS.format(object=object_name, image_url=image_url)
    else:
        prompt = PHASE1_PROMPT.format(object=object_name, image_url=image_url)

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        model="gpt-oss-120b",
        stream=False,
        max_completion_tokens=1024,
        temperature=0.2
    )

    text = response.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model did not return valid JSON.")

    return json.loads(match.group(0))


# -------------------- API ROUTES --------------------
@app.route(f"{API_PREFIX}/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route(f"{API_PREFIX}/objects", methods=["POST"])
def process_object_detection():
    payload = request.get_json(silent=True) or {}
    client_object_id = payload.get("clientObjectId") or str(uuid4())
    lens_mode = (payload.get("lensMode") or "math").lower()
    label = payload.get("label") or payload.get("object", "object")
    image_base64 = payload.get("imageBase64")

    try:
        image_path, image_url = persist_uploaded_image(image_base64, client_object_id, label)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        phase1 = generate_equation_facts({
            "object": label,
            "image_url": image_url,
            "lens_mode": lens_mode
        })

        for key in ["equation", "explanation", "guide"]:
            if key in phase1:
                phase1[key] = clean_text_for_prompt(phase1[key])

        explanation = clean_explanation(phase1.get("explanation", ""))
        guide = phase1.get("guide", "Create a neon diagram of the object.")

        overlay_path = generate_vertex_overlay(
            guide,
            label,
            image_path,
            lens_mode=lens_mode,
            explanation=explanation
        )

        if not overlay_path:
            raise RuntimeError("Vertex overlay generation returned no image.")

        annotated_b64 = encode_file_to_base64(overlay_path)

        return jsonify({
            "clientObjectId": client_object_id,
            "annotatedImageBase64": annotated_b64,
            "message": phase1.get("equation") or "Overlay generated.",
            "equation": phase1.get("equation"),
            "explanation": explanation,
            "lensMode": lens_mode
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"❌ Object processing error: {exc}")
        return jsonify({"error": "Failed to process object.", "details": str(exc)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    object_name = data.get("object", "cup").lower()
    lens_mode = data.get("lens_mode", "math").lower()  # 🔹 New field

    image_filename = f"{object_name}.png"
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)

    if not os.path.exists(image_path):
        return jsonify({"error": f"No image found for '{object_name}'"}), 404

    image_url = f"http://127.0.0.1:5000/uploads/{image_filename}"

    try:
        # === Phase 1 ===
        phase1 = generate_equation_facts({
            "object": object_name,
            "image_url": image_url,
            "lens_mode": lens_mode
        })

        for key in ["equation", "explanation", "guide"]:
            if key in phase1:
                phase1[key] = clean_text_for_prompt(phase1[key])

        explanation = clean_explanation(phase1.get("explanation", ""))
        guide = phase1.get("guide", "Draw the object with glowing skeleton lines.")

        # === Phase 2 ===
        output2_path = generate_vertex_overlay(
            guide, object_name, image_path, lens_mode=lens_mode, explanation=explanation
        )

        return jsonify({
            "phase1_output": phase1,
            "used_image": image_url,
            "lens_mode": lens_mode,
            "phase2_output_image": output2_path
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


# -------------------- STATIC ROUTE --------------------
@app.route("/uploads/<path:filename>")
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# -------------------- ENTRY --------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5050"))
    print(f"Uploads folder: {UPLOAD_FOLDER}")
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
