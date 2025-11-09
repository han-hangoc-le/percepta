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
MATHEMATICIAN_PROMPT = (
    "You are a mathematics lens assistant. Translate real-world objects into measurable geometry, "
    "proportions, and equations. Keep the tone precise and structured, focusing on formulas that "
    "describe dimensions or capacity."
)

PHYSICIST_PROMPT = (
    "You are a physics lens assistant. Describe the object's motion, energy, or forces. Highlight "
    "the governing physics equation (F = m·a, P = ρgh, etc.) and explain how it applies to the scene."
)

BIOLOGIST_PROMPT = (
    "You are a biology lens assistant. Reveal cellular, anatomical, or ecological insights. Reference "
    "biological processes and explain how structure supports function."
)

ARTIST_PROMPT = (
    "You are an artist lens assistant. Describe composition, palette, texture, and lighting cues to "
    "inspire creative overlays. Emphasize style and storytelling over scientific diagrams."
)

ECO_PROMPT = (
    "You are an eco lens assistant. Highlight an object's carbon impact, energy usage, or sustainability "
    "facts using concrete metrics. Keep explanations grounded in lifecycle thinking."
)

CULTURAL_PROMPT = (
    "You are a cultural lens assistant. Reveal the linguistic, regional, or historical origins of objects, "
    "including notable traditions or etymology."
)

LENS_PROMPTS = {
    "mathematician": MATHEMATICIAN_PROMPT,
    "physicist": PHYSICIST_PROMPT,
    "biologist": BIOLOGIST_PROMPT,
    "artist": ARTIST_PROMPT,
    "eco": ECO_PROMPT,
    "cultural": CULTURAL_PROMPT,
    "math": MATHEMATICIAN_PROMPT,
    "physics": PHYSICIST_PROMPT,
}

DEFAULT_LENS = "mathematician"
LENS_ALIASES = {
    "math": "mathematician",
    "mathematician": "mathematician",
    "physics": "physicist",
    "physicist": "physicist",
    "bio": "biologist",
    "biologist": "biologist",
    "artist": "artist",
    "eco": "eco",
    "sustainability": "eco",
    "environmental": "eco",
    "cultural": "cultural",
    "culture": "cultural",
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
4. "guide" – a fully specified rendering brief describing how to visualize the object as a neon-blueprint schematic.

In "guide", provide detailed bullet points that cover ALL of the following:
- Subject framing: describe the exact camera angle (e.g., isometric 30°, orthographic side profile) and which parts
  of the object must remain visible.
- Geometry markup: specify every line, arc, and measurement indicator to draw, including where dimension arrows start
  and end, which variables to label, and how to annotate derived shapes (cross-sections, exploded insets, etc.).
- Rendering style: define line weight hierarchy, glow color palette (primary/secondary neon hues), background tone,
  and any grid or dashed reference planes.
- Equation placement: state the exact position (above, below, floating panel), styling (font color, divider lines),
  and whether to include explanatory callouts for each variable.
- Extra data layers: mention optional overlays such as axis ticks, coordinate labels, volumetric fill gradients,
  or translucency settings to highlight interior geometry.

Example output:
{example}
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
4. "guide" — a step-by-step overlay specification with explicit instructions.

In "guide", include detailed bullet points for:
- Physical context: describe the scenario (static equilibrium, projectile motion, torque, fluid pressure) and what
  aspect of the object the user is observing.
- Vector design: for every force/velocity/acceleration vector, specify color, thickness, arrowhead size, start/end
  coordinates on the object, and whether it should pulse, taper, or include magnitude tick marks.
- Scalar fields or surfaces: explain how to draw planes, pressure gradients, or energy wells, including transparency
  levels and color ramps.
- Equation and legend layout: define the exact panel style, background opacity, iconography, and how to align variable
  callouts with the corresponding arrows.
- Interaction hints: mention any dynamic cues (e.g., animated ripple, dashed prediction path, rotational indicators)
  the model should visualize while keeping the real image visible.

Example output:
{example}
"""

PHASE1_PROMPT_BIOLOGY = """
You are generating structured data for a biology lens AI that overlays educational facts on real imagery.

Given the following information:
Object: {object}
Image (optional): {image_url}

Return a JSON object with exactly three keys:
1. "object" – the object's name.
2. "equation" – the core biological relationship, proportionality, or rate to highlight.
3. "explanation" – a concise description (2–4 sentences) that interprets the equation in the context of this object
   and mentions low-level anatomical or ecological cues (e.g., tissue types, diffusion, stomata behavior).

Do NOT include any other keys. Keep the tone instructive and grounded in observable biological features.

Example output:
{example}
"""

PHASE1_PROMPT_ART = """
You are generating structured data for an artist lens AI that overlays creative commentary on a live camera feed.

Given the following information:
Object: {object}
Image (optional): {image_url}

Return a JSON object with exactly three keys:
1. "object" – the object's name.
2. "equation" – describe the aesthetic recipe, palette shorthand, rhythm, or lighting ratio in a compact formula-like
   phrase (e.g., "Palette = (#hex, #hex) · Rhythm = repeating crescents").
3. "explanation" – a concise note (2–4 sentences) describing how that artistic principle applies to the observed object,
   referencing low-level cues such as brush direction, color temperature shifts, or composition lines.

Do NOT include any other keys. Keep wording vivid but focused on actionable visual insights.

Example output:
{example}
"""

PHASE1_PROMPT_ECO = """
You are generating structured data for an eco lens AI that overlays sustainability facts on top of real imagery.

Given the following information:
Object: {object}
Image (optional): {image_url}

Return a JSON object with exactly three keys:
1. "object" – the object's name.
2. "equation" – express a low-level sustainability relationship or carbon metric (e.g., "Annual CO₂ savings = baseline - efficient usage").
3. "explanation" – 2–4 sentences that quantify the impact, reference typical usage assumptions, and mention at least one concrete stat
   such as grams of CO₂, liters of water, recycled content percentage, or payback period.

Do NOT include any other keys. Treat the "equation" as a compact formula or data expression, not a sentence.

Example output:
{example}
"""

PHASE1_PROMPT_CULTURAL = """
You are generating structured data for a cultural lens AI that reveals linguistic or historical origins of objects.

Given the following information:
Object: {object}
Image (optional): {image_url}

Return a JSON object with exactly three keys:
1. "object" – the object's name.
2. "equation" – summarize the cultural data as a compact expression (e.g., "Origin = Edo Japan · Meaning = 'writing brush'").
3. "explanation" – 2–4 sentences describing the item's cultural roots, etymology, or traditional usage, referencing at least one region,
   community, or language detail.

Do NOT include any other keys. Keep tone respectful and factual.

Example output:
{example}
"""

MATHEMATICIAN_EXAMPLE = """
{
  "object": "bottle",
  "equation": "V = πr²h",
  "explanation": "Volume of a cylinder where r is radius and h is height.",
  "guide": "Step 1: Overlay a transparent cylinder matching the bottle silhouette and highlight the central axis. Step 2: Draw a radial arrow from the axis to the rim to mark r, attach a floating label, and add a semicircular arc referencing π. Step 3: Extend a vertical arrow along the axis for h with tick marks at the top and bottom surfaces. Step 4: Draw a dashed disk on the base to show πr² and annotate it with 'base area'. Step 5: Place the equation V = πr²h in a mini HUD panel under the bottle with bullet notes explaining each variable and an arrow linking the HUD to the respective geometry."
}

{
  "object": "pizza",
  "equation": "A = πr²",
  "explanation": "Area of a circular pizza crust.",
  "guide": "Step 1: Align a neon circular outline with the pizza. Step 2: Draw a radius arrow from the center to the crust edge, labeling it r and marking a right angle at the center. Step 3: Add a subtle polar grid inside the circle to visualize π integration, with opacity fading near the edge. Step 4: Cut out one slice using dashed lines and annotate the slice wedge as representing a differential area. Step 5: Place the equation A = πr² in a circular badge near the slice, with notes 'π handles circular sweep' and 'r² scales area', both with leader lines back to the grid and radius arrow."
}

{
  "object": "laptop",
  "equation": "d² = w² + h²",
  "explanation": "Screen diagonal relates orthogonal width and height via the Pythagorean theorem.",
  "guide": "Step 1: Outline the laptop display in perspective, adding corner anchors. Step 2: Draw width w along the bottom bezel and height h up the side, each with measurement arrows and tick marks. Step 3: Stretch a contrasting diagonal arrow from lower-left to upper-right and label it d. Step 4: Insert a glowing right-angle marker in the lower-left corner to tie w and h together. Step 5: Present the equation d² = w² + h² in a floating card that includes mini bullet text explaining how squaring w and h sums to the diagonal, with leader lines pointing to the respective arrows."
}

{
  "object": "chair",
  "equation": "Load per leg = W / 4",
  "explanation": "A four-legged chair ideally splits the sitter's weight evenly between legs.",
  "guide": "Step 1: Render the chair with a top-down inset showing the rectangular footprint. Step 2: Draw a single downward vector W centered on the seat. Step 3: Branch that vector into four equal arrows aligned with each leg, labeling them W/4. Step 4: Add a small equation card near one leg showing W ÷ 4 and connect it back to the main weight vector. Step 5: Include a balance note explaining that uneven floors shift the ratio, but the baseline calculation assumes symmetric contact."
}

{
  "object": "bowl",
  "equation": "C = 2πr",
  "explanation": "Circumference of the circular rim with radius r.",
  "guide": "Step 1: Trace the rim of the bowl with a glowing circle and mark its center. Step 2: Draw a radius arrow r from the center to the rim, labeling both ends. Step 3: Wrap a dashed measurement tape around the rim to represent circumference C and tag the tape with arrowheads to show continuity. Step 4: Add a mini arc annotation describing how the radius sweeps the full circle. Step 5: Place the equation C = 2πr beside the tape, with leader lines from 2π to the full circle and from r to the radius arrow."
}

{
  "object": "potted plant",
  "equation": "Soil volume = πr²h",
  "explanation": "Approximate cylindrical pot interior by radius r and soil height h.",
  "guide": "Step 1: Draw the planter with a cutaway revealing the soil column. Step 2: Highlight the circular base and add a radius arrow r from center to rim. Step 3: Mark soil depth h with a vertical arrow from the base to the surface. Step 4: Tint the soil region to emphasize the volume being measured. Step 5: Display πr²h in a HUD bubble with icons linking πr² to the base disk and h to the vertical arrow, reinforcing how the pot's geometry controls capacity."
}
"""

PHYSICIST_EXAMPLE = """{
  "object": "laptop",
  "equation": "P = V·I",
  "explanation": "Electrical power equals supply voltage times current draw.",
  "guide": "Step 1: Outline the laptop chassis in orthographic view and highlight the power jack. Step 2: Draw a voltage vector entering the device and label it V near the adapter port. Step 3: Add a current arrow I flowing toward the internal circuitry with a thin glowing path. Step 4: Annotate the battery pack and motherboard zones to show where the energy is consumed. Step 5: Place a HUD card with P = V·I, using leader lines from V to the adapter arrow and from I to the internal current path, with the final power value linked to the screen backlight."
}

{
  "object": "bottle",
  "equation": "P = ρgh",
  "explanation": "Hydrostatic pressure increases with fluid density, gravity, and depth.",
  "guide": "Step 1: Render the bottle upright with a transparent fluid interior. Step 2: Add a vertical depth arrow h measured from the surface to a sample point near the base. Step 3: Show density ρ as a floating tag attached to the fluid column and gravity g as a downward arrow outside the bottle. Step 4: Apply a color gradient inside the liquid that darkens toward the base to visualize increasing pressure. Step 5: Present P = ρgh in a panel beside the bottle with connectors from each symbol to the respective depth arrow, density tag, and gravity vector."
}

{
  "object": "chair",
  "equation": "σ = F / A",
  "explanation": "Mechanical stress equals applied force divided by contact area.",
  "guide": "Step 1: Draw the chair legs with a cutaway showing the area A where they meet the floor. Step 2: Add the weight force F as a downward vector centered on the seat. Step 3: Highlight the footprint patch on the floor and label it with area brackets. Step 4: Use a heat-map overlay along the legs to show how stress travels from the seat to the ground. Step 5: Place the formula σ = F / A in a HUD near the footprint, with arrows connecting F to the load vector and A to the highlighted contact patch."
}

{
  "object": "microwave oven",
  "equation": "E = h·f",
  "explanation": "Photon energy is Planck's constant times microwave frequency.",
  "guide": "Step 1: Depict the microwave door slightly transparent to reveal the cavity. Step 2: Draw standing-wave patterns inside the cavity, marking successive peaks with wavelength brackets. Step 3: Add an antenna icon where the magnetron injects radiation and label the emitted frequency f. Step 4: Place Planck's constant h in a side legend with a note linking it to quantum energy packets. Step 5: Display E = h·f in a floating card above the oven, using dotted leaders from f to the wave pattern and from h to the legend, emphasizing the resulting photon energy reaching the food."
}

{
  "object": "bicycle",
  "equation": "τ = r × F",
  "explanation": "Wheel torque equals crank radius times applied pedal force.",
  "guide": "Step 1: Show the bicycle from the side with the crank arm highlighted. Step 2: Draw a radius vector r from the crank center to the pedal. Step 3: Add the rider's foot force F as a downward arrow on the pedal. Step 4: Illustrate the resulting torque τ as a curved arrow about the crank spindle, feeding into the chainring. Step 5: Present τ = r × F next to the crankset, with color-coded leader lines linking r to the radius vector and F to the pedal force, while a note explains how the torque drives the chain."
}"""

BIOLOGIST_EXAMPLE = """
{
  "object": "person",
  "equation": "Heat loss ∝ surface_area × ΔT",
  "explanation": "Standing humans radiate more warmth when uncovered skin area is large and the air temperature drops, so limbs act as major heat-exchange fins."
}

{
  "object": "dog",
  "equation": "Panting rate ∝ metabolic_heat / airflow",
  "explanation": "Dogs evaporate saliva across the tongue to dump excess heat; more internal heat or less airflow forces faster panting to keep body temperature stable."
}

{
  "object": "cat",
  "equation": "Jump impulse = m × Δv",
  "explanation": "When a cat launches from a sofa, leg muscles deliver an impulse equal to mass times the change in velocity, letting it reach shelves without sprinting laps."
}

{
  "object": "bird",
  "equation": "Lift ≈ ½ρv²S C_L",
  "explanation": "Even perched birds reveal wing architecture: the same surface area S and camber that keeps them aloft also dictates the minimum airflow needed for takeoff."
}

{
  "object": "cow",
  "equation": "Rumen gas ∝ fiber_intake × microbiome_activity",
  "explanation": "Chewing cud loads the rumen with cellulose; resident microbes ferment it and release methane, so diets richer in fiber drive higher gas production."
}

{
  "object": "potted plant",
  "equation": "Transpiration rate = stomatal_conductance × VPD",
  "explanation": "Houseplants pull water upward when stomata stay open and the vapor-pressure deficit (dryness gap) is high, moving moisture from soil to room air even indoors."
}
"""

ARTIST_EXAMPLE = """
{
  "object": "bottle",
  "equation": "Palette = (#0f172a, #38bdf8, #e0f2fe) · Highlight = rim specular",
  "explanation": "Deep navy surroundings make the aqua liquid glow; pick out the rim with a cool highlight to separate glass from background."
}

{
  "object": "vase",
  "equation": "Composition = centered vertical · Texture = matte vs gloss",
  "explanation": "Keep the vase upright on the midline and contrast a matte wall with glossy ceramic reflections to lead the eye upward."
}

{
  "object": "bicycle",
  "equation": "Rule-of-thirds anchors = saddle & front hub",
  "explanation": "Place the saddle on the upper-left intersection and the front hub on the lower-right to stretch the frame diagonally across the canvas."
}

{
  "object": "dining table",
  "equation": "Rhythm = repeating rectangles",
  "explanation": "Plates, placemats, and chair backs echo rectangular beats; align them to create a steady left-to-right cadence."
}

{
  "object": "book",
  "equation": "Contrast = warm cover vs cool shadow",
  "explanation": "Prop the book open near a window so the cover catches warm light while pages fall into blue-gray shade, emphasizing depth."
}

{
  "object": "couch",
  "equation": "Palette = (#f4f1ed, #c08457, #1f2933) · Balance = asymmetrical pillows",
  "explanation": "Use a light base fabric, caramel leather accents, and charcoal wall, then offset the frame with pillows grouped on one side for casual balance."
}
"""
ECO_EXAMPLE = """{
  "object": "LED bulb",
  "equation": "Annual CO₂ savings = baseline_incandescent - LED_usage",
  "explanation": "Swapping a 60 W incandescent for a 9 W LED saves roughly 45 kg of CO₂ per year assuming three hours of daily use on an average US grid."
}

{
  "object": "steel bottle",
  "equation": "Refill payback ≈ (steel_embodied - single_use) / per_fill_savings",
  "explanation": "After about 30 refills the stainless bottle repays its higher manufacturing footprint compared to buying a new disposable plastic bottle each trip."
}

{
  "object": "cotton tote",
  "equation": "Neutral trips ≈ manufacturing_emissions / plastic_bags_avoided",
  "explanation": "A heavy cotton tote may need over 100 grocery runs before its footprint beats thin plastic bags, so longevity and frequent reuse matter."
}

{
  "object": "dishwasher cycle",
  "equation": "Water_saved = handwash_volume - machine_volume",
  "explanation": "An Energy Star dishwasher uses about 3–4 gallons per load, whereas handwashing can exceed 15 gallons, so full loads cut both water and energy per plate."
}"""

CULTURAL_EXAMPLE = """{
  "object": "ceramic tea bowl",
  "equation": "Origin = Momoyama Japan · Meaning = wabi-sabi",
  "explanation": "Raku bowls emerged in 16th-century Kyoto kilns; their asymmetry embodies wabi-sabi, the celebration of imperfection central to the tea ceremony."
}

{
  "object": "mortar and pestle",
  "equation": "Etymology = Latin mortarium · Regions = Mediterranean & West Africa",
  "explanation": "The Roman mortarium described a grinding vessel; modern cousins like the Ghanaian asanka and Mexican molcajete show how the tool migrated across cuisines."
}

{
  "object": "paisley scarf",
  "equation": "Motif = Persian boteh · Route = Kashmir → Paisley",
  "explanation": "The teardrop boteh symbol traveled from Sassanid Persia to Kashmiri shawls, then to Paisley, Scotland, whose mills industrialized the pattern in the 1800s."
}

{
  "object": "origami crane",
  "equation": "Symbol = senbazuru wishes · Era = Edo Japan",
  "explanation": "Folding a thousand cranes (senbazuru) became a Japanese tradition for recovery and peace, especially after the story of Sadako Sasaki."
}"""

PHASE1_CONFIG = {
    "mathematician": {"template": PHASE1_PROMPT, "example": MATHEMATICIAN_EXAMPLE},
    "physicist": {"template": PHASE1_PROMPT_PHYSICS, "example": PHYSICIST_EXAMPLE},
    "biologist": {"template": PHASE1_PROMPT_BIOLOGY, "example": BIOLOGIST_EXAMPLE},
    "artist": {"template": PHASE1_PROMPT_ART, "example": ARTIST_EXAMPLE},
    "eco": {"template": PHASE1_PROMPT_ECO, "example": ECO_EXAMPLE},
    "cultural": {"template": PHASE1_PROMPT_CULTURAL, "example": CULTURAL_EXAMPLE},
}


def resolve_lens_mode(raw_mode: str | None) -> str:
    if raw_mode:
        normalized = raw_mode.strip().lower()
        candidate = LENS_ALIASES.get(normalized, normalized)
        if candidate in PHASE1_CONFIG:
            return candidate
    return DEFAULT_LENS


def build_phase1_prompt(lens_mode: str, object_name: str, image_url: str | None) -> str:
    config = PHASE1_CONFIG.get(lens_mode, PHASE1_CONFIG[DEFAULT_LENS])
    safe_image = image_url or "none"
    return config["template"].format(object=object_name, image_url=safe_image, example=config["example"])

# -------------------- CLEANER --------------------
def _ensure_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_ensure_text(item) for item in value)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return "" if value is None else str(value)


def clean_text_for_prompt(text) -> str:
    """Normalize special math or formatting symbols for model safety."""
    text = _ensure_text(text)
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

def clean_explanation(text) -> str:
    """Cleans and normalizes physics explanations for visual rendering."""
    text = _ensure_text(text)
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
    object_name = (obj.get("object") or "").strip()
    if not object_name:
        raise ValueError("Object name is required for lens generation.")

    lens_mode = resolve_lens_mode(obj.get("lens_mode"))
    image_url = obj.get("image_url") or ""

    system_prompt = LENS_PROMPTS.get(lens_mode, LENS_PROMPTS[DEFAULT_LENS])
    prompt = build_phase1_prompt(lens_mode, object_name, image_url)

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


@app.route(f"{API_PREFIX}/facts", methods=["POST"])
def process_object_facts():
    payload = request.get_json(silent=True) or {}
    client_object_id = payload.get("clientObjectId") or str(uuid4())
    lens_mode = resolve_lens_mode(payload.get("lensMode"))
    label = (payload.get("label") or payload.get("object") or "").strip() or "object"
    image_base64 = payload.get("imageBase64")

    if not image_base64:
        return jsonify({"error": "imageBase64 is required."}), 400

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

        if "equation" in phase1:
            phase1["equation"] = clean_text_for_prompt(phase1["equation"])
        explanation = clean_explanation(phase1.get("explanation", ""))

        return jsonify({
            "clientObjectId": client_object_id,
            "lensMode": lens_mode,
            "label": label,
            "equation": phase1.get("equation"),
            "explanation": explanation,
            "imageUrl": image_url
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"❌ Facts generation error: {exc}")
        return jsonify({"error": "Failed to generate facts.", "details": str(exc)}), 500


@app.route(f"{API_PREFIX}/objects", methods=["POST"])
def process_object_detection():
    payload = request.get_json(silent=True) or {}
    client_object_id = payload.get("clientObjectId") or str(uuid4())
    lens_mode = resolve_lens_mode(payload.get("lensMode"))
    label = (payload.get("label") or payload.get("object") or "").strip() or "object"
    image_base64 = payload.get("imageBase64")

    if not image_base64:
        return jsonify({"error": "imageBase64 is required."}), 400

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
