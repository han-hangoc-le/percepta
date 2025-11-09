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

MATHEMATICIAN_EXAMPLE = """{
  "object": "bottle",
  "equation": "V = πr²h",
  "explanation": "Volume of a cylinder where r is radius and h is height.",
  "guide": "Step 1: Overlay a transparent cylinder matching the bottle silhouette and highlight the central axis. Step 2: Draw a radial arrow from the axis to the rim to mark r, attach a floating label, and add a semicircular arc referencing π. Step 3: Extend a vertical arrow along the axis for h with tick marks at the top and bottom surfaces. Step 4: Draw a dashed disk on the base to show πr² and annotate it with 'base area'. Step 5: Place the equation V = πr²h in a mini HUD panel under the bottle with bullet notes explaining each variable and an arrow linking the HUD to the respective geometry."
}

{
  "object": "spiral notebook",
  "equation": "Area = w x h",
  "explanation": "Surface area of the notebook cover treated as a rectangle.",
  "guide": "Step 1: Flatten the notebook cover into an orthographic rectangle with a subtle grid overlay. Step 2: Use a bold horizontal arrow to mark width w along the lower edge and a vertical arrow for height h on the right edge; both arrows get labeled tabs with tiny scale ticks. Step 3: Shade the rectangle with unit squares to imply multiplication of w and h. Step 4: Draw a floating equation card at the corner containing Area = w × h plus short reminders 'w = horizontal span' and 'h = vertical span'. Step 5: Add faint leader lines from the card back to the measurement arrows to reinforce the relationship."
}

{
  "object": "shipping box",
  "equation": "Surface area = 2(lw + lh + wh)",
  "explanation": "Total exposed cardboard on a rectangular prism.",
  "guide": "Step 1: Sketch the box in isometric wireframe and lightly tint each face. Step 2: Use color-coded arrows to tag length l, width w, and height h on the box edges. Step 3: Show an exploded net diagram next to the box with corresponding colored panels representing lw, lh, and wh. Step 4: Draw braces summarizing lw + lh + wh and another brace labelled '×2' to illustrate the paired faces. Step 5: Include the full formula in a HUD with bullet annotations: 'lw = front/back', 'lh = sides', 'wh = top/bottom', each pointing to the respective net panels."
}

{
  "object": "pizza",
  "equation": "A = πr²",
  "explanation": "Area of a circular pizza crust.",
  "guide": "Step 1: Align a neon circular outline with the pizza. Step 2: Draw a radius arrow from the center to the crust edge, labeling it r and marking a right angle at the center. Step 3: Add a subtle polar grid inside the circle to visualize π integration, with opacity fading near the edge. Step 4: Cut out one slice using dashed lines and annotate the slice wedge as representing a differential area. Step 5: Place the equation A = πr² in a circular badge near the slice, with notes 'π handles circular sweep' and 'r² scales area', both with leader lines back to the grid and radius arrow."
}

{
  "object": "laptop screen",
  "equation": "Diagonal² = w² + h²",
  "explanation": "Pythagorean relationship of the display dimensions.",
  "guide": "Step 1: Outline the laptop display in perspective, adding corner anchors. Step 2: Draw width w along the bottom bezel and height h up the side, each with measurement arrows and tick marks. Step 3: Stretch a contrasting diagonal arrow from lower-left to upper-right and label it d. Step 4: Insert a glowing right-angle marker in the lower-left corner to tie w and h together. Step 5: Present the equation d² = w² + h² in a floating card that includes mini bullet text explaining how squaring w and h sums to the diagonal, with leader lines pointing to the respective arrows."
}

{
  "object": "traffic cone",
  "equation": "Volume = (1/3)πr²h",
  "explanation": "Standard cone volume formula.",
  "guide": "Step 1: Render the cone as a transparent shell showing the central axis. Step 2: Project radius r from the axis to the base rim and label it using a curved arrow that hugs the base circle. Step 3: Draw height h along the axis with arrows at both ends and note the ground reference plane. Step 4: Slice out a narrow wedge and display it next to the cone to illustrate rotational symmetry; annotate (1/3) as 'one-third of cylinder'. Step 5: Place the full formula beside the wedge with bullet notes mapping πr² to the base disk and h to the axis arrow."
}

{
  "object": "bookshelf",
  "equation": "Capacity = shelf_width × shelf_depth × shelf_count",
  "explanation": "Approximate storage volume using rectangular prism layers.",
  "guide": "Step 1: Draw the bookshelf in perspective with each shelf rendered semi-transparent. Step 2: Use a horizontal arrow for shelf_width across the front edge and a perpendicular arrow for shelf_depth pointing into the shelf. Step 3: Stack numbered markers along one side to indicate shelf_count, connecting them with a dashed vertical axis. Step 4: Fill the shelf volume with a faint voxel grid to show discrete storage cells. Step 5: Display the capacity equation in a HUD with checkboxes linking each term to the visual measurement arrows and shelf stack."
}

{
  "object": "picture frame",
  "equation": "Perimeter = 2(w + h)",
  "explanation": "Total trim length around a rectangle.",
  "guide": "Step 1: Trace a glowing path around the frame border to represent the perimeter loop. Step 2: Label width w along the top and bottom edges with directional arrows and label height h on the sides. Step 3: Place small corner markers to emphasize the path turning points. Step 4: Unfold the perimeter into a straight line graphic showing two segments of w and two of h, with the bracket '2(w + h)' hovering above. Step 5: Add callouts describing how to cut molding pieces to match these lengths, linking them back to the edges."
}

{
  "object": "cylindrical stool",
  "equation": "Surface area = 2πr² + 2πrh",
  "explanation": "Top and bottom circles plus side wrap.",
  "guide": "Step 1: Illustrate the stool and overlay top and bottom circular outlines with labels 'πr² each'. Step 2: Unwrap the lateral surface into a rectangle adjacent to the stool, marking its width as circumference 2πr and its height as h. Step 3: Color-code the circles and the rectangle to match the terms in the formula. Step 4: Add arrows showing how the lateral rectangle wraps back around the stool surface. Step 5: Display the full equation with braces grouping the circular and lateral components, plus annotations referencing the color codes."
}

{
  "object": "planter pot",
  "equation": "V = (πh/3)(R² + Rr + r²)",
  "explanation": "Volume of a truncated cone with outer radius R and inner radius r.",
  "guide": "Step 1: Cut the pot vertically to reveal interior and exterior walls. Step 2: Mark the upper radius R with a horizontal arrow across the rim and the lower radius r across the base interior. Step 3: Draw height h along the centerline and note the soil level. Step 4: Add three stacked circular slices (top, mean, bottom) to illustrate the R², Rr, and r² terms, connecting them with dashed guides. Step 5: Place the frustum formula adjacent to the slices with bullet notes mapping each term to the visual slices and emphasizing the (πh/3) scalar."
}"""

PHYSICIST_EXAMPLE = """{
  "object": "rolling ball",
  "equation": "F = m · a",
  "explanation": "Net force equals mass times acceleration.",
  "guide": "Step 1: Trace the ball with a translucent outline aligned to the motion direction. Step 2: Draw a thick red force vector pointing forward; annotate its base with 'F'. Step 3: Add a smaller yellow acceleration vector parallel to the force arrow but slightly offset; label it 'a'. Step 4: Place a floating mass tag 'm' near the ball and connect it to the object's center of mass. Step 5: Show the equation F = m·a in a rectangular HUD above the ball with arrows linking each symbol to the corresponding visual element."
}

{
  "object": "swinging pendulum",
  "equation": "T = 2π√(L/g)",
  "explanation": "Period of a simple pendulum with length L.",
  "guide": "Step 1: Render the pendulum at maximum displacement; overlay the circular arc path with tick marks. Step 2: Draw the string length L as a highlighted segment from pivot to bob and label it. Step 3: Add vectors for tension (along the string) and gravity (vertical downward) plus the tangential restoring component. Step 4: Include a curved arrow showing direction of swing and annotate it with period T. Step 5: Display the period formula in a HUD with mini notes 'L controls arc length' and 'g defines gravitational restoring strength'."
}

{
  "object": "sliding block",
  "equation": "F_friction = μN",
  "explanation": "Frictional force equals coefficient times normal force.",
  "guide": "Step 1: Outline the block on the surface and draw a free-body diagram: gravity vector downward, normal vector upward, applied force to the right, friction to the left. Step 2: Color-code friction in red and normal force in blue, adding labels 'F_f' and 'N'. Step 3: Insert a coefficient μ tag near the contact surface with dotted arrows referencing microscopic contact. Step 4: Add a HUD summarizing 'F_f = μN' and include sliders for μ and N to reinforce their multiplicative relationship. Step 5: Provide a ground plane grid showing direction of motion and frictional opposition."
}

{
  "object": "seesaw",
  "equation": "τ = r × F",
  "explanation": "Torque equals lever arm times force.",
  "guide": "Step 1: Draw the seesaw with pivot highlighted and two masses on either side. Step 2: From the pivot, draw lever arm vectors r1 and r2 pointing to each contact point. Step 3: Add downward force vectors corresponding to each child, labeled F1 and F2. Step 4: Illustrate torques as curved arrows around the pivot, with direction indicating rotation tendency. Step 5: Present τ = r × F near the pivot with annotations 'r = distance from fulcrum' and 'F = applied load', linking them to the vectors."
}

{
  "object": "bicycle wheel",
  "equation": "a_c = v² / r",
  "explanation": "Centripetal acceleration for circular motion.",
  "guide": "Step 1: Highlight the wheel rim with luminous segments indicating circular path. Step 2: Draw tangent velocity vectors v at several evenly spaced points along the rim. Step 3: Add inward acceleration arrows a_c pointing toward the hub, differentiating them by color. Step 4: Overlay a dashed circle showing the path radius r and label it near the hub. Step 5: Show the equation a_c = v² / r in a HUD with bullet notes describing how increasing v or decreasing r raises inward acceleration."
}

{
  "object": "desk lamp hinge",
  "equation": "τ = Iα",
  "explanation": "Rotational analogue of Newton's second law.",
  "guide": "Step 1: Depict the lamp arm as rigid segments with the hinge highlighted as the pivot. Step 2: Add a curved arrow around the hinge showing angular acceleration α, labeled at the arc center. Step 3: Represent the moment of inertia I as a block diagram near the arm, connected with dotted lines to the mass distribution. Step 4: Include a torque vector τ at the pivot pointing along the axis of rotation. Step 5: Display τ = Iα beside the hinge with annotations tying α to the curved arrow and I to the mass distribution overlay."
}

{
  "object": "coffee steam plume",
  "equation": "Q = m c ΔT",
  "explanation": "Heat transfer in the rising vapor.",
  "guide": "Step 1: Outline the cup and render the steam as volumetric ribbons rising upward. Step 2: Use gradient arrows traveling through the plume to indicate heat flow direction; label them Q. Step 3: Tag a column of steam with mass m and annotate a side panel showing c (specific heat). Step 4: Place temperature markers near the cup surface and the plume top to show ΔT. Step 5: Insert the equation Q = m c ΔT into a floating panel with bullet text explaining each variable, and connect the panel to the corresponding visual markers."
}

{
  "object": "toy car on ramp",
  "equation": "v² = u² + 2as",
  "explanation": "Kinematics equation for constant acceleration.",
  "guide": "Step 1: Draw the ramp with a distance axis along the incline labeled s and evenly spaced markers. Step 2: Place a toy car at the start with velocity vector u and near the bottom with velocity vector v. Step 3: Draw an acceleration component arrow along the slope pointing downhill. Step 4: Add a timeline strip or progress bar at the base to match positions to time. Step 5: Show the equation v² = u² + 2as above the ramp with color-coded links to the initial velocity, acceleration, and displacement annotations."
}

{
  "object": "dam spillway",
  "equation": "Pressure = ρgh",
  "explanation": "Hydrostatic pressure at depth h.",
  "guide": "Step 1: Render a vertical cross-section of the dam wall and the adjacent water column. Step 2: Apply a blue-to-purple gradient across the water, increasing with depth to visualize pressure. Step 3: Draw a depth arrow h from the surface down to a sample point and label it. Step 4: Include density ρ and gravity g as icons in a side legend with arrows pointing to the water mass and downward vector, respectively. Step 5: Display P = ρgh in a HUD and link each variable to its visual cue with leader lines."
}

{
  "object": "speaker cone",
  "equation": "v = λf",
  "explanation": "Wave speed equals wavelength times frequency.",
  "guide": "Step 1: Outline the speaker cone and show it vibrating forward. Step 2: Visualize concentric wavefronts radiating outward with alternating light/dark rings. Step 3: Mark the distance between successive peaks as wavelength λ with a double-headed arrow and annotate it. Step 4: Include a frequency indicator f in a HUD, perhaps a bar showing beats per second, tied to the speaker motion. Step 5: Draw a velocity arrow v pointing outward along the wave normal and place the equation v = λf beside it with explanatory bullet points."
}"""

BIOLOGIST_EXAMPLE = """{
  "object": "leaf",
  "equation": "Transpiration rate = Stomatal conductance × Vapor pressure deficit",
  "explanation": "Water escapes through stomata on the underside of the blade. The flux scales with both how wide the pores open and how dry the surrounding air is."
}

{
  "object": "tree trunk",
  "equation": "Sap flow = πr² × upward_velocity",
  "explanation": "The conducting xylem ring provides cross-sectional area πr², so thicker trunks move more sap when upward velocity from transpiration pull stays the same."
}

{
  "object": "flower stamen",
  "equation": "Pollen release rate = aperture_area × wind_speed",
  "explanation": "Broader anther openings expose more pollen grains to airflow. Even gentle breezes loft grains efficiently when the aperture area is large."
}

{
  "object": "mushroom gill",
  "equation": "Spore count = gill_area × spore_density",
  "explanation": "Sample a gill, count spores per square millimeter, then multiply by total gill area to estimate nightly spore rain."
}

{
  "object": "coral branch",
  "equation": "Photosynthesis ∝ light_intensity × chlorophyll_fraction",
  "explanation": "Polyps packed with algae convert light into sugar efficiently. Bleached tissue with low chlorophyll yields far less energy."
}

{
  "object": "bee wing",
  "equation": "Wingbeat frequency = airflow_velocity / stroke_amplitude",
  "explanation": "Smaller stroke arcs demand higher beat frequency to maintain lift for the same airflow velocity when hovering near flowers."
}

{
  "object": "human hand",
  "equation": "Blood flow = ΔP / vascular_resistance",
  "explanation": "A pressure drop from wrist to fingertips pushes blood through capillaries. Cold-induced vasoconstriction raises resistance, lowering flow even if ΔP stays constant."
}

{
  "object": "butterfly wing",
  "equation": "Scale overlap ratio = scale_width / scale_pitch",
  "explanation": "Large width relative to pitch packs more scales into each row, stacking reflective plates that intensify iridescent shimmer."
}

{
  "object": "snail shell",
  "equation": "Radius = initial_radius × e^{kθ}",
  "explanation": "The shell follows a logarithmic spiral: every increment in angle θ expands radius exponentially with growth constant k, mirroring mantle secretion."
}

{
  "object": "seedling root",
  "equation": "Osmotic influx = k × (Ψ_soil - Ψ_root)",
  "explanation": "Water flows into root hairs when soil water potential exceeds the root interior. Dry soil narrows the potential gap and slows the inflow."
}"""

ARTIST_EXAMPLE = """{
  "object": "coffee cup",
  "equation": "Palette = (#f5e9db, #c28455, #2f1b0c) · Composition = 2:1 negative space ratio",
  "explanation": "Warm ceramics pop against a cool tabletop when you give the frame breathing room.",
}

{
  "object": "street lamp",
  "equation": "Palette = (#0f172a, #fbbf24, #ecfccb)",
  "explanation": "Midnight blues contrasting with sodium glow and soft foliage accents.",
}

{
  "object": "window plant",
  "equation": "Composition = diagonal flow",
  "explanation": "Leaves guide the viewer from lower-left to upper-right.",
}

{
  "object": "bowl of fruit",
  "equation": "Palette = (#f97316, #facc15, #fef9c3) · Rhythm = repeating crescents",
  "explanation": "Warm triadic palette with echoing curved shapes.",
}

{
  "object": "bicycle",
  "equation": "Rule-of-thirds anchors = saddle & front hub",
  "explanation": "Place critical features on grid intersections.",
}

{
  "object": "book stack",
  "equation": "Contrast = cool shadows vs warm covers",
  "explanation": "Push chroma difference to create depth.",
}

{
  "object": "desk workspace",
  "equation": "Lighting = key + rim",
  "explanation": "Two-source lighting adds dimensionality.",
}

{
  "object": "water glass",
  "equation": "Palette = monochrome teal",
  "explanation": "Single hue with value shifts emphasizes transparency.",
}

{
  "object": "earphones",
  "equation": "Composition ratio = 3:1 cable to buds",
  "explanation": "Use cable curves as leading lines.",
}

{
  "object": "tea kettle",
  "equation": "Palette = (#1c1917, #d97706, #fef3c7) · Texture = brushed metal",
  "explanation": "Dark body with gold highlights and soft specular bloom.",
}"""

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
