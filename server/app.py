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

LENS_PROMPTS = {
    "mathematician": MATHEMATICIAN_PROMPT,
    "physicist": PHYSICIST_PROMPT,
    "biologist": BIOLOGIST_PROMPT,
    "artist": ARTIST_PROMPT,
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
You are generating structured data for a biology lens AI that overlays educational diagrams on real imagery.

Given the following information:
Object: {object}
Image (optional): {image_url}

Return a JSON object with exactly four keys:
1. "object" – the object's name.
2. "equation" – the core biological relationship or rate to highlight.
3. "explanation" – a short description of what the process means and why it matters for this object.
4. "guide" – detailed instructions for visualizing anatomical layers, flows, or labels in a translucent overlay.

In "guide", include bullet points that specify:
- Biological focus: name the exact tissue, organelle, or ecological interaction being highlighted, plus which direction
  the viewer is looking from (cross-section, macro close-up, etc.).
- Layer breakdown: describe each translucent layer (outer skin, vascular bundle, cell interior), its color/opacity, and
  how to offset or clip it to fit the real object silhouette.
- Flow/pathway rendering: detail the path of fluids, nutrients, or signals with arrow styles, particle counts, and
  animation hints (e.g., pulsing cyan droplets, gradient arrows fading along the direction of flow).
- Labeling scheme: list every label with coordinates, connector-line style, and icon badges (e.g., mitochondria icon,
  leaf stomata marker). Include guidance for variable legends or mini info cards.
- Environmental context: if relevant, mention background cues such as humidity halos, soil cross-sections, or light
  direction indicators that help explain the biological process.

Example output:
{example}
"""

PHASE1_PROMPT_ART = """
You are generating structured data for an artist lens AI that overlays creative guidance on a live camera feed.

Given the following information:
Object: {object}
Image (optional): {image_url}

Return a JSON object with exactly four keys:
1. "object" – the object's name.
2. "equation" – describe the aesthetic recipe (palette, composition ratio, rhythm).
3. "explanation" – briefly explain how the artistic principle applies to this object.
4. "guide" – detailed prompts covering brush style, lighting hints, or typography placement.

In "guide", include structured bullet points for:
- Palette strategy: list at least three hex values with roles (base fill, highlights, accent strokes) plus blending
  instructions (e.g., soft gradient, stippled texture).
- Composition geometry: specify the exact overlay guides (rule-of-thirds grid, golden spiral, leading lines) and how
  they align with the object. Mention line weights, glow, and opacity.
- Brushwork & texture: describe the stroke type (ink line, oil smear, watercolor wash), spacing, motion direction, and
  whether strokes should trail off or loop around the object.
- Typography & annotations: note any captions, font styles, or iconography, including placement relative to the object
  and whether they should animate or fade in.
- Lighting cues: indicate rim-light colors, shadow exaggerations, or particle effects that help the scene feel painted
  while preserving the underlying camera feed.

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
  "equation": "Area = w × h",
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
  "explanation": "Water exits the leaf via stomata; the rate depends on how open they are and the humidity difference.",
  "guide": "Step 1: Overlay a semi-transparent cross-section of the leaf showing epidermis, mesophyll, and vein network. Step 2: Highlight stomata along the underside with pulsing cyan markers and label conductance near each pore. Step 3: Draw upward arrows representing vapor flux; color them according to magnitude to visualize vapor pressure deficit. Step 4: Add a side legend tying arrow color thickness to stomatal conductance values. Step 5: Place the equation in a floating panel with bullet notes describing how each variable maps to the highlighted structures."
}

{
  "object": "tree trunk",
  "equation": "Flow = πr²v",
  "explanation": "Sap flux approximated by cross-sectional area times velocity.",
  "guide": "Step 1: Cut the trunk horizontally and expose concentric rings; color-code xylem versus phloem. Step 2: Mark radius r from center to active xylem ring with an arrow. Step 3: Shade the cross-sectional area πr² to show the conducting region. Step 4: Animate sap droplets moving upward in a side view, labeling velocity v along the conduit paths. Step 5: Present Flow = πr²v near the slice with notes linking πr² to the shaded area and v to the droplet velocity arrows."
}

{
  "object": "flower stamen",
  "equation": "Pollen release rate = aperture_area × wind_speed",
  "explanation": "Simple mass flow relation for low-wind dispersal.",
  "guide": "Step 1: Zoom into the stamen tip and render the anther opening with a glowing outline. Step 2: Highlight the aperture area with a semi-transparent disk and annotate its diameter. Step 3: Show pollen grains leaving the aperture as small luminous particles, each with motion trails. Step 4: Draw wind vectors sweeping past the anther, labeling their speed. Step 5: Place the rate equation in a HUD box with callouts linking aperture area to the highlighted disk and wind speed to the vector arrows."
}

{
  "object": "mushroom cap",
  "equation": "Spore count = density × gill_area",
  "explanation": "Estimate of spores produced per gill surface.",
  "guide": "Step 1: Show the underside of the mushroom with gills rendered as alternately shaded ridges. Step 2: Overlay measurement rectangles along representative gill segments to denote sampled area. Step 3: Display spore density numbers above the rectangles and connect them to a scale bar. Step 4: Annotate total gill area with contour lines wrapping the entire cap underside. Step 5: Present the equation with arrows tying 'density' to the sample rectangles and 'gill_area' to the full contour."
}

{
  "object": "coral branch",
  "equation": "Photosynthesis rate ∝ light_intensity × chlorophyll_fraction",
  "explanation": "Shallow coral productivity depends on light capture.",
  "guide": "Step 1: Render a coral branch with polyps outlined; tint chlorophyll-rich tissues with a neon overlay and label 'chlorophyll fraction'. Step 2: Draw downward light beams entering the water, annotating their intensity with numeric values or gradient bars. Step 3: Add oxygen bubbles emanating from polyps to imply photosynthetic output. Step 4: Include a HUD showing a multiplier between light intensity and chlorophyll fraction. Step 5: Connect the HUD back to the beams and chlorophyll overlay using leader lines."
}

{
  "object": "bee wing",
  "equation": "Wingbeat frequency = airflow_velocity / stroke_amplitude",
  "explanation": "Approximate relation tying motion to airflow.",
  "guide": "Step 1: Illustrate the wing as layered translucent membranes with keyed nodes for muscle attachment. Step 2: Draw sweeping arcs that show stroke amplitude, labeling the extremes. Step 3: Overlay airflow streamlines moving past the wing, color-coded for velocity. Step 4: Place a frequency indicator (beats/sec) near the thorax and connect it to the arcs. Step 5: Show the equation panel explaining how airflow velocity feeds into the numerator and stroke amplitude into the denominator."
}

{
  "object": "human hand",
  "equation": "Blood flow = ΔP / vascular_resistance",
  "explanation": "Ohm's law analogy for circulation.",
  "guide": "Step 1: Overlay a vascular map on the hand, using red for arteries and blue for veins. Step 2: Indicate pressure drop ΔP between wrist and fingertips using gauge icons. Step 3: Draw directional arrows along the vessels to show flow direction and thickness proportional to rate. Step 4: Annotate vascular resistance near constriction points or capillary beds. Step 5: Place the equation in a HUD, connecting ΔP to the pressure gauges and resistance to highlighted vessel segments."
}

{
  "object": "butterfly wing",
  "equation": "Scale overlap ratio = scale_width / scale_pitch",
  "explanation": "Geometric relation determining color shimmer.",
  "guide": "Step 1: Zoom into the wing surface and depict parallel rows of scales with translucent overlays. Step 2: Use micrometer-style arrows to measure individual scale width and the spacing (pitch) between successive scales. Step 3: Draw a ratio bar representing width/pitch and color-code it. Step 4: Add shimmer arrows showing how overlap affects interference patterns. Step 5: Show the equation panel with miniature diagrams illustrating numerator (width) and denominator (pitch)."
}

{
  "object": "snail shell",
  "equation": "Growth = initial_radius × e^{kθ}",
  "explanation": "Logarithmic spiral describing shell expansion.",
  "guide": "Step 1: Outline the shell spiral and overlay contour lines for successive growth phases. Step 2: Draw an angle θ measured at the spiral center and annotate it with an arc arrow. Step 3: Show the initial radius and current radius using radial arrows from the center. Step 4: Place a mini graph illustrating exponential growth with parameter k, referencing mantle tissue responsible for secretion. Step 5: Display the equation with arrows pointing to the radius measurement and the angle arc."
}

{
  "object": "seedling root",
  "equation": "Osmotic influx = k (Ψ_soil - Ψ_root)",
  "explanation": "Water uptake driven by water potential difference.",
  "guide": "Step 1: Cross-section the soil and root zone, shading soil layers with moisture gradients. Step 2: Highlight root hairs with glowing tips and show water droplets moving inward. Step 3: Place Ψ_soil and Ψ_root gauges on either side of the membrane and connect them with a ΔΨ bar. Step 4: Draw arrows whose thickness reflects osmotic influx, labeled with constant k. Step 5: Present the equation in a HUD linking each term to the gauges and arrows, explaining the direction of water movement."
}"""

ARTIST_EXAMPLE = """{
  "object": "coffee cup",
  "equation": "Palette = (#f5e9db, #c28455, #2f1b0c) · Composition = 2:1 negative space ratio",
  "explanation": "Warm ceramics pop against a cool tabletop when you give the frame breathing room.",
  "guide": "Step 1: Trace cup and saucer contours with double strokes (inner #f5e9db, outer #2f1b0c) to define the silhouette. Step 2: Overlay a golden spiral guide from saucer edge to rim highlight, labeling start/end nodes and intersection with the focal highlight. Step 3: Place palette swatches near the handle with arrows indicating where each tone appears (base fill, mid-tone, accent). Step 4: Paint a teal rim-light strip opposite the key light and annotate it as 'cool complement'. Step 5: Draw bounding boxes showing the 2:1 negative space ratio and include a caption explaining how empty tabletop reinforces the equation."
}

{
  "object": "street lamp",
  "equation": "Palette = (#0f172a, #fbbf24, #ecfccb)",
  "explanation": "Midnight blues contrasting with sodium glow and soft foliage accents.",
  "guide": "Step 1: Lay down a vertical gradient background from #0f172a to #1e293b. Step 2: Outline the lamp post using ink-like strokes, thickening the base for weight. Step 3: Project a cone of light in #fbbf24 with stippled particles falling through it; annotate intensity falloff. Step 4: Paint nearby foliage silhouettes in #ecfccb to indicate reflected glow. Step 5: Add two vertical leading lines guiding the eye upward and place palette chips with arrows pointing to their use regions."
}

{
  "object": "window plant",
  "equation": "Composition = diagonal flow",
  "explanation": "Leaves guide the viewer from lower-left to upper-right.",
  "guide": "Step 1: Sketch translucent leaves and indicate overlapping fronds. Step 2: Overlay two diagonal bands running from the pot to the window corner, labeling them 'flow guide A' and 'flow guide B'. Step 3: Attach small color swatches near key leaves showing shifts from cool shadow greens to sunlit yellow-greens. Step 4: Annotate brushwork instructions (dry-brush edges, wet glaze near glare). Step 5: Add arrowheads along the diagonal bands and a caption describing how the flow leads the eye through the composition."
}

{
  "object": "bowl of fruit",
  "equation": "Palette = (#f97316, #facc15, #fef9c3) · Rhythm = repeating crescents",
  "explanation": "Warm triadic palette with echoing curved shapes.",
  "guide": "Step 1: Outline each fruit and accentuate their curvature with contour lines. Step 2: Overlay concentric crescent guides across the bowl, labeling them to show repeated rhythms. Step 3: Assign each palette color to specific fruit zones using swatches with leader lines. Step 4: Mark highlight regions with 'glaze' annotations and shadow zones with 'scumble' notes to imply texture. Step 5: Include a mini legend summarizing how the palette rotates through the arrangement and how crescents echo across the forms."
}

{
  "object": "bicycle",
  "equation": "Rule-of-thirds anchors = saddle & front hub",
  "explanation": "Place critical features on grid intersections.",
  "guide": "Step 1: Overlay a rule-of-thirds grid across the camera frame. Step 2: Pin the saddle and front hub to the nearest grid intersections using glowing crosshair icons. Step 3: Trace frame tubes in complementary accent colors and annotate where to apply each tone. Step 4: Add motion direction arrows along wheels and brushstroke cues for spokes. Step 5: Include a caption explaining how anchoring saddle/hub stabilizes the design and show mini diagrams of alternative placements."
}

{
  "object": "book stack",
  "equation": "Contrast = cool shadows vs warm covers",
  "explanation": "Push chroma difference to create depth.",
  "guide": "Step 1: Block in book forms, noting light direction. Step 2: Fill shadow planes with cool violet washes labeled with exact hex codes. Step 3: Apply warm stippled highlights along spines and annotate brush settings. Step 4: Add a radial vignette with gradient arrows showing intensity falloff. Step 5: Provide a two-column legend—cool mix vs warm mix—with leader lines pointing to representative areas."
}

{
  "object": "desk workspace",
  "equation": "Lighting = key + rim",
  "explanation": "Two-source lighting adds dimensionality.",
  "guide": "Step 1: Draw arrows indicating key light direction and annotate their color temperature. Step 2: Highlight rim edges with a contrasting color strip and label thickness. Step 3: Render soft shadow gradients beneath objects with notes about feathering. Step 4: Place palette swatches for key and rim lights, linking them to the annotated edges. Step 5: Add a caption describing how the dual light scheme improves depth perception."
}

{
  "object": "water glass",
  "equation": "Palette = monochrome teal",
  "explanation": "Single hue with value shifts emphasizes transparency.",
  "guide": "Step 1: Trace the glass with vertical contour hatching using a dark teal base. Step 2: Overlay vertical reflection bands in lighter teal and label them 'specular streaks'. Step 3: Draw circular ripples on the water surface with dashed lines and annotate their opacity. Step 4: Create a monochrome palette strip showing value progression along with usage notes. Step 5: Mark tiny white flecks for high-intensity highlights with instructions on brush pressure."
}

{
  "object": "earphones",
  "equation": "Composition ratio = 3:1 cable to buds",
  "explanation": "Use cable curves as leading lines.",
  "guide": "Step 1: Plot the cable as a sweeping S-curve covering roughly three-quarters of the frame; mark the 3:1 ratio with measurement ticks. Step 2: Draw the buds as luminous nodes placed on rule-of-thirds intersections. Step 3: Indicate shadow drops beneath the cable with soft airbrush strokes and annotate blending directions. Step 4: Add arrows along the cable to show eye flow toward the buds. Step 5: Include notes on thickening vs tapering the line to emphasize depth."
}

{
  "object": "tea kettle",
  "equation": "Palette = (#1c1917, #d97706, #fef3c7) · Texture = brushed metal",
  "explanation": "Dark body with gold highlights and soft specular bloom.",
  "guide": "Step 1: Outline the kettle and fill it with curved hatch marks following the form to mimic brushed metal. Step 2: Place palette chips near the body and connect them to usage zones (base fill, highlight, bloom). Step 3: Paint reflective streak overlays along the curvature with gradient arrows showing fade. Step 4: Render the steam plume in #fef3c7 and annotate it as 'bloom highlight'. Step 5: Add instructions on where to leave hard edges versus soft blends to capture metallic sheen."
}"""

PHASE1_CONFIG = {
    "mathematician": {"template": PHASE1_PROMPT, "example": MATHEMATICIAN_EXAMPLE},
    "physicist": {"template": PHASE1_PROMPT_PHYSICS, "example": PHYSICIST_EXAMPLE},
    "biologist": {"template": PHASE1_PROMPT_BIOLOGY, "example": BIOLOGIST_EXAMPLE},
    "artist": {"template": PHASE1_PROMPT_ART, "example": ARTIST_EXAMPLE},
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
