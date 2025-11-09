from google import genai
from google.genai import types
from rembg import remove
from PIL import Image
import os
from dotenv import load_dotenv

# -------------------- SETUP --------------------
load_dotenv()
api_key = os.environ.get("GOOGLE_GENAI_API_KEY")
client = genai.Client(api_key=api_key)

# -------------------- GENERATION --------------------
def generate_vertex_overlay(prompt_text, object_name, image_path=None, lens_mode="math", explanation=None):
    # === MATH PROMPT ===
    if lens_mode == "math":
        lens_prompt = (
            "You are an educational assistant specialized in mathematical visualization.\n\n"
            "🎨 Rendering goal:\n"
            "- Produce a glowing cyan skeleton diagram of the object described.\n"
            "- Include labeled parameters (r, h, w, etc.) and the equation in neon-blue.\n"
            "- Include the short explanation text in white, positioned near the equation.\n"
            "- Keep background transparent.\n"
            "- The result must look like a holographic blueprint, NOT a photo overlay.\n\n"
            "🚫 Do NOT:\n"
            "- Include real photo texture, lighting, reflections, or colors.\n"
            "- Overlay on a real image.\n"
            "- Render realistic objects.\n\n"
            "Your output should visualize only the geometric skeleton and annotations, "
            "like a digital AR teaching overlay with no real background."
        )

    else:
        lens_prompt = (
            "You are an educational AR assistant. "
            "Overlay helpful annotations while keeping the photo realistic. "
        )

    
    contents = [f"{lens_prompt}\n\nInput object: {object_name}\n\nVisual guide:\n{prompt_text}"]

    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            contents.append(image)
        except Exception as e:
            print(f"⚠️ Could not open image: {e}")

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=contents,
    )

    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/{object_name}_{lens_mode}.png"

    for part in response.parts:
        if getattr(part, "inline_data", None):
            image = part.as_image()
            image.save(output_path)
            final = Image.open(output_path).convert("RGBA")
            final_no_bg = remove(final)
            final_no_bg.save(output_path)
            return output_path
    return None