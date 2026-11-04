import os
import io
import re
import json
from flask import Flask, request, jsonify, render_template_string, send_file, send_from_directory
from jinja2 import Environment, FileSystemLoader
from llm_extraction.extractor import (
    extract_text_from_pdf_bytes,
    parse_resume_with_llm,
    get_provider_status,
)

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Configure Jinja2 loader for portfolio templates
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_extraction", "templates")
jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# Themes that each template supports (templates can opt-in to light/dark)
TEMPLATES_MAP = {
    "aurora": {
        "file": "aurora.html.j2",
        "name": "Aurora 3D Glassmorphism",
        "badge": "New - Recommended",
        "description": "Stunning 3D animated background, glassmorphism, full dark/light toggle.",
        "supports_theme": True,
    },
    "hero_spotlight": {
        "file": "hero_spotlight.html.j2",
        "name": "Hero Spotlight Signature",
        "badge": "Featured",
        "description": "Massive glowing Name title, signature monogram badge, and ambient backdrop.",
        "supports_theme": False,
    },
    "glassmorphism": {
        "file": "Portfolio.html.j2",
        "name": "Glassmorphism Cyberpunk",
        "badge": "Popular",
        "description": "Ultra-modern dark aesthetic with blurred glass panels and glowing accents.",
        "supports_theme": False,
    },
    "minimal": {
        "file": "minimal.html.j2",
        "name": "Executive Minimal",
        "badge": "Clean",
        "description": "High-contrast, elegant typography focused on readability and structure.",
        "supports_theme": False,
    },
    "neovibrant": {
        "file": "neovibrant.html.j2",
        "name": "Neo-Vibrant Developer",
        "badge": "Tech",
        "description": "Terminal-inspired cyberpunk design with neon tags and code badges.",
        "supports_theme": False,
    },
}


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/templates", methods=["GET"])
def get_templates():
    return jsonify({
        "templates": [
            {
                "id": tid,
                "name": info["name"],
                "badge": info["badge"],
                "description": info["description"],
                "supports_theme": info["supports_theme"],
            }
            for tid, info in TEMPLATES_MAP.items()
        ],
        "providers": get_provider_status(),
    })


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if file.filename == "" or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file"}), 400

    # Provider is now auto-picked server-side from .env / available keys.
    # The frontend no longer needs to ask the user.
    provider = request.form.get("provider")  # optional override
    api_key = request.form.get("api_key", "").strip() or None

    try:
        pdf_bytes = file.read()
        extracted_text = extract_text_from_pdf_bytes(pdf_bytes)

        if not extracted_text:
            return jsonify({
                "error": "Could not extract text from this PDF. It might be scanned/image-only - try a text-based PDF."
            }), 400

        data = parse_resume_with_llm(extracted_text, provider=provider, api_key=api_key)
        return jsonify({
            "success": True,
            "data": data,
            "text_preview": extracted_text[:400],
            "provider_used": data.get("_provider_used", "fallback"),
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500


@app.route("/api/preview", methods=["POST"])
def preview_portfolio():
    payload = request.get_json() or {}
    template_id = payload.get("template_id", "aurora")
    theme = (payload.get("theme") or "dark").lower()
    data = payload.get("data", {})

    template_info = TEMPLATES_MAP.get(template_id)
    if not template_info:
        return jsonify({"error": "Unknown template"}), 400

    template_file = template_info["file"]

    try:
        tmpl = jinja_env.get_template(template_file)
        # Inject theme (templates that ignore it just don't use it)
        data_with_theme = {**data, "theme": theme}
        rendered_html = tmpl.render(data_with_theme)
        return jsonify({"html": rendered_html})
    except Exception as e:
        return jsonify({"error": f"Rendering error: {str(e)}"}), 500


@app.route("/api/export", methods=["POST"])
def export_portfolio():
    payload = request.get_json() or {}
    template_id = payload.get("template_id", "aurora")
    theme = (payload.get("theme") or "dark").lower()
    data = payload.get("data", {})

    template_info = TEMPLATES_MAP.get(template_id)
    if not template_info:
        return jsonify({"error": "Unknown template"}), 400
    template_file = template_info["file"]

    try:
        tmpl = jinja_env.get_template(template_file)
        data_with_theme = {**data, "theme": theme}
        rendered_html = tmpl.render(data_with_theme)

        mem_file = io.BytesIO(rendered_html.encode("utf-8"))
        name_slug = (data.get("Name") or "Portfolio").replace(" ", "_")
        theme_slug = f"_{theme}" if template_info["supports_theme"] else ""
        return send_file(
            mem_file,
            as_attachment=True,
            download_name=f"{name_slug}{theme_slug}_Portfolio.html",
            mimetype="text/html"
        )
    except Exception as e:
        return jsonify({"error": f"Export error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
