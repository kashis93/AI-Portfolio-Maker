"""
AI Portfolio Maker - Resume extraction engine.

Strategy
--------
1. Try an LLM provider (Groq / Anthropic / Ollama) when an API key is present.
2. Otherwise (or if the LLM call fails) fall back to a deterministic
   regex / heuristic parser so the app ALWAYS produces a portfolio.
"""
import os
import io
import re
import json
import logging
from typing import Any, Dict, List, Optional

import pdfplumber
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Pydantic schema
# --------------------------------------------------------------------------- #

class ProjectItem(BaseModel):
    title: str = Field(default="", description="Title of the project")
    description: str = Field(default="", description="Short 1-2 sentence project description")
    items: list[str] = Field(default_factory=list, description="Bullet points or key highlights")
    github_url: str = Field(default="", description="GitHub repository URL")
    live_url: str = Field(default="", description="Live demo URL")
    tech_stack: list[str] = Field(default_factory=list, description="List of technologies used")


class EducationItem(BaseModel):
    institution: str = Field(default="", description="University, college or school name")
    degree: str = Field(default="", description="Degree or major")
    years: str = Field(default="", description="Years attended e.g. 2020 - 2024")
    gpa: str = Field(default="", description="GPA, CPI, or percentage score")


class ExperienceItem(BaseModel):
    company: str = Field(default="", description="Company or organization name")
    role: str = Field(default="", description="Job title or role")
    dates: str = Field(default="", description="Dates or duration of internship/employment")
    description: str = Field(default="", description="Key responsibilities and achievements")


class SkillCategory(BaseModel):
    category: str = Field(default="Technical Skills", description="Category name e.g. Languages, Frameworks, Tools")
    skills: list[str] = Field(default_factory=list, description="List of skills in this category")


class extract_info(BaseModel):
    Name: str = Field(default="", description="Full name of the candidate")
    Introduction: str = Field(default="", description="Strictly 1-sentence tagline/headline summarizing role, max 15 words.")
    About: str = Field(default="", description="2-4 sentence summary/about paragraph. Return empty string if not found.")
    Email: str = Field(default="", description="Email address")
    Phone: str = Field(default="", description="Phone number")
    Github: str = Field(default="", description="GitHub profile URL. Return empty string if not found.")
    Linkedin: str = Field(default="", description="LinkedIn profile URL. Return empty string if not found.")
    Portfolio_URL: str = Field(default="", description="Personal portfolio URL. Return empty string if not found.")
    Twitter: str = Field(default="", description="Twitter / X profile URL. Return empty string if not found.")
    Education: list[EducationItem] = Field(default_factory=list, description="List of education items")
    Experience: list[ExperienceItem] = Field(default_factory=list, description="List of internships or work experience items")
    Projects: list[ProjectItem] = Field(default_factory=list, description="List of projects")
    SkillsGrouped: list[SkillCategory] = Field(default_factory=list, description="Skills grouped by category")
    Hobbies: list[str] = Field(default_factory=list, description="List of hobbies or interests. Return empty list if not found.")
    missing_fields: list[str] = Field(default_factory=list, description="List of field names that were missing or empty in resume")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _norm_url(url: str) -> str:
    """Make sure every URL has a scheme and no trailing junk."""
    if not url:
        return ""
    url = url.strip().rstrip(".,;:!?)>'\"]")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def compute_missing_fields(data: dict) -> list[str]:
    """Identify key profile fields that were empty or not found in the resume."""
    missing = []
    if not data.get("About"):
        missing.append("About Paragraph")
    if not data.get("Github"):
        missing.append("GitHub Profile Link")
    if not data.get("Linkedin"):
        missing.append("LinkedIn Profile Link")
    if not data.get("Portfolio_URL"):
        missing.append("Portfolio Website Link")
    if not data.get("Experience"):
        missing.append("Work Experience / Internships")
    if not data.get("Projects"):
        missing.append("Projects")
    if not data.get("Hobbies"):
        missing.append("Hobbies & Interests")
    return missing


def get_provider_status() -> dict:
    """Report which providers are configured. Frontend uses this to know
    whether to show a 'no key configured' notice or not."""
    return {
        "groq":      bool(os.getenv("GROQ_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "ollama":    True,  # local, always "available" (may not be running)
        "fallback":  True,  # always available
    }


# --------------------------------------------------------------------------- #
# PDF text extraction
# --------------------------------------------------------------------------- #

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract all text from raw PDF bytes using pdfplumber."""
    all_text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    all_text += f"\n--- PAGE {page_num + 1} ---\n{text}\n"
    except Exception as e:
        logger.error(f"PDF parse error: {e}")
        return ""
    return all_text.strip()


def extract_text_from_pdf_path(file_path: str) -> str:
    """Extract text from a PDF file path."""
    with open(file_path, "rb") as f:
        return extract_text_from_pdf_bytes(f.read())


# --------------------------------------------------------------------------- #
# Deterministic fallback parser
# --------------------------------------------------------------------------- #

_KNOWN_SKILLS: Dict[str, List[str]] = {
    "Languages": [
        "Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#",
        "Go", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R",
        "MATLAB", "SQL", "HTML", "CSS", "Bash", "Shell",
    ],
    "Frameworks & Libraries": [
        "React", "Next.js", "Vue", "Nuxt", "Angular", "Svelte",
        "Node.js", "Express", "NestJS", "Django", "Flask", "FastAPI",
        "Spring Boot", "Rails", "Laravel", "ASP.NET", "PyTorch",
        "TensorFlow", "Keras", "Scikit-learn", "Pandas", "NumPy",
        "Tailwind", "Bootstrap", "Material UI", "jQuery",
    ],
    "Tools & Platforms": [
        "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Heroku",
        "Git", "GitHub", "GitLab", "Linux", "Unix", "Jira",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite",
        "Firebase", "Supabase", "Vercel", "Netlify", "Figma",
        "VS Code", "IntelliJ", "PyCharm", "Postman", "Trello",
    ],
}

_SECTION_HEADERS = {
    "education": ["education", "academic", "qualifications", "academic background"],
    "experience": ["experience", "work experience", "employment", "work history",
                   "professional experience", "internships", "internship experience"],
    "projects": ["projects", "personal projects", "key projects", "academic projects",
                 "project experience", "selected projects"],
    "skills": ["skills", "technical skills", "technologies", "tech stack", "expertise"],
    "hobbies": ["hobbies", "interests", "activities", "extracurricular"],
}


def _detect_section(line: str) -> Optional[str]:
    """Return the section name if a line looks like a section header."""
    norm = re.sub(r"[^a-z\s]", "", line.lower()).strip()
    for section, keywords in _SECTION_HEADERS.items():
        for kw in keywords:
            if norm == kw or norm.startswith(kw + " "):
                return section
    return None


def _guess_name(lines: List[str]) -> str:
    """Heuristic: the first non-empty, non-numeric, short, mostly-titlecased line."""
    for line in lines[:8]:
        s = line.strip()
        if not s or "@" in s or "http" in s.lower():
            continue
        if re.search(r"\d", s):  # contains digits (phone, etc.)
            continue
        # skip obvious section labels
        if _detect_section(s):
            continue
        # 2-4 words, mostly letters. Accept either title case ("Kashis Makwana")
        # or all-lowercase / single-name forms (auto title-cased on the way out).
        words = s.split()
        if 1 <= len(words) <= 5 and all(re.match(r"^[A-Za-z\.\-']+$", w) for w in words):
            if any(w[0].isupper() for w in words):
                # title-case names that came in all lowercase
                if all(w[0].islower() for w in words):
                    return " ".join(w.capitalize() for w in words)
                return s
            # all-lowercase name -> title case it
            return " ".join(w.capitalize() for w in words)
    return "Your Name"


def _extract_socials(text: str) -> Dict[str, str]:
    """Robustly extract all profile URLs from raw resume text.

    Handles:
      - Bare handles like "github.com/ksm" (no scheme)
      - With scheme: "https://github.com/ksm"
      - Trailing junk (punctuation, parentheses)
      - Common variations: "linkedin.com/in/" vs "linkedin.com/company/"
    """
    out = {"github": "", "linkedin": "", "portfolio": "", "twitter": ""}

    # Normalize the text: prepend https:// to bare domains so the regex works uniformly
    def _ensure_scheme(m):
        s = m.group(0)
        return s if s.lower().startswith("http") else "https://" + s

    norm = re.sub(
        r"(?<![\w@])(?:www\.)?(github\.com/[A-Za-z0-9_-]+|linkedin\.com/(?:in|company)/[A-Za-z0-9_-]+|"
        r"(?:twitter|x)\.com/[A-Za-z0-9_]+|linkedin\.com|github\.com|"
        r"[a-zA-Z0-9-]+\.(?:com|io|dev|me|app|net|org|co|tech|xyz|ai|info|biz|site|online))",
        _ensure_scheme,
        text,
    )

    # GitHub - accept github.com/anything
    gh = re.search(r"https?://(?:www\.)?github\.com/([a-zA-Z0-9_-]+)/?", norm, re.IGNORECASE)
    if gh:
        out["github"] = f"https://github.com/{gh.group(1)}"

    # LinkedIn - accept linkedin.com/in/<id> or /company/<id>
    li = re.search(
        r"https?://(?:www\.)?linkedin\.com/(?:in|company)/([a-zA-Z0-9_-]+)/?",
        norm,
        re.IGNORECASE,
    )
    if li:
        out["linkedin"] = f"https://linkedin.com/in/{li.group(1)}"

    # Twitter / X
    tw = re.search(
        r"https?://(?:www\.)?(?:twitter\.com|x\.com)/([a-zA-Z0-9_]+)/?",
        norm,
        re.IGNORECASE,
    )
    if tw:
        out["twitter"] = f"https://x.com/{tw.group(1)}"

    # Portfolio - try a few strategies
    # 1. Look for labelled links like "Portfolio: kashis.dev" or "Website: kashis.dev"
    labelled = re.search(
        r"(?:portfolio|website|personal\s+site|homepage)[:\s]+(https?://[^\s,;]+|[\w.-]+\.[a-z]{2,}[^\s,]*)",
        text,
        re.IGNORECASE,
    )
    if labelled:
        url = _norm_url(labelled.group(1))
        if "github.com" not in url and "linkedin.com" not in url and "twitter.com" not in url and "x.com" not in url:
            out["portfolio"] = url

    # 2. If still empty, scan all URLs and grab the first non-social one
    if not out["portfolio"]:
        for url_match in re.finditer(r"https?://[^\s,;<>'\"]+", norm):
            url = url_match.group(0).rstrip(".,;:!?)>'\"]")
            low = url.lower()
            if any(d in low for d in ("github.com", "linkedin.com", "twitter.com", "x.com", "youtube.com", "facebook.com", "instagram.com", "gmail.com", "mailto:")):
                continue
            out["portfolio"] = url
            break

    # 3. Last resort - a bare custom domain on its own line
    if not out["portfolio"]:
        bare = re.search(r"(?<!@)(?<!://)\b([a-zA-Z0-9-]+\.(?:com|io|dev|me|app|net|org|co|tech|xyz|ai|info|biz|site|online))\b", text)
        if bare and "github.com" not in bare.group(0) and "linkedin.com" not in bare.group(0):
            out["portfolio"] = "https://" + bare.group(1)

    return out


def fallback_extraction(text: str) -> dict:
    """Smart regex & heuristic extraction for when LLM is unavailable.

    Returns a fully-formed dict compatible with our portfolio templates.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("--- PAGE")]

    # ---- 1. Name ---------------------------------------------------------
    name = _guess_name(lines)

    # ---- 2. Contact info -------------------------------------------------
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group(0) if email_match else ""

    phone_match = re.search(
        r"(\+?\d[\d\s\-().]{8,}\d)",
        text,
    )
    phone = phone_match.group(0).strip() if phone_match else ""

    # ---- 3. Social URLs --------------------------------------------------
    socials = _extract_socials(text)

    # ---- 4. Introduction & About -----------------------------------------
    intro = "Software professional passionate about building great things."
    about = ""
    for line in lines[1:10]:
        s = line.strip()
        if not s or _detect_section(s):
            continue
        if "@" in s or "http" in s.lower() or re.search(r"\d{4,}", s):
            continue
        if 20 < len(s) < 200:  # looks like a bio sentence
            about = s
            if s.count(".") >= 1:
                intro = s.split(".")[0].strip() + "."
            else:
                intro = s
            break

    if not about and intro:
        about = f"{intro} Dedicated to clean code, robust system architecture, and modern UX design."

    # ---- 5. Skills grouped -----------------------------------------------
    skills_grouped: List[Dict[str, Any]] = []
    all_skills_flat: List[str] = []
    for cat, skill_list in _KNOWN_SKILLS.items():
        found = []
        for s in skill_list:
            if re.search(r"\b" + re.escape(s) + r"\b", text, re.IGNORECASE):
                found.append(s)
                all_skills_flat.append(s)
        if found:
            skills_grouped.append({"category": cat, "skills": found})

    if not skills_grouped:
        skills_grouped = [
            {"category": "Technical Skills", "skills": ["Software Engineering", "Problem Solving", "System Architecture"]}
        ]

    # ---- 6. Projects -----------------------------------------------------
    projects: List[Dict[str, Any]] = []
    proj_section = False
    current_proj: Optional[Dict[str, Any]] = None

    def _flush_proj():
        nonlocal current_proj
        if current_proj:
            projects.append(current_proj)
            current_proj = None

    for line in lines:
        sec = _detect_section(line)
        if sec == "projects":
            proj_section = True
            continue
        if sec in ("education", "experience", "skills", "hobbies") and proj_section:
            proj_section = False

        if not proj_section:
            continue

        if line.startswith(("-", "\u2022", "*", "\u00b7", "\u2013")):
            if current_proj:
                bullet = line.lstrip("-\u2022*\u00b7\u2013 ").strip()
                if bullet:
                    current_proj["items"].append(bullet)
            continue

        # If line is short, capitalised, no period -> likely a project title
        if 2 < len(line) < 70 and not line.endswith(".") and not re.search(r"^https?://", line):
            _flush_proj()
            # Capture any inline github/live URLs on the same line
            gh = re.search(r"https?://(?:www\.)?github\.com/[^\s,;]+", line, re.IGNORECASE)
            lv = re.search(r"https?://(?!github\.com)[^\s,;]+", line, re.IGNORECASE)
            current_proj = {
                "title": re.sub(r"https?://\S+", "", line).strip() or "Project",
                "description": "Project built with modern technologies and best engineering practices.",
                "items": [],
                "github_url": _norm_url(gh.group(0)) if gh else "",
                "live_url":    _norm_url(lv.group(0)) if lv else "",
                "tech_stack":  all_skills_flat[:3],
            }

    _flush_proj()

    if not projects:
        projects = [
            {
                "title": "Personal Portfolio",
                "description": "A full-stack portfolio website showcasing skills, projects, and experience.",
                "items": ["Built with modern web technologies", "Fully responsive design", "Deployed to the cloud"],
                "github_url": "",
                "live_url": "",
                "tech_stack": all_skills_flat[:3] or ["HTML", "CSS", "JavaScript"],
            }
        ]

    # ---- 7. Experience ---------------------------------------------------
    experience: List[Dict[str, Any]] = []
    exp_section = False
    current_exp: Optional[Dict[str, Any]] = None

    def _flush_exp():
        nonlocal current_exp
        if current_exp:
            experience.append(current_exp)
            current_exp = None

    for line in lines:
        sec = _detect_section(line)
        if sec == "experience":
            exp_section = True
            continue
        if sec in ("education", "projects", "skills", "hobbies") and exp_section:
            exp_section = False

        if not exp_section:
            continue

        if line.startswith(("-", "\u2022", "*")):
            if current_exp:
                current_exp["description"] += " " + line.lstrip("-\u2022* ").strip()
            continue

        # Heuristic: if the line mentions a job-related role and is short
        if re.search(r"\b(engineer|developer|intern|lead|associate|manager|analyst|designer|consultant|founder|architect|researcher)\b", line, re.IGNORECASE) and len(line) < 80:
            _flush_exp()
            role = line
            company = ""
            for sep in [" at ", " @ ", " - ", " | "]:
                if sep in line:
                    role, company = [p.strip() for p in line.split(sep, 1)]
                    break
            dates = ""
            dm = re.search(
                r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4})"
                r"\s*[-to\u2013]+\s*"
                r"((?:Present|Now|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{1,2}/\d{4}|\d{4}))",
                line,
                re.IGNORECASE,
            )
            if dm:
                dates = f"{dm.group(1).strip()} - {dm.group(2).strip()}"
            current_exp = {
                "role": role,
                "company": company or "Company",
                "dates": dates,
                "description": "Contributed to impactful projects and collaborated with cross-functional teams.",
            }

    _flush_exp()

    # ---- 8. Education ----------------------------------------------------
    education_list: List[Dict[str, str]] = []
    edu_section = False
    for line in lines:
        sec = _detect_section(line)
        if sec == "education":
            edu_section = True
            continue
        if sec in ("experience", "projects", "skills", "hobbies") and edu_section:
            edu_section = False
        if not edu_section:
            continue
        if re.search(r"\b(bachelor|master|b\.s\.|m\.s\.|b\.tech|m\.tech|b\.e\.|m\.e\.|degree|university|college|school|phd|diploma)\b", line, re.IGNORECASE):
            years = ""
            ym = re.search(r"\b(19|20)\d{2}\s*[-\u2013]\s*((?:19|20)\d{2}|Present|present)", line)
            if ym:
                years = ym.group(0).replace("\u2013", "-")
            gpa = ""
            gm = re.search(r"(gpa|cgpa|cpi|percentage|%)[\s:]*([\d.]+\s*(?:/\s*[\d.]+)?(?:\s*%)?)", line, re.IGNORECASE)
            if gm:
                gpa = gm.group(2).strip()
            education_list.append({
                "institution": line,
                "degree": "",
                "years": years,
                "gpa": gpa,
            })

    if not education_list:
        education_list = [
            {"institution": "Your University", "degree": "Bachelor of Technology", "years": "", "gpa": ""}
        ]

    # ---- 9. Hobbies ------------------------------------------------------
    hobbies: List[str] = []
    hobby_section = False
    for line in lines:
        sec = _detect_section(line)
        if sec == "hobbies":
            hobby_section = True
            continue
        if sec and hobby_section:
            hobby_section = False
        if hobby_section:
            parts = re.split(r"[,;|]", line)
            for p in parts:
                p = p.strip().lstrip("-\u2022*\u00b7\u2013 ")
                if p and len(p) < 40:
                    hobbies.append(p)
    inline = re.search(r"\bhobbies?\s*[:\-]\s*([^\n]+)", text, re.IGNORECASE)
    if inline and not hobbies:
        hobbies = [h.strip() for h in re.split(r"[,;|]", inline.group(1)) if h.strip()]

    result = {
        "Name": name,
        "Introduction": intro[:120],
        "About": about,
        "Email": email,
        "Phone": phone,
        "Github": socials["github"],
        "Linkedin": socials["linkedin"],
        "Portfolio_URL": socials["portfolio"],
        "Twitter": socials["twitter"],
        "Education": education_list,
        "Experience": experience,
        "Projects": projects,
        "SkillsGrouped": skills_grouped,
        "Hobbies": hobbies,
        "_provider_used": "fallback",
    }
    result["missing_fields"] = compute_missing_fields(result)
    return result


# --------------------------------------------------------------------------- #
# LLM-driven extraction
# --------------------------------------------------------------------------- #

def _try_llm(resume_text: str, provider: str, api_key: Optional[str]) -> Optional[dict]:
    """Attempt one LLM call. Return parsed dict or None on any failure."""
    try:
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import PromptTemplate
    except Exception as e:
        logger.warning(f"langchain_core not importable: {e}")
        return None

    provider = (provider or "").lower()
    model = None

    try:
        if provider == "groq":
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                return None
            from langchain_groq import ChatGroq
            model = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=key)
        elif provider == "anthropic":
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                return None
            from langchain_anthropic import ChatAnthropic
            model = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=key)
        elif provider == "ollama":
            try:
                from langchain_ollama import OllamaLLM
                model = OllamaLLM(model="llama3.2:latest")
            except Exception:
                try:
                    from langchain_community.llms import Ollama
                    model = Ollama(model="llama3.2")
                except Exception:
                    return None
        else:
            return None
    except Exception as e:
        logger.warning(f"Provider init failed for {provider}: {e}")
        return None

    try:
        parser = PydanticOutputParser(pydantic_object=extract_info)
        template = PromptTemplate(
            template="""You are an expert resume parser. Extract the following information from the resume text into strict JSON format.

CRITICAL RULES:
1. DO NOT HALLUCINATE OR MAKE UP PLACEHOLDER DATA. If a field is NOT present in the resume text, return empty string "" or empty list [].
2. Extract every URL EXACTLY as written in the resume. Do not invent handles.
3. If a "Github" or "LinkedIn" handle appears without a scheme, prefix it with "https://".
4. SkillsGrouped: group by category. Common categories: "Languages", "Frameworks & Libraries", "Tools & Platforms".
5. Projects: include GitHub repo URL in `github_url` and live demo URL in `live_url` IF present in the resume. Tech stack = the technologies listed for that project.

Resume Content:
{text}

{format_instruction}""",
            input_variables=["text"],
            partial_variables={"format_instruction": parser.get_format_instructions()},
        )
        chain = template | model | parser
        result = chain.invoke({"text": resume_text})
        data = result.dict() if hasattr(result, "dict") else result.model_dump()

        # Backfill any URLs the LLM may have missed with our regex
        socials = _extract_socials(resume_text)
        if not data.get("Github")       and socials["github"]:    data["Github"]       = socials["github"]
        if not data.get("Linkedin")     and socials["linkedin"]:  data["Linkedin"]     = socials["linkedin"]
        if not data.get("Twitter")      and socials["twitter"]:   data["Twitter"]      = socials["twitter"]
        if not data.get("Portfolio_URL") and socials["portfolio"]: data["Portfolio_URL"] = socials["portfolio"]

        data["missing_fields"] = compute_missing_fields(data)
        data["_provider_used"] = provider
        return data
    except Exception as e:
        logger.warning(f"LLM call failed for {provider}: {e}")
        return None


def parse_resume_with_llm(resume_text: str, provider: Optional[str] = None, api_key: Optional[str] = None) -> dict:
    """Try providers in order, fall back to deterministic extraction.

    Order:
        1. explicit `provider` arg, if available
        2. LLM_PROVIDER env var
        3. any provider with a key (groq -> anthropic -> ollama)
        4. fallback regex parser
    """
    requested = (provider or os.getenv("LLM_PROVIDER") or "").lower()
    candidates: List[str] = []
    if requested and requested != "fallback":
        candidates.append(requested)
    for p in ("groq", "anthropic", "ollama"):
        if p not in candidates:
            candidates.append(p)

    for prov in candidates:
        if prov == "fallback":
            continue
        result = _try_llm(resume_text, prov, api_key)
        if result is not None:
            return result

    # Final fallback - always works
    return fallback_extraction(resume_text)
