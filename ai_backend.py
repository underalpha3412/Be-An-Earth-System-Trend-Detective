import json
import re
import os
import ssl
import traceback
import urllib.error
import urllib.request
from typing import Any

import certifi
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# Create an NVIDIA API key at https://build.nvidia.com/ and put it in .env.
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3.5-lightning-30b-a3b"
).strip()
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Use a trusted CA bundle on macOS; do not disable HTTPS certificate checks.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

app = FastAPI(title="Earth Trends AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local development only; restrict before deployment.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    city: str
    latitude: float
    longitude: float
    period: str
    parameters: list[dict[str, Any]] = []
    charts: list[dict[str, Any]] = []
    selected_year: str = "ALL"
    analysis_scope: str = "all"


def calculate_statistics(parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []

    for param in parameters:
        label = param.get("parameter") or param.get("label") or param.get("name") or param.get("key") or "Unknown parameter"
        unit = param.get("unit", "")
        points = (
            param.get("values")
            or param.get("data")
            or param.get("years")
            or []
        )

        if isinstance(points, dict):
            points = [
                {"year": year, "value": value}
                for year, value in points.items()
            ]

        clean_points = []
        for point in points:
            if not isinstance(point, dict):
                continue
            year = point.get("year")
            value = point.get("value")
            if year is None or value is None:
                continue
            try:
                clean_points.append({
                    "year": int(year),
                    "value": float(value),
                })
            except (TypeError, ValueError):
                continue

        clean_points.sort(key=lambda item: item["year"])
        if len(clean_points) < 2:
            continue

        first, last = clean_points[0], clean_points[-1]
        change = last["value"] - first["value"]
        percentage = (
            change / abs(first["value"]) * 100
            if first["value"] != 0 else None
        )

        changes = [
            {
                "from_year": previous["year"],
                "to_year": current["year"],
                "change": round(current["value"] - previous["value"], 3),
            }
            for previous, current in zip(clean_points, clean_points[1:])
        ]

        results.append({
            "parameter": label,
            "unit": unit,
            "first_year": first["year"],
            "last_year": last["year"],
            "starting_value": first["value"],
            "ending_value": last["value"],
            "overall_change": round(change, 3),
            "percentage_change": (
                round(percentage, 2) if percentage is not None else None
            ),
            "largest_yearly_increase": (
                max(changes, key=lambda item: item["change"])
                if changes else None
            ),
            "largest_yearly_decrease": (
                min(changes, key=lambda item: item["change"])
                if changes else None
            ),
            "yearly_values": clean_points,
        })

    return results


def call_nemotron(prompt: str) -> str:
    """Call NVIDIA NIM's OpenAI-compatible chat-completions endpoint."""
    if not NVIDIA_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="NVIDIA_API_KEY is missing from .env. Add your NVIDIA API key and restart the backend.",
        )

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful environmental data analyst. "
                    "Use only the supplied data for numerical claims. "
"Every user-facing insight must begin with one relevant Unicode emoji. "
"Use natural climate-related emojis such as 🌡️, ❄️, 🌧️, 💧, 🌱, 💨, 🔎, or 📊. "
"Never write emoji names in place of the actual emoji characters."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.35,
        "top_p": 0.9,
        "max_tokens": 3000,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    req = urllib.request.Request(
        NVIDIA_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req,
            timeout=120,
            context=SSL_CONTEXT,
        ) as response:
            result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        print(f"NVIDIA NIM HTTP error {error.code}: {details}")
        raise HTTPException(
            status_code=error.code if error.code in (400, 401, 403, 404, 429) else 502,
            detail=f"NVIDIA Nemotron API error ({error.code}): {details}",
        )

    except urllib.error.URLError as error:
        print(f"NVIDIA NIM connection error: {error}")
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not establish a verified HTTPS connection to NVIDIA NIM. "
                f"Check your network, certificates, or proxy settings. Details: {error.reason}"
            ),
        )

    choices = result.get("choices") or []
    if not choices:
        raise HTTPException(
            status_code=502,
            detail=f"NVIDIA Nemotron returned no choices: {json.dumps(result)[:1500]}",
        )

    message = choices[0].get("message") or {}
    content = message.get("content")

    # Some reasoning-model responses can contain no user-facing content when
    # thinking consumes the generation budget. Thinking is disabled above;
    # still fail clearly rather than displaying the string "None".
    if isinstance(content, list):
        content = "\\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("text")
        )

    if not isinstance(content, str) or not content.strip():
        finish_reason = choices[0].get("finish_reason")
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        print(
            "Nemotron returned no visible content. "
            f"finish_reason={finish_reason}; reasoning_present={bool(reasoning)}"
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "Nemotron returned no visible answer. "
                f"Finish reason: {finish_reason}. Please retry the analysis."
            ),
        )

    return content.strip()



@app.get("/")
def home():
    return {
        "message": "Earth Trends Nemotron backend is running",
        "provider": "NVIDIA NIM",
        "model": NVIDIA_MODEL,
        "key_loaded": bool(NVIDIA_API_KEY),
    }



def make_parameter_insights(statistics: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build a concise, data-grounded summary for every selected annual parameter."""
    output = []
    for item in statistics:
        label = str(item.get("parameter") or "Selected parameter")
        unit = str(item.get("unit") or "")
        start = item.get("starting_value")
        end = item.get("ending_value")
        change = item.get("overall_change")
        if start is None or end is None or change is None:
            summary = "Not enough annual values are available to calculate a start-to-end trend."
        else:
            direction = "increased" if change > 0 else "decreased" if change < 0 else "remained broadly stable"
            summary = (
                f"{direction.capitalize()} from {start:g} to {end:g} {unit}; "
                f"net change {change:+g} {unit} across the selected record."
            )
        output.append({"label": label, "summary": summary})
    return output


def make_fallback_field_notes(statistics: list[dict[str, Any]]) -> list[str]:
    """Produce six compact field notes if the model omits or underfills its list."""
    labels = [
        ("temperature", "🌡️ Temperature"),
        ("precip", "🌧️ Precipitation"),
        ("humidity", "💧 Humidity"),
        ("soil", "🌱 Soil heat"),
        ("wind", "💨 Wind"),
    ]
    notes = []
    used = set()
    for item in statistics:
        name = str(item.get("parameter", "")).lower()
        category = next(((key, title) for key, title in labels if key in name), None)
        if not category:
            continue
        key, title = category
        if key in used:
            continue
        used.add(key)
        start, end, change = item.get("starting_value"), item.get("ending_value"), item.get("overall_change")
        unit = item.get("unit", "")
        if start is None or end is None or change is None:
            body = "The selected series does not contain enough annual values for a reliable comparison."
        else:
            direction = "rose" if change > 0 else "fell" if change < 0 else "was broadly stable"
            body = f"{direction.capitalize()} from {start:g} to {end:g} {unit} over the selected record."
        notes.append(f"{title}: {body}")
    if len(notes) < 6:
        notes.append("🔎 Combined pattern: compare the timing of changes across selected variables; correlation alone does not establish a cause.")
    while len(notes) < 6:
        notes.append("📊 Data coverage: this selected view contains fewer distinct climate categories than the full dashboard.")
    return notes[:6]


def emoji_for_parameter(label: str) -> str:
    """Choose a stable, relevant emoji for a climate parameter."""
    name = (label or "").casefold()
    if any(word in name for word in ("max temperature", "maximum temperature", "t2m_max", "temperature max")):
        return "🌡️"
    if any(word in name for word in ("min temperature", "minimum temperature", "t2m_min", "temperature min")):
        return "❄️"
    if any(word in name for word in ("precip", "rain")):
        return "🌧️"
    if any(word in name for word in ("humidity", "qv2m", "moisture")):
        return "💧"
    if any(word in name for word in ("soil", "tsoil")):
        return "🌱"
    if any(word in name for word in ("wind", "ws10m")):
        return "💨"
    if "temperature" in name or "t2m" in name:
        return "🌡️"
    return "📊"


def ensure_leading_emoji(text: str, emoji: str) -> str:
    """Ensure the visible text starts with an emoji without duplicating one."""
    value = (text or "").strip()
    if not value:
        return emoji
    # If the model already starts with an emoji/symbol, preserve its choice.
    if value[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#*-•":
        return value
    return f"{emoji} {value}"


def emoji_for_field_note(index: int) -> str:
    return ("🌡️", "🌧️", "💧", "🌱", "💨", "🔎")[index]


@app.post("/analyze-graph")
def analyze_graph(request: AnalysisRequest):
    try:
        scope = request.analysis_scope if request.analysis_scope in {
            "all", "annual", "temperature", "precipitation", "distribution"
        } else "all"

        statistics = calculate_statistics(request.parameters) if request.parameters else []
        charts = request.charts or []

        if not charts and not statistics:
            raise HTTPException(status_code=400, detail="No graph data was supplied for analysis.")

        parsed = {}
        if scope == "annual":
            # One Nemotron request dedicated only to the currently selected annual parameters.
            prompt = f"""
You are Team Vanguard's climate-data analyst.
City: {request.city}
Selected year filter: {request.selected_year}
Period: {request.period}

ANNUAL STATISTICS FOR THE SELECTED GRAPH PARAMETERS:
{json.dumps(statistics, ensure_ascii=False, indent=2)}

Return ONLY valid JSON:
{{
  "parameter_insights": [
    {{"label": "exact parameter label from the data", "summary": "one concise, specific observation"}}
  ]
}}

Rules:
- Return exactly one object for EVERY supplied parameter, in the same order.
- Copy each parameter label exactly.
- Describe its observed direction and notable variation using only its supplied values.
- Mention starting and ending values or net change when useful.
- Keep each summary to one short sentence, ideally 18–28 words.
- Begin every summary with a relevant actual Unicode emoji.
- Do not discuss parameters that are not in the supplied statistics.
- Do not add general climate theory or claim causes that the data cannot prove.
"""
            raw = call_nemotron(prompt).strip()
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                first, last = cleaned.find("{"), cleaned.rfind("}")
                if first < 0 or last <= first:
                    raise HTTPException(status_code=502, detail="Nemotron returned an unreadable annual response.")
                parsed = json.loads(cleaned[first:last + 1])

        elif scope in {"temperature", "precipitation", "distribution"}:
            prompt = f"""
You are Team Vanguard's climate-data analyst.
City: {request.city}
Selected year: {request.selected_year}
Analyse ONLY this graph: {json.dumps(charts, ensure_ascii=False, indent=2)}

Return ONLY valid JSON in this exact format:
{{"insight": "two concise bullet lines, each based on the supplied graph"}}

Do not discuss other graphs. Use only supplied data. Keep it compact and avoid unproven causal claims.
"""
            raw = call_nemotron(prompt).strip()
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                first, last = cleaned.find("{"), cleaned.rfind("}")
                if first < 0 or last <= first:
                    raise HTTPException(status_code=502, detail="Nemotron returned an unreadable graph response.")
                parsed = json.loads(cleaned[first:last + 1])

        else:
            prompt = f"""
You are the climate-data analyst for Team Vanguard's Earth Trends dashboard.

LOCATION: {request.city}
COORDINATES: {request.latitude}, {request.longitude}
SELECTED YEAR: {request.selected_year}
PERIOD: {request.period}

ANNUAL PARAMETER STATISTICS:
{json.dumps(statistics, ensure_ascii=False, indent=2)}

GRAPH DATA:
{json.dumps(charts, ensure_ascii=False, indent=2)}

Return ONLY valid JSON:
{{
  "chart_insights": {{
    "temperature": "1 or 2 short bullet lines",
    "precipitation": "1 or 2 short bullet lines",
    "distribution": "1 or 2 short bullet lines"
  }},
  "parameter_insights": [
    {{"label": "exact selected parameter label", "summary": "one concise observation"}}
  ],
  "overall_insights": [
    "concise temperature observation",
    "concise rainfall observation",
    "concise humidity observation",
    "concise soil-temperature observation",
    "concise wind observation",
    "concise cross-graph takeaway"
  ]
}}

Rules:
- Include one parameter_insights entry for every supplied annual parameter, in the same order.
- Use exact supplied values only. Do not invent values or events.
- Keep each parameter insight to one concise sentence.
- Keep each graph insight to at most two short bullet lines.
- Return exactly six short overall insights; mention unsupported categories as unavailable rather than guessing.
- Possible drivers must be clearly described as possibilities, not proven causes.
- Avoid generic theory, long introductions, and repeated information.
- Begin every parameter insight and every overall insight with a relevant actual Unicode emoji.
"""
            raw = call_nemotron(prompt).strip()
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                first, last = cleaned.find("{"), cleaned.rfind("}")
                if first < 0 or last <= first:
                    raise HTTPException(status_code=502, detail="Nemotron returned an unreadable response.")
                parsed = json.loads(cleaned[first:last + 1])

        # Match Nemotron's output to the exact input parameter list. Fill any omitted
        # item with a transparent data-derived sentence so no selected series vanishes.
        model_items = parsed.get("parameter_insights") or []
        model_map = {
            str(item.get("label", "")).strip().casefold(): str(item.get("summary", "")).strip()
            for item in model_items if isinstance(item, dict)
        } if isinstance(model_items, list) else {}

        parameter_insights = []
        for stat in statistics:
            label = str(stat.get("parameter") or "Selected parameter")
            summary = model_map.get(label.casefold(), "")
            if not summary:
                start_value = stat.get("starting_value")
                end_value = stat.get("ending_value")
                change = stat.get("overall_change")
                unit = stat.get("unit", "")
                if start_value is not None and end_value is not None and change is not None:
                    direction = "increased" if change > 0 else "decreased" if change < 0 else "remained broadly stable"
                    summary = (
                        f"{direction.capitalize()} from {start_value:g} to {end_value:g} {unit}; "
                        f"net change {change:+g} {unit} over the selected period."
                    )
                else:
                    summary = "Not enough annual values are available for a reliable comparison."
            parameter_insights.append({"label": label, "summary": ensure_leading_emoji(summary, emoji_for_parameter(label))})

        if scope == "annual":
            return {
                "city": request.city,
                "selected_year": request.selected_year,
                "analysis_scope": scope,
                "parameter_insights": parameter_insights,
                "statistics": statistics,
            }

        if scope in {"temperature", "precipitation", "distribution"}:
            return {
                "city": request.city,
                "selected_year": request.selected_year,
                "analysis_scope": scope,
                "insight": str(parsed.get("insight", "")).strip(),
            }

        chart_insights = parsed.get("chart_insights") or {}
        overall = parsed.get("overall_insights") or []
        if not isinstance(overall, list):
            overall = [str(overall)]
        overall = [str(item).strip() for item in overall if str(item).strip()][:6]
        overall = [ensure_leading_emoji(item, emoji_for_field_note(index)) for index, item in enumerate(overall)]
        while len(overall) < 6:
            index = len(overall)
            overall.append(emoji_for_field_note(index) + " No additional observation was returned for this category.")
        return {
            "city": request.city,
            "selected_year": request.selected_year,
            "analysis_scope": scope,
            "chart_insights": {
                "annual": "",
                "temperature": ensure_leading_emoji(str(chart_insights.get("temperature", "")).strip(), "🌡️"),
                "precipitation": ensure_leading_emoji(str(chart_insights.get("precipitation", "")).strip(), "🌧️"),
                "distribution": ensure_leading_emoji(str(chart_insights.get("distribution", "")).strip(), "📊"),
            },
            "parameter_insights": parameter_insights,
            "overall_insights": overall,
            "statistics": statistics,
        }

    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Backend error: {type(error).__name__}: {error}",
        )
