"""
RescueNet - AI-Based Emergency Reporting and Resource Allocation System
Run with:  streamlit run app.py
"""
import os
import json
import math
import random
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from bayesian_utils import bayesian_severity_confidence
from astar_utils import astar_route

# =====================================================================
# CONFIG
# =====================================================================
st.set_page_config(page_title="RescueNet", page_icon="🚨", layout="wide")

LOG_FILE = "emergency_logs.csv"
RESOURCE_FILE = "resources.csv"
RESOURCE_TYPES = ["Ambulance", "Fire Brigade", "Police", "Rescue Team"]
RESOURCE_ICONS = {"Ambulance": "🚑", "Fire Brigade": "🚒", "Police": "🚓", "Rescue Team": "🛟"}
RESOURCE_COLORS = {"Ambulance": "#34D399", "Fire Brigade": "#FF4D5E",
                   "Police": "#5AA2FF", "Rescue Team": "#FFB02E"}
STATUS_FLOW = ["Pending", "Dispatched", "Accepted", "En Route", "Reached", "Resolved"]
SEVERITY_COLORS = {"Critical": "#E5243B", "High": "#F28C28", "Medium": "#D4A017", "Low": "#1FA97A"}
AVG_SPEED_KMH = 30      # average city speed used for ETA
ROAD_FACTOR = 1.3       # straight-line distance -> approximate road distance
TIME_FMT = "%Y-%m-%d %H:%M:%S"

# Simulated GPS positions (change these to your own city)
LOCATIONS = {
    "Shivajinagar": (18.5308, 73.8475),
    "Kothrud": (18.5074, 73.8077),
    "Hadapsar": (18.5089, 73.9260),
    "Pimpri": (18.6279, 73.8009),
    "Swargate": (18.5018, 73.8636),
    "Viman Nagar": (18.5679, 73.9143),
    "Baner": (18.5590, 73.7868),
    "Katraj": (18.4575, 73.8677),
}

# Demo accounts  (role -> {username: password})
CONTROL_USERS = {"control": "control123"}
ADMIN_USERS = {"admin": "admin123"}
UNIT_PASSWORD = "unit123"   # same demo password for every resource unit

LOG_COLS = ["emergency_id", "time", "citizen_name", "phone", "report_text",
            "emergency_type", "severity", "priority", "required_resources",
            "lat", "lon", "location_name", "recommended", "assigned_units",
            "status", "dispatched_time", "resolved_time"]

# =====================================================================
# STYLE
# =====================================================================
import altair as alt

STATUS_COLORS = {"Pending": "#FFB02E", "Dispatched": "#5AA2FF", "Accepted": "#A78BFA",
                 "En Route": "#2DD4D4", "Reached": "#34D399", "Resolved": "#9AA7BC"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
:root { --bg:#0B1220; --panel:#131C2E; --panel2:#1A2540; --line:#2A3A5E; --text:#E8EDF5; --muted:#9AA9C2; --red:#FF4D5E; }
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Manrope', sans-serif; }
.stApp { background: radial-gradient(1100px 480px at 88% -8%, rgba(255,77,94,.20), transparent 62%),
         radial-gradient(900px 400px at -5% 30%, rgba(90,162,255,.10), transparent 60%), var(--bg); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1250px; }

/* text colours (explicit so nothing is ever invisible) */
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] h1, .stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3, .stApp [data-testid="stMarkdownContainer"] h4,
.stApp label, .stApp label p, .stApp [data-testid="stWidgetLabel"] p, .stApp [data-testid="stRadio"] div { color: var(--text); }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p { color: var(--muted); }
.stApp code { background: var(--panel2); color: #FFD166; }

/* sidebar */
[data-testid="stSidebar"] { background: #070C17; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] button { background: transparent; border: 1px solid var(--line); border-radius: 10px; color: var(--text); }
[data-testid="stSidebar"] button:hover { border-color: var(--red); color: #fff; }
.side-brand { font-size: 1.5rem; font-weight: 800; color: #fff; margin-bottom: 6px; }
.side-pill { display: inline-block; padding: 3px 12px; border-radius: 99px; background: var(--red); color: #fff; font-weight: 700; font-size: .8rem; }

/* hero */
.rn-hero { position: relative; overflow: hidden; border-radius: 20px; margin-bottom: 20px; border: 1px solid var(--line);
  background: linear-gradient(120deg, #101B33 0%, #1B2A4F 55%, #7E1A2E 100%); box-shadow: 0 12px 34px rgba(0,0,0,.45); }
.rn-hero::after { content: ""; position: absolute; right: -60px; top: -60px; width: 280px; height: 280px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,77,94,.55) 0%, rgba(255,77,94,0) 70%); }
.rn-hero-in { position: relative; z-index: 1; padding: 30px 34px; }
.rn-hero h1 { margin: 0; font-size: 2.3rem; font-weight: 800; color: #fff !important; letter-spacing: -.5px; }
.rn-hero p { margin: 6px 0 14px 0; color: #C9D5EA !important; font-size: 1.02rem; max-width: 640px; }
.hero-chip { display: inline-block; margin: 0 8px 6px 0; padding: 6px 14px; border-radius: 99px;
  background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.25); color: #fff; font-weight: 600; font-size: .85rem; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: var(--panel2); border: 1px solid var(--line); border-radius: 99px; padding: 8px 18px; height: auto; }
.stTabs [data-baseweb="tab"] p { color: var(--muted); font-weight: 600; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #FF4D5E, #C71F36); border-color: transparent; }
.stTabs [aria-selected="true"] p { color: #fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* buttons, inputs, containers */
button[kind="primary"] { background: linear-gradient(135deg, #FF4D5E, #C71F36); border: none; border-radius: 12px; font-weight: 700; color: #fff; }
button[kind="primary"]:hover { filter: brightness(1.1); color: #fff; }
button[kind="secondary"] { background: var(--panel2); border: 1px solid var(--line); border-radius: 12px; font-weight: 600; color: var(--text); }
button[kind="secondary"]:hover { border-color: var(--red); color: #fff; }
button:disabled { opacity: .4; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], textarea, input
  { background: #0E1627 !important; color: var(--text) !important; border-radius: 10px !important; border-color: var(--line) !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; background: var(--panel); border-color: var(--line); box-shadow: 0 6px 20px rgba(0,0,0,.30); }
[data-testid="stMetric"] { background: var(--panel2); border: 1px solid var(--line); border-radius: 14px; padding: 12px 16px; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] p { color: var(--text); }

/* kpi cards */
.kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 14px 18px; box-shadow: 0 6px 18px rgba(0,0,0,.30); margin-bottom: 12px; }
.kpi-top { display: flex; justify-content: space-between; color: var(--muted); font-weight: 600; font-size: .9rem; }
.kpi-val { font-size: 2rem; font-weight: 800; color: #fff; line-height: 1.2; }

/* generic bits */
.rn-card { background: var(--panel2); border: 1px solid var(--line); border-radius: 14px; padding: 14px 18px; margin-bottom: 10px; }
.rn-card h4 { margin: 0 0 4px 0; font-size: .9rem; color: var(--muted); font-weight: 600; }
.rn-card .big { font-size: 1.4rem; font-weight: 800; color: #fff; }
.rn-badge { display: inline-block; padding: 3px 12px; border-radius: 99px; color: #fff; font-weight: 700; font-size: .85rem; }
.chip { display: inline-block; padding: 6px 14px; margin: 3px 6px 3px 0; border-radius: 99px; background: var(--panel2); color: var(--text); border: 2px solid; font-weight: 700; }
.sub { color: var(--muted); font-size: .85rem; }
.step { font-size: 1.15rem; font-weight: 800; color: #fff; margin-bottom: 6px; }
.step-n { display: inline-flex; width: 30px; height: 30px; border-radius: 50%; background: linear-gradient(135deg, #FF4D5E, #C71F36); color: #fff;
  align-items: center; justify-content: center; margin-right: 8px; font-size: .95rem; }
.how { display: flex; gap: 12px; align-items: flex-start; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; margin-bottom: 10px; }
.how .n { flex: none; width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #FF4D5E, #C71F36); color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 800; }
.how b { color: #fff; }

/* unit cards */
.unit { display: flex; justify-content: space-between; align-items: center; gap: 10px; background: var(--panel2); border: 1px solid var(--line);
  border-left: 7px solid #5AA2FF; border-radius: 12px; padding: 10px 14px; margin-bottom: 8px; }
.unit .nm { font-weight: 800; color: #fff; }
.eta { background: rgba(45,212,212,.15); color: #5EEAD4; border: 1px solid rgba(45,212,212,.5); padding: 5px 12px; border-radius: 99px; font-weight: 700; font-size: .85rem; white-space: nowrap; }

/* severity banner */
.sev { display: flex; align-items: center; gap: 18px; border-radius: 16px; padding: 18px 24px; color: #fff; margin-bottom: 14px; box-shadow: 0 8px 24px rgba(0,0,0,.35); }
.sev .ic { font-size: 2.6rem; }
.sev .t { font-size: 1.6rem; font-weight: 800; line-height: 1.1; color: #fff; }
.sev .s { color: #fff; opacity: .92; font-weight: 600; }
.sev .p { margin-left: auto; background: rgba(0,0,0,.25); border-radius: 12px; padding: 8px 16px; font-weight: 800; font-size: 1.1rem; text-align: center; color: #fff; }

/* status timeline */
.tl { display: flex; margin: 8px 0 18px; }
.tl-step { flex: 1; text-align: center; position: relative; }
.tl-step::before { content: ""; position: absolute; top: 16px; left: -50%; width: 100%; height: 4px; background: var(--line); z-index: 0; }
.tl-step:first-child::before { display: none; }
.tl-step.done::before, .tl-step.now::before { background: #34D399; }
.tl-dot { position: relative; z-index: 1; width: 36px; height: 36px; border-radius: 50%; margin: 0 auto 6px; background: var(--panel2); border: 2px solid var(--line); color: var(--muted);
  display: flex; align-items: center; justify-content: center; font-weight: 800; }
.tl-step.done .tl-dot { background: #34D399; border-color: #34D399; color: #062; }
.tl-step.now .tl-dot { background: var(--red); border-color: var(--red); color: #fff; box-shadow: 0 0 0 6px rgba(255,77,94,.25); }
.tl-label { font-size: .78rem; color: var(--muted); font-weight: 600; }
.tl-step.now .tl-label { color: var(--red); }
.tl-step.done .tl-label { color: #34D399; }

/* SOS button */
.sos-wrap { text-align: center; margin: 6px 0 4px; }
.sos-wrap h2 { margin: 0; color: #fff; font-weight: 800; }
.sos-wrap p { color: var(--muted); margin: 4px 0 18px; }
.st-key-sos_btn button { width: 100%; height: 210px; border-radius: 50%; background: radial-gradient(circle at 35% 30%, #FF6B7A, #D01F35 70%);
  color: #fff; font-size: 3.4rem; font-weight: 800; border: 10px solid rgba(255,77,94,.35); animation: pulse 1.8s infinite; }
.st-key-sos_btn button:hover { background: radial-gradient(circle at 35% 30%, #FF4D5E, #B01228 70%); color: #fff; border-color: rgba(255,77,94,.5); }
.st-key-sos_btn button p { font-size: 3.4rem; font-weight: 800; color: #fff; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255,77,94,.6); } 70% { box-shadow: 0 0 0 32px rgba(255,77,94,0); } 100% { box-shadow: 0 0 0 0 rgba(255,77,94,0); } }
@media (prefers-reduced-motion: reduce) { .st-key-sos_btn button { animation: none; } }
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle, chips=None):
    chip_html = "".join(f"<span class='hero-chip'>{c}</span>" for c in (chips or []))
    st.markdown(f"<div class='rn-hero'><div class='rn-hero-in'><h1>{title}</h1><p>{subtitle}</p>"
                f"<div>{chip_html}</div></div></div>", unsafe_allow_html=True)


def badge(text, color):
    return f"<span class='rn-badge' style='background:{color}'>{text}</span>"


def card(title, value):
    st.markdown(f"<div class='rn-card'><h4>{title}</h4><div class='big'>{value}</div></div>",
                unsafe_allow_html=True)


def kpi(label, value, color="#5AA2FF", icon=""):
    st.markdown(f"<div class='kpi' style='border-top:5px solid {color}'><div class='kpi-top'>"
                f"<span>{label}</span><span>{icon}</span></div><div class='kpi-val'>{value}</div></div>",
                unsafe_allow_html=True)


def step_title(n, text):
    st.markdown(f"<div class='step'><span class='step-n'>{n}</span>{text}</div>", unsafe_allow_html=True)


def timeline(status):
    status = {"Assigned": "Dispatched"}.get(status, status)
    idx = STATUS_FLOW.index(status) if status in STATUS_FLOW else 0
    items = ""
    for i, s in enumerate(STATUS_FLOW):
        cls = "done" if i < idx else ("now" if i == idx else "")
        mark = "✓" if i < idx else str(i + 1)
        items += f"<div class='tl-step {cls}'><div class='tl-dot'>{mark}</div><div class='tl-label'>{s}</div></div>"
    st.markdown(f"<div class='tl'>{items}</div>", unsafe_allow_html=True)


def severity_banner(severity, etype, priority):
    color = SEVERITY_COLORS.get(severity, "#5B6778")
    icon = {"Critical": "🚨", "High": "⚠️", "Medium": "🟡", "Low": "🟢"}.get(severity, "ℹ️")
    msg = {"Critical": "Immediate dispatch recommended", "High": "High-priority emergency",
           "Medium": "Moderate emergency", "Low": "Low urgency"}.get(severity, "Report recorded")
    st.markdown(f"<div class='sev' style='background:linear-gradient(120deg,{color},{color}CC)'>"
                f"<div class='ic'>{icon}</div><div><div class='t'>{etype} — {severity}</div><div class='s'>{msg}</div></div>"
                f"<div class='p'>Priority<br>{priority}</div></div>", unsafe_allow_html=True)


def unit_card(r):
    if r["unit_id"]:
        color = RESOURCE_COLORS[r["type"]]
        st.markdown(f"<div class='unit' style='border-left-color:{color}'><div><div class='nm'>"
                    f"{RESOURCE_ICONS[r['type']]} {r['name']}</div><div class='sub'>{r['distance_km']} km away</div></div>"
                    f"<div class='eta'>ETA {r['eta_min']} min</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='unit' style='border-left-color:#FF4D5E'><div><div class='nm'>"
                    f"{RESOURCE_ICONS[r['type']]} {r['name']}</div><div class='sub'>All units busy</div></div></div>",
                    unsafe_allow_html=True)


def resource_chips(names):
    names = [n.strip() for n in names if n.strip()]
    if not names:
        return "<span class='sub'>None needed</span>"
    return "".join(f"<span class='chip' style='border-color:{RESOURCE_COLORS[n]}'>{RESOURCE_ICONS[n]} {n}</span>"
                   for n in names)


def bar_chart(series, color_map=None, default="#FF6B7A", height=280):
    df = series.rename_axis("label").reset_index(name="count")
    df["color"] = df["label"].map(color_map or {}).fillna(default)
    chart = (alt.Chart(df).mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
             .encode(x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30)),
                     y=alt.Y("count:Q", title=None), color=alt.Color("color:N", scale=None),
                     tooltip=["label", "count"]).properties(height=height))
    st.altair_chart(chart, use_container_width=True)


def style_table(df):
    def sev(v):
        return f"background-color:{SEVERITY_COLORS.get(v, '')};color:white;font-weight:700"

    def stat(v):
        return f"color:{STATUS_COLORS.get(v, '#000')};font-weight:700"
    s = df.style
    mapper = getattr(s, "map", None) or s.applymap
    s = mapper(sev, subset=["severity"])
    mapper = getattr(s, "map", None) or s.applymap
    return mapper(stat, subset=["status"])


# =====================================================================
# LOAD AI MODELS (cached so they load once)
# =====================================================================
@st.cache_resource
def load_models():
    return {
        "vectorizer": joblib.load("models/vectorizer.pkl"),
        "type": joblib.load("models/emergency_type_model.pkl"),
        "severity": joblib.load("models/severity_model.pkl"),
        "priority": joblib.load("models/priority_model.pkl"),
        "resource": joblib.load("models/resource_model.pkl"),
    }


@st.cache_resource
def load_whisper():
    import whisper
    return whisper.load_model("base")


def analyze_text(text):
    m = load_models()
    features = m["vectorizer"].transform([text])
    values = np.array(m["resource"].predict(features)).reshape(-1)
    required = [n for n, v in zip(RESOURCE_TYPES, values) if int(v) == 1]
    return {
        "emergency_type": str(m["type"].predict(features)[0]),
        "severity": str(m["severity"].predict(features)[0]),
        "priority": int(m["priority"].predict(features)[0]),
        "required_resources": required,
    }


def transcribe(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_file.getbuffer())
        path = f.name
    try:
        return load_whisper().transcribe(path, fp16=False)["text"].strip()
    finally:
        os.remove(path)


# =====================================================================
# DATA LAYER (CSV files)
# =====================================================================
def seed_resources():
    rows = []
    units = {
        "Ambulance": [("A1", "Shivajinagar"), ("A2", "Kothrud"), ("A3", "Hadapsar"), ("A4", "Pimpri")],
        "Fire Brigade": [("F1", "Swargate"), ("F2", "Baner"), ("F3", "Hadapsar")],
        "Police": [("P1", "Shivajinagar"), ("P2", "Viman Nagar"), ("P3", "Katraj"), ("P4", "Kothrud")],
        "Rescue Team": [("R1", "Pimpri"), ("R2", "Swargate"), ("R3", "Viman Nagar")],
    }
    for rtype, items in units.items():
        for uid, loc in items:
            lat, lon = LOCATIONS[loc]
            rows.append({"resource_id": uid, "type": rtype, "name": f"{rtype} {uid}",
                         "base": loc, "lat": lat + random.uniform(-0.004, 0.004),
                         "lon": lon + random.uniform(-0.004, 0.004),
                         "available": 1, "assigned_emergency": "", "unit_status": "Idle"})
    pd.DataFrame(rows).to_csv(RESOURCE_FILE, index=False)


def load_resources():
    if not os.path.exists(RESOURCE_FILE):
        seed_resources()
    df = pd.read_csv(RESOURCE_FILE, dtype={"assigned_emergency": str, "unit_status": str})
    df["assigned_emergency"] = df["assigned_emergency"].fillna("")
    df["unit_status"] = df["unit_status"].fillna("Idle")
    return df


def save_resources(df):
    df.to_csv(RESOURCE_FILE, index=False)


def load_logs():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=LOG_COLS).to_csv(LOG_FILE, index=False)
    return pd.read_csv(LOG_FILE, dtype=str).fillna("")


def save_logs(df):
    df.to_csv(LOG_FILE, index=False)


def update_log(eid, **fields):
    df = load_logs()
    for k, v in fields.items():
        df.loc[df["emergency_id"] == eid, k] = str(v)
    save_logs(df)


def get_emergency(eid):
    df = load_logs()
    row = df[df["emergency_id"] == eid]
    return None if row.empty else row.iloc[0]


# =====================================================================
# RESOURCE RECOMMENDATION
# =====================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def recommend(types, lat, lon):
    """Nearest AVAILABLE unit for each requested resource type."""
    res = load_resources()
    out = []
    for t in types:
        pool = res[(res["type"] == t) & (res["available"] == 1)].copy()
        if pool.empty:
            out.append({"type": t, "unit_id": None, "name": f"No {t} available",
                        "distance_km": None, "eta_min": None})
            continue
        pool["dist"] = pool.apply(lambda r: haversine_km(lat, lon, r["lat"], r["lon"]) * ROAD_FACTOR, axis=1)
        best = pool.sort_values("dist").iloc[0]
        out.append({"type": t, "unit_id": best["resource_id"], "name": best["name"],
                    "distance_km": round(float(best["dist"]), 2),
                    "eta_min": max(1, round(float(best["dist"]) / AVG_SPEED_KMH * 60))})
    return out


def create_emergency(text, pred, lat, lon, loc_name, name, phone):
    logs = load_logs()
    eid = f"E{1001 + len(logs)}"
    recs = recommend(pred["required_resources"], lat, lon)
    row = {
        "emergency_id": eid, "time": datetime.now().strftime(TIME_FMT),
        "citizen_name": name, "phone": phone, "report_text": text,
        "emergency_type": pred["emergency_type"], "severity": pred["severity"],
        "priority": str(pred["priority"]),
        "required_resources": ", ".join(pred["required_resources"]),
        "lat": str(lat), "lon": str(lon), "location_name": loc_name,
        "recommended": json.dumps(recs), "assigned_units": "",
        "status": "Pending", "dispatched_time": "", "resolved_time": "",
    }
    save_logs(pd.concat([logs, pd.DataFrame([row])], ignore_index=True))
    return eid


def dispatch(eid, types):
    """Assign the nearest currently-available unit of each type."""
    em = get_emergency(eid)
    recs = recommend(types, float(em["lat"]), float(em["lon"]))
    res = load_resources()
    assigned = [u for u in em["assigned_units"].split(",") if u]
    for r in recs:
        if r["unit_id"] and r["unit_id"] not in assigned:
            idx = res["resource_id"] == r["unit_id"]
            res.loc[idx, "available"] = 0
            res.loc[idx, "assigned_emergency"] = eid
            res.loc[idx, "unit_status"] = "Assigned"
            assigned.append(r["unit_id"])
    save_resources(res)
    update_log(eid, assigned_units=",".join(assigned), status="Dispatched",
               dispatched_time=datetime.now().strftime(TIME_FMT), recommended=json.dumps(recs))
    return recs


def free_units(eid=None, unit_id=None):
    res = load_resources()
    idx = (res["assigned_emergency"] == eid) if eid else (res["resource_id"] == unit_id)
    res.loc[idx, ["available", "assigned_emergency", "unit_status"]] = [1, "", "Idle"]
    save_resources(res)


def set_status(eid, status):
    """Control-room status change."""
    if status == "Resolved":
        free_units(eid=eid)
        update_log(eid, status=status, resolved_time=datetime.now().strftime(TIME_FMT))
    else:
        update_log(eid, status=status)


def unit_update(unit_id, eid, action):
    """Resource-dashboard status change."""
    if action == "Rejected":
        em = get_emergency(eid)
        left = [u for u in em["assigned_units"].split(",") if u and u != unit_id]
        free_units(unit_id=unit_id)
        update_log(eid, assigned_units=",".join(left), status="Pending" if not left else em["status"])
    elif action == "Resolved":
        free_units(unit_id=unit_id)
        res = load_resources()
        if res[res["assigned_emergency"] == eid].empty:
            update_log(eid, status="Resolved", resolved_time=datetime.now().strftime(TIME_FMT))
    else:
        res = load_resources()
        res.loc[res["resource_id"] == unit_id, "unit_status"] = action
        save_resources(res)
        update_log(eid, status=action)


def build_map(em=None):
    pts = []
    if em is not None:
        pts.append({"lat": float(em["lat"]), "lon": float(em["lon"]), "color": "#FFE600", "size": 220})
    for r in load_resources().itertuples():
        color = RESOURCE_COLORS[r.type] if r.available == 1 else "#9AA5B4"
        pts.append({"lat": r.lat, "lon": r.lon, "color": color, "size": 90})
    return pd.DataFrame(pts)


def map_legend():
    items = [f"<span style='color:{c}'>●</span> {RESOURCE_ICONS[t]} {t}" for t, c in RESOURCE_COLORS.items()]
    st.caption("Map key:  🟡 Emergency   " + "   ".join(items) + "   <span style='color:#9AA5B4'>●</span> Busy",
               unsafe_allow_html=True)


def seed_demo_data(n=25):
    """Fake history so analytics charts look good in a demo."""
    samples = [
        ("Fire", "Critical", 1, "Ambulance, Fire Brigade, Police, Rescue Team", "Major fire in a building, people trapped"),
        ("Accident", "Critical", 1, "Ambulance, Police", "Bike accident, rider bleeding heavily"),
        ("Medical Emergency", "High", 2, "Ambulance", "Elderly person unconscious at home"),
        ("Flood", "High", 2, "Ambulance, Police, Rescue Team", "Floodwater entering houses"),
        ("Gas Leakage", "High", 2, "Ambulance, Fire Brigade", "Strong gas smell, people feeling dizzy"),
        ("Personal Safety", "Medium", 3, "Police", "Someone is following me at night"),
        ("Civic Issue", "Low", 4, "", "Streetlight not working on main road"),
        ("Prank/False Report", "Low", 5, "Police", "Fake fire report given as a joke"),
    ]
    logs = load_logs()
    start = len(logs)
    rows = []
    for i in range(n):
        s = random.choice(samples)
        loc = random.choice(list(LOCATIONS))
        t = datetime.now() - timedelta(hours=random.randint(1, 72), minutes=random.randint(0, 59))
        disp = t + timedelta(minutes=random.randint(2, 12))
        rows.append({
            "emergency_id": f"E{1001 + start + i}", "time": t.strftime(TIME_FMT), "citizen_name": "Demo",
            "phone": "", "report_text": s[4], "emergency_type": s[0], "severity": s[1], "priority": str(s[2]),
            "required_resources": s[3], "lat": str(LOCATIONS[loc][0]), "lon": str(LOCATIONS[loc][1]),
            "location_name": loc, "recommended": "[]",
            "assigned_units": ",".join(random.sample(["A1", "A2", "F1", "P1", "P2", "R1"], random.randint(1, 3))),
            "status": "Resolved", "dispatched_time": disp.strftime(TIME_FMT),
            "resolved_time": (disp + timedelta(minutes=30)).strftime(TIME_FMT),
        })
    save_logs(pd.concat([logs, pd.DataFrame(rows)], ignore_index=True))


# =====================================================================
# SESSION STATE
# =====================================================================
for key, default in {"auth": None, "sos_active": False, "last_eid": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================================
# LOGIN PAGE
# =====================================================================
def page_login():
    hero("🚨 RescueNet", "Press SOS and speak. AI understands the emergency, "
                         "and the control room sends the nearest help.",
         chips=["🎙️ Voice SOS", "🧠 AI triage", "📍 Nearest unit", "📡 Live tracking"])
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("### Sign in")
            t1, t2, t3, t4 = st.tabs(["🆘 Citizen", "🎛️ Control Room", "🚑 Response Unit", "📊 Admin"])

            with t1:
                st.write("No account needed. Add your details so responders can reach you.")
                name = st.text_input("Your name", key="c_name")
                phone = st.text_input("Phone number", key="c_phone")
                if st.button("Continue to SOS", type="primary", key="login_citizen"):
                    st.session_state.auth = {"role": "Citizen", "name": name or "Anonymous", "phone": phone}
                    st.rerun()

            with t2:
                u = st.text_input("Username", key="cr_u")
                p = st.text_input("Password", type="password", key="cr_p")
                if st.button("Open control room", type="primary", key="login_control"):
                    if CONTROL_USERS.get(u) == p and p:
                        st.session_state.auth = {"role": "Control Room", "name": u}
                        st.rerun()
                    else:
                        st.error("Wrong username or password.")

            with t3:
                units = load_resources()
                unit = st.selectbox("Select your unit", units["resource_id"] + " — " + units["name"], key="ru_unit")
                p = st.text_input("Password", type="password", key="ru_p")
                if st.button("Open unit dashboard", type="primary", key="login_unit"):
                    if p == UNIT_PASSWORD:
                        st.session_state.auth = {"role": "Resource", "name": unit.split(" — ")[1],
                                                 "unit_id": unit.split(" — ")[0]}
                        st.rerun()
                    else:
                        st.error("Wrong password.")

            with t4:
                u = st.text_input("Username", key="ad_u")
                p = st.text_input("Password", type="password", key="ad_p")
                if st.button("Open admin panel", type="primary", key="login_admin"):
                    if ADMIN_USERS.get(u) == p and p:
                        st.session_state.auth = {"role": "Admin", "name": u}
                        st.rerun()
                    else:
                        st.error("Wrong username or password.")

        with st.expander("Demo logins"):
            st.markdown("- Control Room: `control` / `control123`\n- Admin: `admin` / `admin123`\n"
                        f"- Response unit: pick any unit, password `{UNIT_PASSWORD}`")

    with right:
        st.markdown("### How RescueNet works")
        steps = [("Press SOS and speak", "Or type if you cannot talk. Whisper turns your voice into text."),
                 ("AI reads the report", "It finds the emergency type, severity and dispatch priority."),
                 ("Resources are chosen", "Ambulance, fire brigade, police and rescue team, as needed."),
                 ("Nearest unit is sent", "The control room approves and the unit heads to you."),
                 ("Track help live", "Watch the status move from Pending to Resolved.")]
        for i, (t, d) in enumerate(steps, 1):
            st.markdown(f"<div class='how'><div class='n'>{i}</div><div><b>{t}</b><br>"
                        f"<span style='color:#9AA9C2'>{d}</span></div></div>", unsafe_allow_html=True)


# =====================================================================
# CITIZEN SOS PAGE
# =====================================================================
QUICK_REPORTS = {
    "🔥 Fire": "Fire in a building, heavy smoke, people trapped inside",
    "🚗 Accident": "Road accident, rider bleeding heavily and not moving",
    "🩺 Medical": "Person unconscious and not breathing properly",
    "🌊 Flood": "Floodwater entering houses, people stuck on the ground floor",
}


def show_result(eid):
    em = get_emergency(eid)
    if em is None:
        return
    st.markdown(f"### Your request {eid}")
    severity_banner(em["severity"], em["emergency_type"], em["priority"])

    with st.container(border=True):
        st.markdown("**Help needed**")
        st.markdown(resource_chips(em["required_resources"].split(",")), unsafe_allow_html=True)
        st.markdown("**Live status**")
        timeline(em["status"])
        st.button("🔄 Refresh status", key="refresh_citizen")

    # ---- Bayesian severity confidence (uses the saved report text of this emergency)
    user_text = em["report_text"]
    b_col, a_col = st.columns(2, gap="large")

    with b_col:
        with st.container(border=True):
            st.subheader("Emergency Severity Assessment")
            st.caption("Severity likelihood based on reported emergency details")
            bayes_result = bayesian_severity_confidence(user_text)

            for severity, probability in bayes_result["probabilities"].items():
                st.write(f"{severity}: {probability * 100:.1f}%")
                st.progress(probability)

            st.success(
                f"Assessed Emergency Severity: {bayes_result['final_severity']} "
                f"(Assessment confidence: {bayes_result['confidence'] * 100:.1f}%)"
            )
            st.info(
                "Key details identified from the emergency report: "
                + (", ".join(bayes_result["matched_evidence"]) or "No key details detected")
            )

    # ---- A* emergency route recommendation
    with a_col:
        with st.container(border=True):
            st.subheader("Nearest Rescue Unit & Fastest Route")
            route_result = astar_route("A1", "EMERGENCY")

            if route_result:
                st.write("Fastest response route:")
                st.write(" → ".join(route_result["path"]))

                col1, col2 = st.columns(2)
                col1.metric("Distance to emergency location", f"{route_result['distance']} km")
                col2.metric("Estimated arrival time", f"{route_result['time_minutes']} min")

                st.success("Fastest available route selected for emergency response.")
            else:
                st.error("No route found to emergency location.")

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Nearest available help**")
        recs = json.loads(em["recommended"] or "[]")
        if not recs:
            st.caption("No units needed for this report.")
        for r in recs:
            unit_card(r)
    with right:
        st.map(build_map(em), latitude="lat", longitude="lon", color="color", size="size", height=300)
        map_legend()


def page_citizen():
    hero("🆘 Report an emergency", "Stay calm. Tell us what is happening and where you are.")

    if not st.session_state.sos_active:
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            st.markdown("<div class='sos-wrap'><h2>Are you in an emergency?</h2>"
                        "<p>Press the button to start a report.</p></div>", unsafe_allow_html=True)
            if st.button("SOS", key="sos_btn"):
                st.session_state.sos_active = True
                st.rerun()
        if st.session_state.last_eid:
            st.divider()
            show_result(st.session_state.last_eid)
        return

    with st.container(border=True):
        step_title(1, "What happened?")
        method = st.radio("How do you want to report?", ["🎤 Speak", "⌨️ Type"], horizontal=True)
        if method == "🎤 Speak":
            audio = st.audio_input("Press the microphone and describe the emergency")
            if audio is not None and st.button("Convert voice to text", key="btn_transcribe"):
                with st.spinner("Listening..."):
                    try:
                        st.session_state["report_text"] = transcribe(audio)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not convert voice to text ({e}). Please type your report instead.")
        else:
            st.caption("Quick fill:")
            cols = st.columns(len(QUICK_REPORTS))
            for col, (label, text) in zip(cols, QUICK_REPORTS.items()):
                if col.button(label, key=f"q_{label}", use_container_width=True):
                    st.session_state["report_text"] = text
        report = st.text_area("Emergency report (you can edit this)", key="report_text", height=110,
                              placeholder="Example: Fire in a building, people trapped inside")

    with st.container(border=True):
        step_title(2, "Where are you?")
        loc_mode = st.radio("Location method", ["📍 Auto-detect (simulated GPS)", "✍️ Enter latitude / longitude"],
                            horizontal=True)
        if loc_mode.startswith("📍"):
            loc_name = st.selectbox("Detected area", list(LOCATIONS))
            lat, lon = LOCATIONS[loc_name]
            st.caption(f"Coordinates: {lat:.4f}, {lon:.4f}")
        else:
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", value=18.5204, format="%.5f")
            lon = c2.number_input("Longitude", value=73.8567, format="%.5f")
            loc_name = "Manual location"

    with st.container(border=True):
        step_title(3, "Send for help")
        if st.button("🚨 Analyze and request help", type="primary", key="btn_analyze", use_container_width=True):
            if not report.strip():
                st.warning("Please speak or type the emergency first.")
            else:
                with st.spinner("AI is analyzing your emergency..."):
                    try:
                        pred = analyze_text(report.strip())
                    except Exception as e:
                        st.error(f"Could not load the AI models ({e}). Check the 'models' folder.")
                        return
                    auth = st.session_state.auth
                    st.session_state.last_eid = create_emergency(
                        report.strip(), pred, lat, lon, loc_name, auth.get("name", ""), auth.get("phone", ""))
                st.success("Help requested. The control room has been notified.")

    if st.session_state.last_eid:
        st.divider()
        show_result(st.session_state.last_eid)

    if st.button("Start a new report", key="btn_new"):
        st.session_state.sos_active = False
        st.session_state.last_eid = None
        st.session_state.pop("report_text", None)
        st.rerun()


# =====================================================================
# CONTROL ROOM
# =====================================================================
def page_control():
    hero("🎛️ Control room", "Critical cases appear first. Approve or change the units, then dispatch.",
         chips=["Priority queue", "Nearest unit", "Live status"])
    logs = load_logs()
    if logs.empty:
        st.info("No emergencies yet. Reports from citizens will appear here.")
        return
    logs["priority_n"] = pd.to_numeric(logs["priority"], errors="coerce").fillna(5)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Total emergencies", len(logs), "#5AA2FF", "📋")
    with k2:
        kpi("Critical cases", int((logs["severity"] == "Critical").sum()), "#FF4D5E", "🚨")
    with k3:
        kpi("Pending", int((logs["status"] == "Pending").sum()), "#FFB02E", "⏳")
    with k4:
        kpi("Active", int(logs["status"].isin(["Dispatched", "Accepted", "En Route", "Reached"]).sum()), "#5AA2FF", "🚑")

    with st.container(border=True):
        st.markdown("**Emergency queue** (highest priority first)")
        f1, f2, f3 = st.columns(3)
        ftype = f1.selectbox("Emergency type", ["All"] + sorted(logs["emergency_type"].unique()))
        fsev = f2.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"])
        fstat = f3.selectbox("Status", ["All"] + STATUS_FLOW)
        view = logs.copy()
        if ftype != "All":
            view = view[view["emergency_type"] == ftype]
        if fsev != "All":
            view = view[view["severity"] == fsev]
        if fstat != "All":
            view = view[view["status"] == fstat]
        view = view.sort_values(["priority_n", "time"])
        table = view[["emergency_id", "time", "emergency_type", "severity", "priority",
                      "location_name", "required_resources", "assigned_units", "status"]]
        st.dataframe(style_table(table), use_container_width=True, hide_index=True)

    if view.empty:
        st.warning("No emergencies match these filters.")
        return

    with st.container(border=True):
        eid = st.selectbox("Open emergency", view["emergency_id"])
        em = get_emergency(eid)
        severity_banner(em["severity"], em["emergency_type"], em["priority"])
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown(f"**Report:** “{em['report_text']}”")
            st.caption(f"Reporter: {em['citizen_name']} {em['phone']} • Location: {em['location_name']} "
                       f"({float(em['lat']):.4f}, {float(em['lon']):.4f}) • Reported {em['time']}")
            timeline(em["status"])
            required = [r.strip() for r in em["required_resources"].split(",") if r.strip()]
            chosen = st.multiselect("Resources to dispatch (change the AI suggestion if needed)",
                                    RESOURCE_TYPES, default=required, key=f"ms_{eid}")
            st.markdown("**Nearest available units right now**")
            for r in recommend(chosen, float(em["lat"]), float(em["lon"])):
                unit_card(r)
            if em["assigned_units"]:
                st.caption(f"Already assigned: {em['assigned_units']}")

            b1, b2 = st.columns(2)
            if b1.button("✅ Dispatch resources", type="primary", key=f"disp_{eid}", disabled=not chosen,
                         use_container_width=True):
                dispatch(eid, chosen)
                st.rerun()
            new_status = b2.selectbox("Update status", STATUS_FLOW, index=STATUS_FLOW.index(em["status"]),
                                      key=f"st_{eid}", label_visibility="collapsed")
            if new_status != em["status"]:
                set_status(eid, new_status)
                st.rerun()

        with right:
            st.map(build_map(em), latitude="lat", longitude="lon", color="color", size="size", height=360)
            map_legend()


# =====================================================================
# RESOURCE DASHBOARD
# =====================================================================
def page_resource():
    unit_id = st.session_state.auth["unit_id"]
    res = load_resources()
    unit = res[res["resource_id"] == unit_id].iloc[0]
    hero(f"{RESOURCE_ICONS[unit['type']]} {unit['name']}",
         "Accept the case, update your status, and go back to available when done.",
         chips=[unit["type"], f"Base: {unit['base']}"])

    assigned = unit["assigned_emergency"]
    c1, c2 = st.columns(2)
    with c1:
        kpi("Availability", "Available" if unit["available"] == 1 else "Busy",
            "#34D399" if unit["available"] == 1 else "#FF4D5E", "🟢" if unit["available"] == 1 else "🔴")
    with c2:
        kpi("Current status", unit["unit_status"], RESOURCE_COLORS[unit["type"]], RESOURCE_ICONS[unit["type"]])

    if not assigned:
        with st.container(border=True):
            avail = st.toggle("I am available for new emergencies", value=bool(unit["available"] == 1))
            if avail != bool(unit["available"] == 1):
                res.loc[res["resource_id"] == unit_id, "available"] = int(avail)
                save_resources(res)
                st.rerun()
            st.info("No emergency assigned yet. Stay ready; the control room will dispatch you.")
            st.button("🔄 Check for new assignment", key="refresh_unit")
        return

    em = get_emergency(assigned)
    dist = haversine_km(float(em["lat"]), float(em["lon"]), unit["lat"], unit["lon"]) * ROAD_FACTOR
    eta = max(1, round(dist / AVG_SPEED_KMH * 60))
    with st.container(border=True):
        st.markdown(f"### Assigned emergency {assigned}")
        severity_banner(em["severity"], em["emergency_type"], em["priority"])
        st.markdown(f"**“{em['report_text']}”**")
        d1, d2 = st.columns(2)
        d1.metric("Distance", f"{dist:.1f} km")
        d2.metric("Estimated arrival", f"{eta} min")
        st.caption(f"Location: {em['location_name']} • Reporter: {em['citizen_name']} {em['phone']}")
        timeline(unit["unit_status"])

        status = unit["unit_status"]
        b1, b2, b3, b4, b5 = st.columns(5)
        if b1.button("Accept", disabled=status != "Assigned", key="u_accept", type="primary", use_container_width=True):
            unit_update(unit_id, assigned, "Accepted"); st.rerun()
        if b2.button("Reject", disabled=status != "Assigned", key="u_reject", use_container_width=True):
            unit_update(unit_id, assigned, "Rejected"); st.rerun()
        if b3.button("En Route", disabled=status != "Accepted", key="u_enroute", use_container_width=True):
            unit_update(unit_id, assigned, "En Route"); st.rerun()
        if b4.button("Reached", disabled=status != "En Route", key="u_reached", use_container_width=True):
            unit_update(unit_id, assigned, "Reached"); st.rerun()
        if b5.button("Resolved", disabled=status != "Reached", key="u_resolved", use_container_width=True):
            unit_update(unit_id, assigned, "Resolved"); st.rerun()

    pts = pd.DataFrame([
        {"lat": float(em["lat"]), "lon": float(em["lon"]), "color": "#FFE600", "size": 220},
        {"lat": unit["lat"], "lon": unit["lon"], "color": RESOURCE_COLORS[unit["type"]], "size": 140}])
    st.map(pts, latitude="lat", longitude="lon", color="color", size="size", height=320)
    st.caption("🟡 Emergency   ● Your unit")


# =====================================================================
# ADMIN ANALYTICS
# =====================================================================
def page_analytics():
    hero("📊 Admin analytics", "How emergencies are distributed and how resources are used.")
    logs = load_logs()

    if st.button("Load demo data", key="seed"):
        seed_demo_data(); st.rerun()
    if logs.empty:
        st.info("No data yet. Submit emergencies or press 'Load demo data'.")
        return

    logs["priority_n"] = pd.to_numeric(logs["priority"], errors="coerce")
    logs["t"] = pd.to_datetime(logs["time"], errors="coerce")
    logs["d"] = pd.to_datetime(logs["dispatched_time"], errors="coerce")
    resp = ((logs["d"] - logs["t"]).dt.total_seconds() / 60).dropna()

    k = st.columns(5)
    vals = [("Total emergencies", len(logs), "#5AA2FF", "📋"),
            ("Critical cases", int((logs["severity"] == "Critical").sum()), "#FF4D5E", "🚨"),
            ("Resolved", int((logs["status"] == "Resolved").sum()), "#34D399", "✅"),
            ("Average priority", f"{logs['priority_n'].mean():.1f}", "#FFB02E", "🎯"),
            ("Avg. dispatch time", f"{resp.mean():.1f} min" if len(resp) else "—", "#5AA2FF", "⏱️")]
    for col, v in zip(k, vals):
        with col:
            kpi(*v)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Emergencies by type**")
            bar_chart(logs["emergency_type"].value_counts())
    with c2:
        with st.container(border=True):
            st.markdown("**Severity distribution**")
            order = ["Critical", "High", "Medium", "Low"]
            bar_chart(logs["severity"].value_counts().reindex(order).fillna(0), SEVERITY_COLORS)

    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            st.markdown("**Emergencies over time**")
            trend = logs.dropna(subset=["t"]).set_index("t").resample("D").size().rename("count").reset_index()
            area = (alt.Chart(trend).mark_area(opacity=.25, color="#FF6B7A", line={"color": "#FF6B7A"})
                    .encode(x=alt.X("t:T", title=None), y=alt.Y("count:Q", title=None),
                            tooltip=["t:T", "count:Q"]).properties(height=280))
            st.altair_chart(area, use_container_width=True)
    with c4:
        with st.container(border=True):
            st.markdown("**Resource utilization**")
            units = logs["assigned_units"].str.split(",").explode()
            units = units[units != ""]
            if units.empty:
                st.caption("No resources dispatched yet.")
            else:
                fleet = load_resources().set_index("resource_id")["type"]
                cmap = {u: RESOURCE_COLORS[fleet[u]] for u in units.unique() if u in fleet.index}
                bar_chart(units.value_counts(), cmap)

    with st.container(border=True):
        st.markdown("**Fleet status**")
        st.dataframe(load_resources()[["resource_id", "type", "base", "available", "assigned_emergency", "unit_status"]],
                     use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("**Recent emergency reports**")
        recent = logs.sort_values("time", ascending=False).head(15)[
            ["emergency_id", "time", "emergency_type", "severity", "priority", "location_name", "status"]]
        st.dataframe(style_table(recent), use_container_width=True, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download emergency report (CSV)", logs[LOG_COLS].to_csv(index=False),
                       file_name="emergency_logs.csv", mime="text/csv", use_container_width=True)
    if d2.button("Reset all units to available", key="reset_units", use_container_width=True):
        seed_resources(); st.rerun()


# =====================================================================
# MAIN ROUTER
# =====================================================================
auth = st.session_state.auth
if not auth:
    page_login()
else:
    with st.sidebar:
        st.markdown("<div class='side-brand'>🚨 RescueNet</div>", unsafe_allow_html=True)
        st.markdown(f"<span class='side-pill'>{auth['role']}</span>", unsafe_allow_html=True)
        st.write("")
        st.write(f"Signed in as **{auth['name']}**")
        st.divider()
        if st.button("Log out", key="logout", use_container_width=True):
            st.session_state.auth = None
            st.session_state.sos_active = False
            st.session_state.last_eid = None
            st.rerun()

    role = auth["role"]
    if role == "Citizen":
        page_citizen()
    elif role == "Resource":
        page_resource()
    elif role == "Control Room":
        page_control()
    else:  # Admin sees everything
        t1, t2, t3, t4 = st.tabs(["🆘 Citizen SOS", "🎛️ Control Room", "🚑 Resources", "📊 Analytics"])
        with t1:
            page_citizen()
        with t2:
            page_control()
        with t3:
            st.dataframe(load_resources(), use_container_width=True, hide_index=True)
        with t4:
            page_analytics()