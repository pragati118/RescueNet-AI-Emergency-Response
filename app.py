"""
RescueNet - AI-Based Emergency Reporting and Resource Allocation System
Run with:  streamlit run app.py
"""
import os
import re
import time
import html
import json
import math
import random
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Feature 1: browser location (pip install streamlit-geolocation)
try:
    from streamlit_geolocation import streamlit_geolocation
    GEO_OK = True
except ImportError:
    GEO_OK = False

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
            "status", "dispatched_time", "resolved_time",
            "report_count", "reports", "last_report_time"]

# ---- Feature 3: incident grouping rules
GROUP_RADIUS_KM = 1.0          # reports within 1 km can be the same incident
GROUP_WINDOW_MIN = 15          # ...and within 15 minutes of the incident's latest report
SIMILAR_TYPES = [{"Fire", "Gas Leakage"}, {"Accident", "Medical Emergency"}]
NO_GROUP_TYPES = {"Prank/False Report", "Non-Emergency"}
STATE_MAP = {"Pending": "Active", "Dispatched": "Assigned", "Accepted": "Assigned",
             "En Route": "In Progress", "Reached": "In Progress", "Resolved": "Resolved"}
STOPWORDS = {"the", "and", "are", "was", "has", "have", "with", "from", "near", "there", "this", "that",
             "into", "some", "very", "please", "help", "need", "here", "its", "our", "his", "her"}
EMERGENCY_KEYWORDS = {"fire", "smoke", "trapped", "burning", "flames", "blast", "explosion", "bleeding",
                      "blood", "unconscious", "accident", "crash", "collision", "injured", "flood",
                      "water", "drowning", "gas", "leak", "collapse", "collapsed", "heart", "chest",
                      "building", "bike", "car", "truck"}

# ---- Feature 1: manual place lookup + default demo location
PLACES = {**LOCATIONS, "Pune": (18.5204, 73.8567), "Mumbai": (19.0760, 72.8777),
          "Nashik": (19.9975, 73.7898), "Nagpur": (21.1458, 79.0882),
          "Delhi": (28.6139, 77.2090), "Bengaluru": (12.9716, 77.5946)}
DEFAULT_LOCATION = ("Shivajinagar (default demo location)",) + LOCATIONS["Shivajinagar"]

# ---- Feature 2: helplines, sample hospitals, safety guidance
HELPLINES = [("Ambulance", "108", "#34D399", "🚑"), ("Police", "100", "#5AA2FF", "🚓"),
             ("Fire Brigade", "101", "#FF4D5E", "🚒"), ("National Emergency", "112", "#FFB02E", "🆘")]

# Sample data for demonstration only (fictional names and numbers)
HOSPITALS = [
    {"name": "City General Hospital", "lat": 18.5330, "lon": 73.8440, "phone": "020-5550-0101", "facilities": "24x7 ER, ICU, Trauma"},
    {"name": "Lifeline Trauma Centre", "lat": 18.5020, "lon": 73.8600, "phone": "020-5550-0102", "facilities": "Trauma, Ambulance, Blood bank"},
    {"name": "Green Cross Heart Institute", "lat": 18.5120, "lon": 73.8120, "phone": "020-5550-0103", "facilities": "Cardiac ICU, Cath lab, ER"},
    {"name": "Riverside Community Hospital", "lat": 18.5600, "lon": 73.7900, "phone": "020-5550-0104", "facilities": "ER, Maternity, Pediatrics"},
    {"name": "Metro Burn & Emergency Care", "lat": 18.5100, "lon": 73.9200, "phone": "020-5550-0105", "facilities": "Burn unit, ER, ICU"},
    {"name": "Northgate Multispeciality Hospital", "lat": 18.6250, "lon": 73.8050, "phone": "020-5550-0106", "facilities": "ER, ICU, Surgery"},
    {"name": "Eastside Medical Centre", "lat": 18.5650, "lon": 73.9100, "phone": "020-5550-0107", "facilities": "ER, Orthopedics, Ambulance"},
    {"name": "Southview Emergency Hospital", "lat": 18.4600, "lon": 73.8650, "phone": "020-5550-0108", "facilities": "24x7 ER, ICU, Dialysis"},
]

SAFETY_GUIDE = {
    "cardiac": ("🫀 Suspected cardiac emergency", [
        "Call an ambulance (108) immediately.",
        "Keep the person calm and comfortable, and loosen tight clothing.",
        "If the person is unconscious and not breathing, start CPR only if you are trained, "
        "and follow the dispatcher's instructions."]),
    "bleeding": ("🩸 Bleeding", [
        "Apply firm, steady pressure on the wound with a clean cloth.",
        "If the cloth soaks through, add another on top instead of removing it.",
        "Call an ambulance (108) if the bleeding is heavy or does not stop."]),
    "fire": ("🔥 Fire", [
        "Move away from the smoke and stay low to the ground.",
        "Do not use lifts. Use the stairs.",
        "Call the fire brigade (101) as soon as you are in a safe place."]),
    "accident": ("🚗 Accident", [
        "Do not move a seriously injured person unless there is immediate danger.",
        "Keep other people and traffic away from the injured person.",
        "Call an ambulance (108) and the police (100)."]),
    "drowning": ("🌊 Drowning / flood water", [
        "Call emergency services (112 or 108) right away.",
        "Do not enter the water unless you are trained. Throw a rope or floating object instead.",
        "If the person is out of the water and not breathing, give CPR only if you are trained."]),
    "general": ("🛡️ Stay safe while help is coming", [
        "Stay calm and move to a safe place if you can.",
        "Keep your phone nearby and answer calls from the control room.",
        "Do not put yourself in danger to help others."]),
}

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

/* facts grid, helplines */
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 6px 0 12px; }
.fact { background: var(--panel2); border: 1px solid var(--line); border-radius: 12px; padding: 9px 14px; }
.fact .k { color: var(--muted); font-size: .76rem; font-weight: 600; }
.fact .v { color: #fff; font-weight: 800; }
.helpline { text-align: center; background: var(--panel2); border: 1px solid var(--line); border-top: 5px solid; border-radius: 14px; padding: 12px 8px; }
.helpline .nm { color: var(--muted); font-weight: 700; font-size: .9rem; }
.helpline .num { font-size: 2.1rem; font-weight: 800; line-height: 1.15; }
.helpline a { text-decoration: none; }
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
    df = pd.read_csv(LOG_FILE, dtype=str).fillna("")
    for c in LOG_COLS:                      # older CSV files get the new columns automatically
        if c not in df.columns:
            df[c] = "1" if c == "report_count" else ""
    return df


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
    now = datetime.now().strftime(TIME_FMT)
    row = {
        "report_count": "1", "last_report_time": now,
        "reports": json.dumps([{"time": now, "name": name, "phone": phone, "text": text}]),
        "emergency_id": eid, "time": now,
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


# ---------------------------------------------------------------------
# Feature 3: duplicate SOS detection and incident grouping
# ---------------------------------------------------------------------
def n_reports(em):
    try:
        return max(1, int(em["report_count"] or 1))
    except ValueError:
        return 1


def get_reports(em):
    try:
        reps = json.loads(em["reports"] or "[]")
    except ValueError:
        reps = []
    if not reps:   # incidents created before grouping existed
        reps = [{"time": em["time"], "name": em["citizen_name"], "phone": em["phone"], "text": em["report_text"]}]
    return reps


def incident_state(status):
    return STATE_MAP.get(status, "Active")


def _tokens(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def descriptions_similar(a, b):
    """Similar if they share an emergency keyword (fire, smoke, trapped...) or enough common words."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return bool(ta & tb & EMERGENCY_KEYWORDS) or len(ta & tb) / len(ta | tb) >= 0.25


def types_similar(a, b):
    return a == b or any({a, b} <= s for s in SIMILAR_TYPES)


def find_matching_incident(pred, text, lat, lon):
    """Return (incident_id, distance_km) of the closest ACTIVE incident that matches, else None."""
    now = datetime.now()
    best = None
    for _, row in load_logs().iterrows():
        if row["status"] == "Resolved" or not types_similar(row["emergency_type"], pred["emergency_type"]):
            continue
        try:
            dist = haversine_km(lat, lon, float(row["lat"]), float(row["lon"]))
            last = datetime.strptime(row["last_report_time"] or row["time"], TIME_FMT)
        except ValueError:
            continue
        if dist > GROUP_RADIUS_KM or abs((now - last).total_seconds()) / 60 > GROUP_WINDOW_MIN:
            continue
        if not any(descriptions_similar(text, r["text"]) for r in get_reports(row)):
            continue
        if best is None or dist < best[1]:
            best = (row["emergency_id"], dist)
    return best


def add_report_to_incident(eid, text, pred, name, phone):
    em = get_emergency(eid)
    reps = get_reports(em)
    now = datetime.now().strftime(TIME_FMT)
    reps.append({"time": now, "name": name, "phone": phone, "text": text})
    fields = {"reports": json.dumps(reps), "report_count": len(reps), "last_report_time": now}
    if int(pred["priority"]) < int(em["priority"] or 5):        # new report is more urgent -> escalate
        fields.update(severity=pred["severity"], priority=pred["priority"])
    required = [r.strip() for r in em["required_resources"].split(",") if r.strip()]
    merged = required + [r for r in pred["required_resources"] if r not in required]
    if merged != required:
        fields["required_resources"] = ", ".join(merged)
        if not em["assigned_units"]:
            fields["recommended"] = json.dumps(recommend(merged, float(em["lat"]), float(em["lon"])))
    update_log(eid, **fields)


def submit_sos(text, pred, lat, lon, loc_name, name, phone):
    """Add to an existing incident if it matches, otherwise create a new one. Returns (incident_id, merged)."""
    if pred["emergency_type"] not in NO_GROUP_TYPES:
        match = find_matching_incident(pred, text, lat, lon)
        if match:
            add_report_to_incident(match[0], text, pred, name, phone)
            return match[0], True
    return create_emergency(text, pred, lat, lon, loc_name, name, phone), False


def dispatch(eid, types):
    """Assign the nearest currently-available unit of each type."""
    em = get_emergency(eid)
    res = load_resources()
    assigned = [u for u in em["assigned_units"].split(",") if u]
    # one dispatch per resource type per incident: skip types that already have a unit assigned
    fleet = res.set_index("resource_id")["type"]
    done_types = {fleet[u] for u in assigned if u in fleet.index}
    types = [x for x in types if x not in done_types]
    if not types:
        return []
    recs = recommend(types, float(em["lat"]), float(em["lon"]))
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
for key, default in {"auth": None, "sos_active": False, "last_eid": None, "last_merged": False}.items():
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


def incident_facts(em):
    assigned = ", ".join(u for u in em["assigned_units"].split(",") if u) or "Not assigned yet"
    items = [("Incident ID", em["emergency_id"]), ("Emergency type", em["emergency_type"]),
             ("Severity", em["severity"]), ("Priority", f"Priority {em['priority']}"),
             ("Location", f"{em['location_name']} ({float(em['lat']):.4f}, {float(em['lon']):.4f})"),
             ("Total SOS reports", n_reports(em)), ("First report", em["time"]),
             ("Incident status", incident_state(em["status"])), ("Assigned unit", assigned),
             ("Required resources", em["required_resources"] or "None")]
    cells = "".join(f"<div class='fact'><div class='k'>{k}</div><div class='v'>{html.escape(str(v))}</div></div>"
                    for k, v in items)
    return f"<div class='facts'>{cells}</div>"


def lookup_place(query):
    q = query.strip().lower()
    if not q:
        return None
    for name, coords in PLACES.items():
        if q == name.lower() or q in name.lower() or name.lower() in q:
            return name, coords
    try:                                   # optional online lookup if geopy is installed
        from geopy.geocoders import Nominatim
        g = Nominatim(user_agent="rescuenet-demo", timeout=5).geocode(query)
        if g:
            return query.title(), (g.latitude, g.longitude)
    except Exception:
        pass
    return None


def reverse_geocode(lat, lon):
    """Area, city and state for a coordinate using OpenStreetMap Nominatim. Returns None on ANY failure."""
    try:
        from geopy.geocoders import Nominatim
        place = Nominatim(user_agent="RescueNet", timeout=6).reverse(
            (lat, lon), language="en", addressdetails=True, zoom=16)
        if not place:
            return None
        addr = place.raw.get("address") or {}

        def first(*keys):
            return next((addr[k] for k in keys if addr.get(k)), "")

        parts = []
        for part in (first("suburb", "neighbourhood", "quarter", "city_district", "residential", "hamlet", "village"),
                     first("city", "town", "municipality", "county", "state_district"),
                     first("state")):
            if part and part not in parts:
                parts.append(part)
        return ", ".join(parts) or ", ".join(s.strip() for s in place.address.split(",")[:3]) or None
    except Exception:          # no internet, geopy missing, rate limit... the app must keep working
        return None


def place_label(lat, lon):
    """Cached reverse geocode, so Streamlit reruns do not call the service again and again."""
    cache = st.session_state.setdefault("place_cache", {})
    key = (round(lat, 4), round(lon, 4))
    name, tried_at = cache.get(key, (None, 0))
    if name is None and time.time() - tried_at > 60:      # retry a failed lookup after 1 minute
        with st.spinner("Finding your area name..."):
            name = reverse_geocode(lat, lon)
        cache[key] = (name, time.time())
    return name


def location_picker():
    """Feature 1. Returns (lat, lon, location_name). GPS first, manual fallback, demo default last."""
    ss = st.session_state
    mode = st.radio("Location method", ["📍 Current location (GPS)", "🏙️ City / area name",
                                        "✍️ Latitude / longitude"], horizontal=True, key="loc_mode")
    default_name, default_lat, default_lon = DEFAULT_LOCATION

    if mode.startswith("📍"):
        if st.button("📍 Use My Current Location", key="btn_geo"):
            ss["geo_requested"] = True
        if ss.get("geo_requested"):
            if not GEO_OK:
                st.warning("The 'streamlit-geolocation' package is not installed (pip install streamlit-geolocation). "
                           "Please use a manual option above.")
            else:
                st.caption("Click the location icon below and choose **Allow** when your browser asks for permission.")
                loc = streamlit_geolocation()
                if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
                    ss["geo_lat"], ss["geo_lon"] = float(loc["latitude"]), float(loc["longitude"])
        if ss.get("geo_lat") is not None:
            lat, lon = ss["geo_lat"], ss["geo_lon"]
            area = place_label(lat, lon)
            if area:
                st.success(f"**Current location detected: {area}**  \nLatitude: {lat:.5f}, Longitude: {lon:.5f}")
            else:   # lookup failed or offline: coordinates only
                st.success(f"Current location detected: Latitude {lat:.5f}, Longitude {lon:.5f}")
            return lat, lon, area or "Current location (GPS)"
        if ss.get("geo_requested"):
            st.warning("Location is not available yet. If permission was denied or blocked, choose "
                       "“City / area name” or “Latitude / longitude” above.")
        else:
            st.info("Press the button to detect your location, or choose a manual option. "
                    "A demo location is used only if nothing is provided.")

    elif mode.startswith("🏙️"):
        query = st.text_input("City or area name", placeholder="Example: Kothrud, Pune")
        if query.strip():
            found = lookup_place(query)
            if found:
                name, (lat, lon) = found
                st.success(f"Using {name}: Latitude {lat:.5f}, Longitude {lon:.5f}")
                return lat, lon, name
            st.warning("Could not find that place. Try another name or enter latitude / longitude.")

    else:
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=None, min_value=-90.0, max_value=90.0, format="%.5f", placeholder="18.52040")
        lon = c2.number_input("Longitude", value=None, min_value=-180.0, max_value=180.0, format="%.5f", placeholder="73.85670")
        if lat is not None and lon is not None:
            area = place_label(lat, lon)
            st.success((f"**Using manual location: {area}**  \n" if area else "Using manual location  \n")
                       + f"Latitude: {lat:.5f}, Longitude: {lon:.5f}")
            return lat, lon, area or "Manual location"

    st.caption(f"Nothing provided yet, so the demo location will be used: {default_name}.")
    return default_lat, default_lon, default_name


def safety_guidance_keys(etype, text):
    t = text.lower()
    keys = []
    if any(k in t for k in ["heart", "cardiac", "chest pain", "not breathing"]) or \
            (etype == "Medical Emergency" and any(k in t for k in ["unconscious", "collapsed"])):
        keys.append("cardiac")
    if "bleed" in t or "blood" in t:
        keys.append("bleeding")
    if etype == "Fire" or "fire" in t or "smoke" in t:
        keys.append("fire")
    if etype == "Accident" or "accident" in t or "crash" in t:
        keys.append("accident")
    if "drown" in t or etype == "Flood":
        keys.append("drowning")
    return keys or ["general"]


def nearest_hospitals(lat, lon, n=5):
    rows = []
    for h in HOSPITALS:
        d = haversine_km(lat, lon, h["lat"], h["lon"]) * ROAD_FACTOR
        rows.append({**h, "distance_km": round(d, 1), "eta_min": max(1, round(d / AVG_SPEED_KMH * 60))})
    rows.sort(key=lambda r: r["distance_km"])
    return rows[:n]


def hospital_map(lat, lon, rows, height=380, victim_label="📍 Emergency location"):
    try:
        import pydeck as pdk
    except ImportError:
        pts = [{"lat": lat, "lon": lon, "color": "#FFE600", "size": 220}] + \
              [{"lat": r["lat"], "lon": r["lon"], "color": "#FFFFFF", "size": 120} for r in rows]
        st.map(pd.DataFrame(pts), latitude="lat", longitude="lon", color="color", size="size", height=height)
        return
    hosp = pd.DataFrame([{"lat": r["lat"], "lon": r["lon"], "color": [52, 211, 153, 230],
                          "label": f"🏥 {r['name']}  |  📞 {r['phone']}  |  {r['distance_km']} km"} for r in rows])
    victim = pd.DataFrame([{"lat": lat, "lon": lon, "color": [255, 230, 0, 255], "label": victim_label}])
    layers = [pdk.Layer("ScatterplotLayer", data=d, get_position="[lon, lat]", get_fill_color="color",
                        get_radius=r, radius_min_pixels=m, pickable=True)
              for d, r, m in [(hosp, 200, 8), (victim, 300, 11)]]
    far = max(r["distance_km"] for r in rows)
    zoom = 12 if far < 6 else 11 if far < 15 else 9 if far < 60 else 6
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
                             tooltip={"text": "{label}", "style": {"backgroundColor": "#131C2E", "color": "#E8EDF5"}}),
                    height=height)


def nearby_help(em):
    """Feature 2: helplines, nearest hospitals, immediate safety guidance."""
    lat, lon = float(em["lat"]), float(em["lon"])
    st.markdown("## Nearby Emergency Help")

    # 1. helplines
    with st.container(border=True):
        st.markdown("**📞 Emergency helplines**")
        for col, (name, num, color, icon) in zip(st.columns(4), HELPLINES):
            col.markdown(f"<div class='helpline' style='border-top-color:{color}'><div class='nm'>{icon} {name}</div>"
                         f"<div class='num'><a href='tel:{num}' style='color:{color}'>{num}</a></div></div>",
                         unsafe_allow_html=True)

    # 2. nearest hospitals
    near = nearest_hospitals(lat, lon)
    with st.container(border=True):
        st.markdown("**🏥 Nearest hospitals**")
        left, right = st.columns([2, 3], gap="large")
        with left:
            st.markdown(f"**📍 {em['location_name']}**")
            st.caption(f"Latitude: {lat:.5f}, Longitude: {lon:.5f}. Distances are measured from this location. "
                       "Hospital data is a small sample for demonstration.")
            kpi("Nearest hospital", near[0]["name"], "#34D399", "🏥")
            kpi("Distance", f"{near[0]['distance_km']} km", "#5AA2FF", "📏")
            kpi("Estimated travel time", f"{near[0]['eta_min']} min", "#FFB02E", "⏱️")
            if near[0]["distance_km"] > 50:
                st.info("The sample hospital list covers only the demo area, so these hospitals are far away. "
                        "In a real deployment this list would cover your city.")
        with right:
            hospital_map(lat, lon, near, height=420)
            st.caption("🟡 Emergency location   🟢 Hospital (hover for name and phone)")
        table = pd.DataFrame([{"Hospital": r["name"], "Phone": r["phone"], "Emergency facilities": r["facilities"],
                               "Distance (km)": r["distance_km"], "Travel time (min)": r["eta_min"]} for r in near])
        st.dataframe(table, use_container_width=True, hide_index=True)

    # 3. immediate safety guidance
    with st.container(border=True):
        st.markdown("**🩺 Immediate safety guidance**")
        for key in safety_guidance_keys(em["emergency_type"], em["report_text"]):
            title, steps = SAFETY_GUIDE[key]
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- {s}" for s in steps))
        st.warning("This guidance is only for immediate support. Professional medical help is essential.")


def show_result(eid):
    em = get_emergency(eid)
    if em is None:
        return
    st.markdown(f"### Your request {eid}")
    severity_banner(em["severity"], em["emergency_type"], em["priority"])

    if st.session_state.get("last_merged") and eid == st.session_state.get("last_eid"):
        st.info("Your report has been added to an existing emergency incident. Help is already being coordinated.")
    with st.container(border=True):
        st.markdown("**Incident details**")
        st.markdown(incident_facts(em), unsafe_allow_html=True)

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

    nearby_help(em)


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
        loc_left, loc_right = st.columns([3, 2], gap="large")
        with loc_left:
            lat, lon, loc_name = location_picker()
        with loc_right:
            st.markdown("**Your location and nearby hospitals**")
            hospital_map(lat, lon, nearest_hospitals(lat, lon), height=300, victim_label="📍 Your location")
            st.caption("🟡 Your location   🟢 Nearby hospital")

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
                    eid, merged = submit_sos(
                        report.strip(), pred, lat, lon, loc_name, auth.get("name", ""), auth.get("phone", ""))
                    st.session_state.last_eid = eid
                    st.session_state.last_merged = merged
                if merged:
                    st.info("Your report has been added to an existing emergency incident. "
                            "Help is already being coordinated.")
                else:
                    st.success("Help requested. The control room has been notified.")

    if st.session_state.last_eid:
        st.divider()
        show_result(st.session_state.last_eid)

    if st.button("Start a new report", key="btn_new"):
        st.session_state.sos_active = False
        st.session_state.last_eid = None
        st.session_state.last_merged = False
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

    logs["state"] = logs["status"].map(incident_state)
    logs["reports_n"] = logs.apply(n_reports, axis=1)

    k = st.columns(5)
    with k[0]:
        kpi("Incidents", len(logs), "#5AA2FF", "📋")
    with k[1]:
        kpi("Total SOS reports", int(logs["reports_n"].sum()), "#A78BFA", "📨")
    with k[2]:
        kpi("Critical incidents", int((logs["severity"] == "Critical").sum()), "#FF4D5E", "🚨")
    with k[3]:
        kpi("Active (not dispatched)", int((logs["state"] == "Active").sum()), "#FFB02E", "⏳")
    with k[4]:
        kpi("In response", int(logs["state"].isin(["Assigned", "In Progress"]).sum()), "#34D399", "🚑")

    with st.container(border=True):
        st.markdown("**Incident queue** (duplicate SOS reports are grouped, highest priority first)")
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
        table = view[["emergency_id", "time", "emergency_type", "severity", "priority", "reports_n", "location_name",
                      "required_resources", "assigned_units", "state", "status"]]
        st.dataframe(style_table(table), use_container_width=True, hide_index=True,
                     column_config={"emergency_id": "Incident ID", "time": "First report", "reports_n": "SOS reports",
                                    "state": "Incident state", "status": "Detailed status",
                                    "emergency_type": "Type", "location_name": "Location",
                                    "required_resources": "Required resources", "assigned_units": "Assigned units"})

    if view.empty:
        st.warning("No emergencies match these filters.")
        return

    with st.container(border=True):
        eid = st.selectbox("Open incident", view["emergency_id"])
        em = get_emergency(eid)
        severity_banner(em["severity"], em["emergency_type"], em["priority"])
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown(incident_facts(em), unsafe_allow_html=True)
            with st.expander(f"All SOS reports for this incident ({n_reports(em)})", expanded=n_reports(em) > 1):
                for i, r in enumerate(get_reports(em), 1):
                    st.markdown(f"**{i}.** {r['time']} — {r.get('name') or 'Anonymous'} {r.get('phone', '')}  \n"
                                f"“{r['text']}”")
            timeline(em["status"])

            # resources are assigned only once per incident: types already assigned are not offered again
            required = [r.strip() for r in em["required_resources"].split(",") if r.strip()]
            fleet = load_resources().set_index("resource_id")["type"]
            assigned_ids = [u for u in em["assigned_units"].split(",") if u]
            assigned_types = {fleet[u] for u in assigned_ids if u in fleet.index}
            options = [r for r in RESOURCE_TYPES if r not in assigned_types]
            if assigned_types:
                st.success("Already assigned to this incident: " + ", ".join(sorted(assigned_types))
                           + ". They will not be dispatched again.")
            chosen = st.multiselect("Resources to dispatch (change the AI suggestion if needed)", options,
                                    default=[r for r in required if r in options], key=f"ms_{eid}_{len(assigned_ids)}")
            st.markdown("**Nearest available units right now**")
            for r in recommend(chosen, float(em["lat"]), float(em["lon"])):
                unit_card(r)

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