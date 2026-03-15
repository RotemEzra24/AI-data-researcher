"""
Tel Aviv Offline-First Tactical Shelter Locator — Streamlit frontend.

UI only: layout, session state, chat interface, map rendering, expanders.
Data and agent logic live in data_engine and agent_logic.
"""

import os
from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from streamlit_folium import st_folium

from agent_logic import create_agent_executor
from data_engine import build_satellite_map, get_data_dir, load_shelters_data

# --- Page configuration ---
st.set_page_config(
    page_title="Tel Aviv Shelter Locator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_dotenv()

# --- Premium styling (CSS) ---
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        background: radial-gradient(circle at top, #ffffff 0, #f5f5f7 45%, #eaeaee 100%) !important;
        letter-spacing: -0.01em !important;
        line-height: 1.5 !important;
    }
    #MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 2.5rem !important; max-width: 1100px !important; }
    h1 { font-weight: 700 !important; font-size: 3.4rem !important; text-align: center; color: #1d1d1f !important; margin-bottom: 0.25rem !important;}
    h3 { font-weight: 400 !important; color: #86868b !important; text-align: center; font-size: 1.25rem !important; margin-bottom: 2.75rem !important;}
    .hero-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(245,245,247,0.96));
        border-radius: 28px;
        padding: 32px 40px;
        box-shadow: 0 18px 55px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.9);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        margin-bottom: 32px;
    }
    .hero-subtitle { font-size: 1.05rem; color: #86868b; margin-top: 0.5rem; margin-bottom: 0; text-align: center; }
    .section-card {
        background: rgba(255,255,255,0.96);
        border-radius: 22px;
        padding: 22px 24px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f3;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        margin-bottom: 26px;
    }
    .section-title { color: #1d1d1f; font-weight: 600; font-size: 1.05rem; margin-bottom: 0.4rem; }
    .section-subtitle { color: #a1a1a6; font-size: 0.9rem; margin-bottom: 0.4rem; }
    .stTextInput input {
        border-radius: 16px !important; padding: 16px !important; font-size: 16px !important;
        border: 1px solid #E5E5EA !important; background: #FFFFFF !important; color: #1d1d1f !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.03) !important;
    }
    .stTextInput input::placeholder { color: #a1a1a6 !important; opacity: 1 !important; }
    .stTextInput input:focus { border-color: #0071e3 !important; box-shadow: 0 0 0 4px rgba(0,113,227,0.1) !important; }
    .stButton>button {
        background: linear-gradient(135deg, #0071e3 0%, #4facfe 100%) !important;
        color: white !important; border-radius: 980px !important; padding: 10px 24px !important;
        font-weight: 600 !important; border: none !important; transition: transform 0.2s !important;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,113,227,0.3) !important; }
</style>
<script>window.scrollTo(0, 0); document.documentElement.scrollTop = 0; document.body.scrollTop = 0;</script>
""", unsafe_allow_html=True)


def premium_widget(title, value, subtitle, gradient="linear-gradient(90deg, #1d1d1f 0%, #434344 100%)"):
    """Render a premium KPI card."""
    html = f"""
    <div style="background: #ffffff; border-radius: 20px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.04); border: 1px solid #F0F0F2; height: 100%;">
        <p style="color: #86868b; font-size: 14px; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">{title}</p>
        <h2 style="margin: 10px 0; font-size: 32px; font-weight: 700; background: {gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{value}</h2>
        <p style="color: #a1a1a6; font-size: 13px; margin: 0; font-weight: 500;">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _api_key_configured() -> bool:
    """True if OpenAI API key is set via env or Streamlit secrets."""
    if os.environ.get("OPENAI_API_KEY"):
        return True
    try:
        return bool(st.secrets.get("OPENAI_API_KEY"))
    except Exception:
        return False


# --- Hero ---
st.markdown("""
<div class="hero-card">
    <h1>Tel Aviv Emergency Shelters.</h1>
    <p class="hero-subtitle">Describe your location in chat to find the closest shelters immediately.</p>
</div>
""", unsafe_allow_html=True)

# --- Validation (warn only; keep chat visible) ---
data_dir = get_data_dir()
shelters_path = data_dir / "tlv_shelters.csv"
addresses_path = data_dir / "tlv_addresses.csv"
validation_ok = True
if not _api_key_configured():
    st.error("OpenAI API Key is missing. Set OPENAI_API_KEY in .env or in Streamlit secrets.")
    validation_ok = False
if not shelters_path.exists():
    st.error("Dataset file missing: data/tlv_shelters.csv")
    validation_ok = False
if not addresses_path.exists():
    st.error("Address database missing: data/tlv_addresses.csv. Run get_addresses.py to generate it.")
    validation_ok = False

df = load_shelters_data() if shelters_path.exists() else pd.DataFrame()

# --- Chat UI ---
st.markdown("<p class='section-subtitle' style='margin-top: 1rem;'>Type your location below to find nearby shelters.</p>", unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state.messages = []

input_col, btn_col = st.columns([5, 1])
with input_col:
    user_message = st.text_input(
        "Your location or question",
        placeholder="e.g  אני בדיזנגוף 50",
        key="chat_text_input",
        label_visibility="collapsed",
    )
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    send_clicked = st.button("Send", type="primary")

for msg in st.session_state.messages[-2:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Geolocation button ---
loc = None
_q_lat = st.query_params.get("lat")
_q_lon = st.query_params.get("lon")
if _q_lat is not None and _q_lon is not None:
    try:
        loc = {"latitude": float(_q_lat), "longitude": float(_q_lon)}
    except (ValueError, TypeError):
        loc = None
st.markdown("<p class='section-subtitle' style='margin-top: 1rem;'>Or use your live device location (may be limited by GPS signal):</p>", unsafe_allow_html=True)
loc_col_left, loc_col_btn = st.columns([5, 1])
with loc_col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    share_location_html = """
    <div id="share-loc-container" style="box-sizing: border-box; padding: 0; margin: 0;">
        <button type="button" id="share-loc-btn" style="
            background: linear-gradient(135deg, #0071e3 0%, #4facfe 100%);
            color: white; border-radius: 980px; padding: 6px 14px;
            font-weight: 600; border: none; font-size: 13px; cursor: pointer;
            width: 100%; max-width: 120px; white-space: nowrap;
            line-height: 1.3; box-sizing: border-box; min-height: 32px;
        ">Share location</button>
    </div>
    <script>
        (function() {
            var btn = document.getElementById('share-loc-btn');
            if (!btn) return;
            btn.onclick = function() {
                btn.disabled = true;
                btn.textContent = 'Getting location...';
                if (!navigator.geolocation) {
                    btn.textContent = 'Not supported';
                    btn.disabled = false;
                    return;
                }
                var opts = { timeout: 15000, maximumAge: 60000 };
                navigator.geolocation.getCurrentPosition(
                    function(pos) {
                        var w = window.top || window.parent || window;
                        var url = new URL(w.location.href);
                        url.searchParams.set('lat', pos.coords.latitude);
                        url.searchParams.set('lon', pos.coords.longitude);
                        w.location.href = url.toString();
                    },
                    function(err) {
                        btn.disabled = false;
                        if (err.code === 3) btn.textContent = 'Timed out – try again or type address';
                        else if (err.code === 1) btn.textContent = 'Location denied';
                        else btn.textContent = 'Share location';
                    },
                    opts
                );
            };
        })();
    </script>
    """
    components.html(share_location_html, height=44)

# --- GPS location flow ---
if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
    lat, lon = loc["latitude"], loc["longitude"]
    loc_str = f"{lat},{lon}"
    if st.session_state.get("last_processed_loc") != loc_str:
        st.session_state.last_processed_loc = loc_str
        gps_prompt = f"My exact location is latitude {lat} and longitude {lon}. Find nearby shelters."
        st.session_state.messages.append({"role": "user", "content": "Using current GPS location..."})
        if validation_ok:
            agent_executor = create_agent_executor()
            if agent_executor:
                chat_history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                    for m in st.session_state.messages[:-1]
                ]
                try:
                    result = agent_executor.invoke({"input": gps_prompt, "chat_history": chat_history})
                    reply = result.get("output", str(result))
                except Exception as e:
                    reply = f"Error: {e}"
            else:
                reply = "Agent not available. Set OPENAI_API_KEY in .env or Streamlit secrets."
        else:
            reply = "Configure OPENAI_API_KEY and ensure data/tlv_shelters.csv and data/tlv_addresses.csv exist."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# --- Text / chat input flow ---
prompt_input = (user_message.strip() if (send_clicked and user_message) else
                st.chat_input("e.g. אני בדיזנגוף 50"))
if prompt_input:
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    if not validation_ok:
        reply = "Configure OPENAI_API_KEY and ensure data/tlv_shelters.csv and data/tlv_addresses.csv exist."
    else:
        agent_executor = create_agent_executor()
        if not agent_executor:
            reply = "Agent not available. Set OPENAI_API_KEY in .env or Streamlit secrets."
        else:
            chat_history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            with st.spinner("Analyzing location and finding shelters..."):
                try:
                    result = agent_executor.invoke({"input": prompt_input, "chat_history": chat_history})
                    reply = result.get("output", str(result))
                except Exception as e:
                    reply = f"Error: {e}"
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# --- Map and Google Maps links (from latest tool result) ---
if "latest_map_data" in st.session_state:
    map_data = st.session_state.latest_map_data
    user_lat = map_data.get("user_lat")
    user_lon = map_data.get("user_lon")
    shelters = map_data.get("shelters") or []
    if user_lat is not None and user_lon is not None and shelters:
        col_text, col_map = st.columns([1, 1], gap="large")
        with col_text:
            for shelter in shelters:
                lat = shelter.get("lat")
                lon = shelter.get("lon")
                address = shelter.get("address") or shelter.get("t_ktovet") or "Unknown Address"
                distance = shelter.get("distance_m", 0)
                if address and address != "Unknown Address" and user_lat and user_lon:
                    destination_encoded = quote(f"{address}, Tel Aviv, Israel", safe="")
                    gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lon}&destination={destination_encoded}&travelmode=walking"
                elif lat is not None and lon is not None:
                    gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lon}&destination={lat},{lon}&travelmode=walking"
                else:
                    gmaps_url = "#"
                st.markdown(f"**{address}** - {int(distance)} m | [Open in Google Maps]({gmaps_url})")
            st.markdown("*Note: GPS might not be accurate due to signal blocking.*")
        with col_map:
            if shelters and "lat" in shelters[0] and "lon" in shelters[0]:
                map_obj = build_satellite_map(user_lat, user_lon, shelters)
                st_folium(map_obj, use_container_width=True, height=400)

# --- KPI widgets ---
st.markdown("<br><br>", unsafe_allow_html=True)
col_kpi1, col_kpi2 = st.columns(2)
with col_kpi1:
    premium_widget("Total Shelters", f"{len(df):,}", "Active public shelters",
                  "linear-gradient(135deg, #FF9A9E 0%, #FECFEF 99%, #FECFEF 100%)")
with col_kpi2:
    premium_widget("City Area", "Tel Aviv", "Official Municipality Data",
                  "linear-gradient(135deg, #43E97B 0%, #38F9D7 100%)")

# --- Emergency & System Info expander ---
with st.expander("Emergency & System Info", expanded=False):
    st.markdown("<h4 style='text-align: center; color: #86868b; margin-bottom: 12px;'>Emergency Lines</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-bottom: 16px;'>
        <a href='tel:104' style='display: inline-block; text-decoration: none; background: linear-gradient(135deg, #0071e3 0%, #4facfe 100%); color: white; padding: 12px 20px; border-radius: 12px; font-weight: 600; font-size: 14px; box-shadow: 0 2px 8px rgba(0,113,227,0.3);'>104 – Home Front</a>
        <a href='tel:100' style='display: inline-block; text-decoration: none; background: linear-gradient(135deg, #0071e3 0%, #4facfe 100%); color: white; padding: 12px 20px; border-radius: 12px; font-weight: 600; font-size: 14px; box-shadow: 0 2px 8px rgba(0,113,227,0.3);'>100 – Police</a>
        <a href='tel:101' style='display: inline-block; text-decoration: none; background: linear-gradient(135deg, #0071e3 0%, #4facfe 100%); color: white; padding: 12px 20px; border-radius: 12px; font-weight: 600; font-size: 14px; box-shadow: 0 2px 8px rgba(0,113,227,0.3);'>101 – Ambulance</a>
    </div>
    """, unsafe_allow_html=True)
    st.info("**Why trust this app?** System relies on local municipal data for zero-latency routing and manual-entry fallback to bypass GPS spoofing and jamming.")

# --- About the Developer ---
with st.expander("About the Developer", expanded=False):
    st.markdown("<h4 style='text-align: center; color: #86868b; margin-bottom: 10px;'>About the Developer</h4>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #86868b; font-size: 15px; line-height: 1.6; max-width: 600px; margin: 0 auto;'>Hi, I'm Rotem, an Electrical and Electronics Engineering student with a growing focus on GenAI, Data Science, and Data Engineering. I'm passionate about building practical tools that combine data and technology to solve real-world problems.</p>", unsafe_allow_html=True)
    st.markdown("""
<div style='text-align: center; margin-top: 20px; margin-bottom: 10px; display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;'>
    <a href='https://www.linkedin.com/in/rotem-ezra-24-07-97re/' target='_blank' style='text-decoration: none; font-weight: 500; background-color: #0a66c2; color: white; padding: 8px 20px; border-radius: 20px; font-size: 14px;'>LinkedIn</a>
    <a href='https://github.com/RotemEzra24/AI-data-researcher' target='_blank' style='text-decoration: none; font-weight: 500; background-color: #24292f; color: white; padding: 8px 20px; border-radius: 20px; font-size: 14px;'>GitHub</a>
</div>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div style='text-align: center; color: #86868b; font-size: 12px; margin-top: 2rem;'>
    <p style='margin: 0;'>Dataset — This dashboard is connected to Tel Aviv Municipality emergency shelters data.</p>
    <p style='margin: 0.25rem 0 0 0;'>Connected: Tel Aviv Emergency Shelters DB</p>
</div>
""", unsafe_allow_html=True)
