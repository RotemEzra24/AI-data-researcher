import streamlit as st
import pandas as pd
import os
import math
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from prompts import build_agent_prompt

# --- 1. Page Configuration ---
st.set_page_config(page_title="Tel Aviv Emergency Shelters", page_icon="", layout="wide")
load_dotenv()

# --- 2. Premium Styling (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background: radial-gradient(circle at top, #ffffff 0, #f5f5f7 45%, #eaeaee 100%) !important;
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

    .hero-subtitle {
        font-size: 1.05rem;
        color: #86868b;
        margin-top: 0.5rem;
        margin-bottom: 0;
        text-align: center;
    }

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

    .section-title {
        color: #1d1d1f;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.4rem;
    }

    .section-subtitle {
        color: #a1a1a6;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    
    /* Input Field */
    .stTextInput input {
        border-radius: 16px !important;
        padding: 16px !important;
        font-size: 16px !important;
        border: 1px solid #E5E5EA !important;
        background: #FFFFFF !important;
        color: #1d1d1f !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.03) !important;
    }
    .stTextInput input::placeholder {
        color: #a1a1a6 !important;
        opacity: 1 !important;
    }
    .stTextInput input:focus { border-color: #0071e3 !important; box-shadow: 0 0 0 4px rgba(0,113,227,0.1) !important;}
    
    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #0071e3 0%, #4facfe 100%) !important;
        color: white !important; border-radius: 980px !important; padding: 10px 24px !important;
        font-weight: 600 !important; border: none !important; transition: transform 0.2s !important;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,113,227,0.3) !important; }
</style>
""", unsafe_allow_html=True)

# --- Function to draw premium HTML widgets ---
def premium_widget(title, value, subtitle, gradient="linear-gradient(90deg, #1d1d1f 0%, #434344 100%)"):
    html = f"""
    <div style="background: #ffffff; border-radius: 20px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.04); border: 1px solid #F0F0F2; height: 100%;">
        <p style="color: #86868b; font-size: 14px; font-weight: 600; margin: 0; text-transform: uppercase; letter-spacing: 0.5px;">{title}</p>
        <h2 style="margin: 10px 0; font-size: 32px; font-weight: 700; background: {gradient}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{value}</h2>
        <p style="color: #a1a1a6; font-size: 13px; margin: 0; font-weight: 500;">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


class DataResearchAgent:
    """
    The application computes nearest shelters deterministically in Python.
    The LLM is used only to format the output clearly and urgently.
    """

    def __init__(self, df: pd.DataFrame, is_car_data: bool) -> None:
        self.df = df
        self.is_car_data = is_car_data
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_response(self, user_input: str, nearest_rows: list[dict]) -> str:
        """
        Format the precomputed nearest shelters into a clean, urgent response.
        """
        system_prompt = build_agent_prompt(
            user_question=user_input,
            df=self.df,
            is_car_data=False,
        )
        full_prompt = (
            f"{system_prompt}\n\n"
            f"Nearest shelters (precomputed, do not recalculate):\n{nearest_rows}\n"
        )
        resp = self._llm.invoke(full_prompt)
        return getattr(resp, "content", str(resp))


@st.cache_data(show_spinner=False)
def load_shelters_data(path: str = "tlv_shelters.csv") -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def geocode_address(address: str) -> tuple[float, float] | None:
    geolocator = Nominatim(user_agent="tlv-emergency-shelter-locator")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0, swallow_exceptions=True)
    loc = geocode(f"{address}, Tel Aviv, Israel")
    if loc is None:
        return None
    return float(loc.latitude), float(loc.longitude)


def haversine_m(lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
    r = 6371000.0
    lat1 = np.radians(lat.astype(float))
    lon1 = np.radians(lon.astype(float))
    lat2 = math.radians(float(ref_lat))
    lon2 = math.radians(float(ref_lon))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_street_names(df: pd.DataFrame) -> list[str]:
    """Extract sorted, unique street names for the selectbox (English preferred for geocoding)."""
    col = pick_col(df, ["shem_rechov_eng", "shem_recho", "Full_Address"])
    if col is None:
        return []
    values = df[col].dropna().astype(str).str.strip()
    values = values[values != ""]
    return sorted(values.unique().tolist())


def find_nearest_shelter(user_address: str, df: pd.DataFrame) -> tuple[list[dict], str | None]:
    coords = geocode_address(user_address)
    if coords is None:
        return [], "Could not geocode the address. Try a more specific street number in Tel Aviv."

    user_lat, user_lon = coords

    if "lat" not in df.columns or "lon" not in df.columns:
        return [], "Shelter dataset does not include lat/lon. Regenerate tlv_shelters.csv using get_shelters.py."

    geo_df = df.dropna(subset=["lat", "lon"]).copy()
    if geo_df.empty:
        return [], "Shelter dataset has no usable coordinates."

    geo_df["distance_m"] = haversine_m(
        geo_df["lat"].to_numpy(),
        geo_df["lon"].to_numpy(),
        user_lat,
        user_lon,
    )
    top = geo_df.sort_values("distance_m", ascending=True).head(3)

    address_col = pick_col(
        df,
        [
            "Full_Address",
            "FULL_ADDRESS",
            "full_address",
            "Address",
            "address",
            "כתובת",
            "LOCATION",
            "location",
        ],
    )
    accessibility_col = pick_col(
        df,
        [
            "miklat_mungash",  # Tel Aviv shelters: accessible shelter indicator
            "Accessibility",
            "accessibility",
            "נגישות",
        ],
    )
    size_col = pick_col(
        df,
        [
            "shetach_mr",  # Tel Aviv shelters: area in square meters
            "Size",
            "size",
            "CAPACITY",
            "capacity",
            "גודל",
        ],
    )

    out: list[dict] = []
    for _, row in top.iterrows():
        out.append(
            {
                "address": (
                    str(row[address_col])
                    if address_col and pd.notna(row.get(address_col))
                    else "Unknown"
                ),
                "distance_m": int(round(float(row["distance_m"]))),
                "accessibility": (
                    str(row[accessibility_col]).strip()
                    if accessibility_col and pd.notna(row.get(accessibility_col))
                    else "Unknown"
                ),
                "size": (
                    f"{float(row[size_col]):,.0f} m²"
                    if size_col and pd.notna(row.get(size_col)) and str(row.get(size_col)).strip() != ""
                    else "Unknown"
                ),
            }
        )

    return out, None


# --- 3. Hero Section ---
st.markdown(
    """
    <div class="hero-card">
        <h1>Tel Aviv Emergency Shelters.</h1>
        <p class="hero-subtitle">
            Enter your current location to find the closest shelters immediately.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API Key is missing. Please configure your .env file.")
    st.stop()

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Dataset</div>
        <div class="section-subtitle">
            This dashboard is connected to Tel Aviv Municipality emergency shelters data.
        </div>
    """,
    unsafe_allow_html=True
)

if os.path.exists("tlv_shelters.csv"):
    df = load_shelters_data("tlv_shelters.csv")
    st.success("🟢 Connected: Tel Aviv Emergency Shelters DB")
else:
    st.error("Dataset file missing: tlv_shelters.csv")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- 4. Smart Input Row (Street + House Number) — immediately below badge ---
street_names = get_street_names(df)
street_options = ["Select street…"] + (street_names if street_names else [])

input_col1, input_col2, input_col_btn = st.columns([3, 1, 1], vertical_alignment="bottom")

with input_col1:
    selected_street = st.selectbox(
        "Street Name",
        options=street_options,
        index=0,
        key="shelter_street",
    )

with input_col2:
    house_number = st.text_input(
        "House Number",
        value="",
        placeholder="e.g. 30",
        key="shelter_house_number",
    )

with input_col_btn:
    st.markdown("<br>", unsafe_allow_html=True)  # align button with inputs
    find_clicked = st.button("Find Closest Shelter", type="primary")

# --- 5. AI Agent (used only when displaying result) ---
data_agent = DataResearchAgent(df=df, is_car_data=False)

# --- 6. AI Result (shown below smart input; persisted in session state) ---
if "shelter_result" not in st.session_state:
    st.session_state.shelter_result = None
if "shelter_result_error" not in st.session_state:
    st.session_state.shelter_result_error = None

if find_clicked:
    if not selected_street or selected_street == "Select street…":
        st.warning("Please select a street name.")
    else:
        user_address = f"{selected_street} {house_number}".strip() if house_number else selected_street.strip()
        with st.spinner("Finding nearest shelters..."):
            try:
                nearest, err = find_nearest_shelter(user_address, df)
                if err:
                    st.session_state.shelter_result_error = err
                    st.session_state.shelter_result = None
                else:
                    st.session_state.shelter_result = data_agent.format_response(user_address, nearest)
                    st.session_state.shelter_result_error = None
            except Exception as e:
                st.session_state.shelter_result_error = str(e)
                st.session_state.shelter_result = None

if st.session_state.shelter_result_error:
    st.error(st.session_state.shelter_result_error)

if st.session_state.shelter_result:
    st.markdown(
        """
        <div style='background: white; border-left: 4px solid #0071e3; padding: 20px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-top: 18px;'>
        """,
        unsafe_allow_html=True,
    )
    st.write(st.session_state.shelter_result)
    st.markdown("</div>", unsafe_allow_html=True)

# --- 7. KPI widgets at the very bottom (secondary info) ---
st.markdown("<br><br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    premium_widget(
        "Total Shelters",
        f"{len(df):,}",
        "Active public shelters",
        "linear-gradient(135deg, #FF9A9E 0%, #FECFEF 99%, #FECFEF 100%)",
    )
with col2:
    premium_widget(
        "Data Points",
        f"{len(df.columns)}",
        "Parameters per shelter",
        "linear-gradient(135deg, #667EEA 0%, #764BA2 100%)",
    )
with col3:
    premium_widget(
        "City Area",
        "Tel Aviv",
        "Official Municipality Data",
        "linear-gradient(135deg, #43E97B 0%, #38F9D7 100%)",
    )