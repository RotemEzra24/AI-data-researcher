import os
from urllib.parse import quote
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from geopy.distance import geodesic

from prompts import build_agent_prompt

# --- 1. Page Configuration ---
st.set_page_config(page_title="Tel Aviv Emergency Shelters", page_icon="", layout="wide")
load_dotenv()

# --- 2. Premium Styling (CSS) ---
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
def load_addresses_data(path: str = "tlv_addresses.csv") -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_streets_from_addresses(path: str = "tlv_addresses.csv") -> list[str]:
    """
    Load tlv_addresses.csv and return a sorted, unique list of Hebrew street names
    for the selectbox (column t_rechov). Sorted alphabetically in Hebrew (standard
    Python string sort gives correct Hebrew Alef–Tav order via Unicode).
    """
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    street_col = "t_rechov" if "t_rechov" in df.columns else None
    if street_col is None:
        return []
    streets = df[street_col].dropna().astype(str).str.strip()
    streets = streets[streets != ""]
    unique = streets.unique().tolist()
    sorted_streets = sorted(
        unique,
        key=lambda x: (1, x) if x and x.strip() and x.strip()[0].isdigit() else (0, x),
    )
    return sorted_streets


def lookup_address_offline(
    df_addresses: pd.DataFrame,
    street_name: str,
    house_number: str,
) -> tuple[float | None, float | None, str | None, bool, str | None]:
    """
    Find the row in tlv_addresses matching the given Hebrew street and house number.
    Returns (lat, lon, display_address, used_fallback, fallback_closest_num).
    If exact match fails and user gave a house number, fallback: closest numeric house on that street.
    """
    if df_addresses.empty or not street_name or not street_name.strip():
        return None, None, None, False, None

    street_col = "t_rechov"
    lat_col = "Latitude" if "Latitude" in df_addresses.columns else "lat"
    lon_col = "Longitude" if "Longitude" in df_addresses.columns else "lon"
    if lat_col not in df_addresses.columns or lon_col not in df_addresses.columns:
        return None, None, None, False, None

    street_clean = street_name.strip()
    house_clean = house_number.strip() if house_number else ""

    mask_street = df_addresses[street_col].astype(str).str.strip() == street_clean

    if house_clean:
        # Exact match: ms_bayit or t_bayit_veknisa
        mask_house = (
            df_addresses["ms_bayit"].astype(str).str.strip() == house_clean
            if "ms_bayit" in df_addresses.columns
            else False
        )
        if "t_bayit_veknisa" in df_addresses.columns:
            mask_house = mask_house | (
                df_addresses["t_bayit_veknisa"].astype(str).str.strip() == house_clean
            )
        matches = df_addresses.loc[mask_street & mask_house]
    else:
        matches = df_addresses.loc[mask_street]

    if not matches.empty:
        row = matches.iloc[0]
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
        except (TypeError, ValueError):
            return None, None, None, False, None
        if "t_rechov_eng" in df_addresses.columns and pd.notna(row.get("t_rechov_eng")):
            eng_street = str(row["t_rechov_eng"]).strip()
            num_part = str(row["ms_bayit"]).strip() if house_clean and "ms_bayit" in df_addresses.columns else ""
            display = f"{eng_street} {num_part}, Tel Aviv".strip()
        else:
            display = f"{street_clean} {house_clean}, Tel Aviv".strip()
        return lat, lon, display, False, None

    # No exact match: fallback to closest numeric house number on this street
    if not house_clean or "ms_bayit" not in df_addresses.columns:
        return None, None, None, False, None
    try:
        user_house_int = int(house_clean)
    except ValueError:
        return None, None, None, False, None

    street_only = df_addresses.loc[mask_street].copy()
    street_only["_ms_bayit_num"] = pd.to_numeric(street_only["ms_bayit"], errors="coerce")
    street_only = street_only.dropna(subset=["_ms_bayit_num"])
    if street_only.empty:
        return None, None, None, False, None

    street_only["_diff"] = (street_only["_ms_bayit_num"] - user_house_int).abs()
    idx = street_only["_diff"].idxmin()
    row = street_only.loc[idx]
    try:
        lat = float(row[lat_col])
        lon = float(row[lon_col])
    except (TypeError, ValueError):
        return None, None, None, False, None

    closest_num = int(row["_ms_bayit_num"])
    if "t_rechov_eng" in df_addresses.columns and pd.notna(row.get("t_rechov_eng")):
        eng_street = str(row["t_rechov_eng"]).strip()
        display = f"{eng_street} {closest_num}, Tel Aviv".strip()
    else:
        display = f"{street_clean} {closest_num}, Tel Aviv".strip()

    return lat, lon, display, True, str(closest_num)


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_nearest_shelters_from_coords(
    user_lat: float,
    user_lon: float,
    df: pd.DataFrame,
) -> tuple[list[dict], str | None]:
    """
    Compute the three closest shelters to the given user coordinates using geodesic distance.
    """
    if "lat" not in df.columns or "lon" not in df.columns:
        return [], "Shelter dataset does not include lat/lon. Regenerate tlv_shelters.csv using get_shelters.py."

    geo_df = df.dropna(subset=["lat", "lon"]).copy()
    if geo_df.empty:
        return [], "Shelter dataset has no usable coordinates."

    geo_df["distance_m"] = geo_df.apply(
        lambda r: geodesic(
            (float(user_lat), float(user_lon)),
            (float(r["lat"]), float(r["lon"])),
        ).meters,
        axis=1,
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
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            }
        )

    return out, None


def build_satellite_map(
    user_lat: float,
    user_lon: float,
    top_3: list[dict],
) -> folium.Map:
    """
    Build an interactive Folium map centered on the user, with a blue user marker
    and a red marker for each of the 3 closest shelters (shield icon + tooltip).
    """
    m = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=16,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
    )
    folium.Marker(
        [user_lat, user_lon],
        icon=folium.Icon(color="blue", icon="user"),
        popup="You are here",
    ).add_to(m)
    for shelter in top_3:
        lat = shelter.get("lat")
        lon = shelter.get("lon")
        if lat is None or lon is None:
            continue
        address = (
            shelter.get("address")
            or shelter.get("t_ktovet")
            or shelter.get("ktovet")
            or shelter.get("Full_Address")
            or "Shelter"
        )
        dist = shelter.get("distance_m", 0)
        tooltip_text = f"{address} ({int(dist)}m)"
        folium.Marker(
            [float(lat), float(lon)],
            icon=folium.Icon(color="red", icon="shield", prefix="fa"),
            tooltip=tooltip_text,
        ).add_to(m)
    return m


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

# --- Sidebar: About the Project / Recruiter trap ---
with st.sidebar:
    st.title("About the Project")
    st.subheader("Data & AI for Emergency Response")
    st.markdown(
        "Built by Rotem, an Electrical Engineering student. This project merges "
        "data engineering and AI to create a real-time, offline-capable tactical solution for the home front."
    )
    st.markdown("[🔗 LinkedIn](#)")
    st.markdown("[🐙 GitHub](#)")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API Key is missing. Please configure your .env file.")
    st.stop()

if not os.path.exists("tlv_shelters.csv"):
    st.error("Dataset file missing: tlv_shelters.csv")
    st.stop()
if not os.path.exists("tlv_addresses.csv"):
    st.error("Address database missing: tlv_addresses.csv. Run get_addresses.py to generate it.")
    st.stop()

df = load_shelters_data("tlv_shelters.csv")
df_addresses = load_addresses_data("tlv_addresses.csv")

# --- 4. Smart Input Row (Street + House Number) ---
street_names = get_streets_from_addresses()
street_options = ["Select street…"] + (street_names if street_names else [])

input_col1, input_col2, input_col_btn = st.columns([3, 1, 1], vertical_alignment="bottom")

with input_col1:
    selected_street = st.selectbox(
        "Street Name",
        options=street_options,
        index=None,
        placeholder="e.g., מאיר דיזנגוף",
        key="shelter_street",
    )

with input_col2:
    house_number = st.text_input(
        "House Number",
        placeholder="e.g., 50",
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
    has_street = selected_street is not None and selected_street != "Select street…"
    has_house = house_number and str(house_number).strip()
    if not has_street or not has_house:
        st.warning("Please select a street and enter a house number first.")
        st.stop()
    user_lat, user_lon, display_address, used_fallback, fallback_closest_num = lookup_address_offline(
        df_addresses, selected_street, house_number or ""
    )
    if user_lat is None or user_lon is None:
        st.session_state.shelter_result_error = "❌ Address not found in the municipal database. Please verify the house number."
        st.session_state.shelter_result = None
    else:
        with st.spinner("Finding nearest shelters..."):
            try:
                nearest, err = find_nearest_shelters_from_coords(user_lat, user_lon, df)
                if err:
                    st.session_state.shelter_result_error = err
                    st.session_state.shelter_result = None
                else:
                    st.success(f"📍 Location identified: {display_address}")
                    st.session_state.shelter_result = data_agent.format_response(display_address, nearest)
                    st.session_state.shelter_result_error = None
                    st.session_state.shelter_map_user_lat = user_lat
                    st.session_state.shelter_map_user_lon = user_lon
                    st.session_state.shelter_map_nearest = nearest
                    st.session_state.shelter_used_fallback = used_fallback
                    st.session_state.shelter_fallback_msg = (
                        f"{selected_street} {fallback_closest_num}" if used_fallback and fallback_closest_num else None
                    )
            except Exception as e:
                st.session_state.shelter_result_error = str(e)
                st.session_state.shelter_result = None

if st.session_state.shelter_result_error:
    st.error(st.session_state.shelter_result_error)
    st.session_state.pop("shelter_map_user_lat", None)
    st.session_state.pop("shelter_map_user_lon", None)
    st.session_state.pop("shelter_map_nearest", None)
    st.session_state.pop("shelter_used_fallback", None)
    st.session_state.pop("shelter_fallback_msg", None)

if st.session_state.shelter_result:
    if st.session_state.get("shelter_used_fallback") and st.session_state.get("shelter_fallback_msg"):
        st.warning(
            f"⚠️ Exact house number not found. Calculating distance from the nearest available address: {st.session_state['shelter_fallback_msg']}"
        )
    col_text, col_map = st.columns([1, 1], gap="large")
    with col_text:
        st.markdown(
            """
            <div style='background: white; border-left: 4px solid #0071e3; padding: 20px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-top: 18px;'>
            """,
            unsafe_allow_html=True,
        )
        st.write(st.session_state.shelter_result)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("*Note: GPS might not be accurate due to signal blocking.*")
        user_lat = st.session_state.get("shelter_map_user_lat")
        user_lon = st.session_state.get("shelter_map_user_lon")
        top_3 = st.session_state.get("shelter_map_nearest") or []
        for i, shelter in enumerate(top_3):
            lat = shelter.get("lat")
            lon = shelter.get("lon")
            address = shelter.get("address") or shelter.get("t_ktovet") or "Unknown Address"
            if address and address != "Unknown Address" and user_lat is not None and user_lon is not None:
                destination_encoded = quote(f"{address}, Tel Aviv, Israel", safe="")
                gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lon}&destination={destination_encoded}&travelmode=walking"
                st.markdown(f"**Option {i+1}:** [📍 Navigate to {address} (Google Maps)]({gmaps_url})")
            elif lat is not None and lon is not None and user_lat is not None and user_lon is not None:
                gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lon}&destination={lat},{lon}&travelmode=walking"
                st.markdown(f"**Option {i+1}:** [📍 Navigate to {address} (Google Maps)]({gmaps_url})")
    with col_map:
        map_lat = st.session_state.get("shelter_map_user_lat")
        map_lon = st.session_state.get("shelter_map_user_lon")
        map_nearest = st.session_state.get("shelter_map_nearest")
        if (
            map_lat is not None
            and map_lon is not None
            and map_nearest
            and isinstance(map_nearest, list)
            and len(map_nearest) > 0
            and "lat" in map_nearest[0]
            and "lon" in map_nearest[0]
        ):
            map_obj = build_satellite_map(map_lat, map_lon, map_nearest)
            st_folium(map_obj, use_container_width=True, height=400)

# --- 7. KPI widgets at the very bottom (secondary info) ---
st.markdown("<br><br>", unsafe_allow_html=True)
col_kpi1, col_kpi2 = st.columns(2)

with col_kpi1:
    premium_widget(
        "Total Shelters",
        f"{len(df):,}",
        "Active public shelters",
        "linear-gradient(135deg, #FF9A9E 0%, #FECFEF 99%, #FECFEF 100%)",
    )
with col_kpi2:
    premium_widget(
        "City Area",
        "Tel Aviv",
        "Official Municipality Data",
        "linear-gradient(135deg, #43E97B 0%, #38F9D7 100%)",
    )

# --- 8. Connection status footer ---
st.markdown(
    """
    <div style='text-align: center; color: #86868b; font-size: 12px; margin-top: 2rem;'>
        <p style='margin: 0;'>Dataset — This dashboard is connected to Tel Aviv Municipality emergency shelters data.</p>
        <p style='margin: 0.25rem 0 0 0;'>🟢 Connected: Tel Aviv Emergency Shelters DB</p>
    </div>
    """,
    unsafe_allow_html=True,
)