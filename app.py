import os
from urllib.parse import quote
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from geopy.distance import geodesic

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Tel Aviv Shelter Locator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
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


# --- LangChain tools for the conversational agent ---
@tool
def geocode_address(street_name: str, house_number: int | str) -> dict | str:
    """
    Look up coordinates for a Hebrew street name and house number in Tel Aviv.
    Use this when the user gives their location (e.g. street name and number).
    street_name: Hebrew street name (e.g. דיזנגוף). house_number: number as int or string.
    Returns a dict with lat, lon and found_address, or an error string if not found.
    """
    df_addresses = load_addresses_data("tlv_addresses.csv")
    lat, lon, display_address, _used_fallback, _fallback_num = lookup_address_offline(
        df_addresses, street_name.strip(), str(house_number).strip()
    )
    if lat is None or lon is None:
        return "Address not found in the municipal database. Please verify the street name and house number."
    return {"lat": lat, "lon": lon, "found_address": display_address or f"{street_name} {house_number}, Tel Aviv"}


@tool
def find_shelters(lat: float, lon: float) -> list:
    """
    Find the 3 closest emergency shelters to the given coordinates in Tel Aviv.
    Returns a list of up to 3 shelters with distance_m, address, lat, lon, size.
    """
    df = load_shelters_data("tlv_shelters.csv")
    top_3_list, err = find_nearest_shelters_from_coords(lat, lon, df)
    if err:
        return []
    if top_3_list:
        try:
            st.session_state.latest_map_data = {
                "user_lat": lat,
                "user_lon": lon,
                "shelters": top_3_list,
            }
        except Exception:
            pass
    return top_3_list


@st.cache_resource
def _create_agent_executor():
    """Build the tool-calling agent and executor (cached per session)."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [geocode_address, find_shelters]
    system_msg = (
        "You are a Tel Aviv tactical emergency assistant. The user will state their location and situation. "
        "1. Use 'geocode_address' to find coordinates (street_name in Hebrew, house_number as string). "
        "2. Use 'find_shelters' to find the 3 closest safe locations. "
        "3. Reply in short, tactical Hebrew: list the 3 shelters with distances only. "
        "4. End with one very short, basic line only (e.g. 'לך למקלט הקרוב. הישאר בטוח.'). No long bullet lists or multiple הנחיות. Do not output raw JSON."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- 3. Hero Section ---
st.markdown(
    """
    <div class="hero-card">
        <h1>Tel Aviv Emergency Shelters.</h1>
        <p class="hero-subtitle">
            Describe your location in chat to find the closest shelters immediately.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Validation (warn but do not stop so chat UI always renders) ---
validation_ok = True
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API Key is missing. Please configure your .env file.")
    validation_ok = False
if not os.path.exists("tlv_shelters.csv"):
    st.error("Dataset file missing: tlv_shelters.csv")
    validation_ok = False
if not os.path.exists("tlv_addresses.csv"):
    st.error("Address database missing: tlv_addresses.csv. Run get_addresses.py to generate it.")
    validation_ok = False
df = load_shelters_data("tlv_shelters.csv") if os.path.exists("tlv_shelters.csv") else pd.DataFrame()

# --- Expander 1: About the Developer (below main header) ---
with st.expander("About the Developer", expanded=False):
    st.markdown("<h4 style='text-align: center; color: #1d1d1f; margin-bottom: 10px;'>About the Developer</h4>", unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; color: #515154; font-size: 15px; line-height: 1.6; max-width: 600px; margin: 0 auto;'>I'm Rotem, an Electrical and Electronics Engineering student with a strong interest in GenAI, Data Science, and Data Engineering. I built this offline-first shelter locator to provide a simple and reliable tool for emergencies. I'm a dedicated hard worker, passionate about learning new technologies and building practical, data-driven solutions.</p>", unsafe_allow_html=True)

    st.markdown("<div style='text-align: center; margin-top: 20px; margin-bottom: 10px;'><a href='https://www.linkedin.com/in/rotem-ezra-24-07-97re/' target='_blank' style='text-decoration: none; font-weight: 500; background-color: #0a66c2; color: white; padding: 8px 20px; border-radius: 20px; font-size: 14px;'>LinkedIn</a></div>", unsafe_allow_html=True)

# --- Sidebar: About the Developer (recruiter exposure) ---
with st.sidebar:
    st.markdown("## 👨‍💻 About the Developer")
    st.subheader("⚡ Data & AI for Emergency Response")
    st.markdown(
        "Built by Rotem, an Electrical Engineering student. This project merges "
        "data engineering and AI to create a real-time, offline-capable tactical solution for the home front."
    )
    st.divider()
    st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/rotem-ezra-24-07-97re/)")
    st.markdown("[🐙 GitHub](#)")

# --- 4. Chat UI: textbox first, then answer block below ---
st.markdown("<p class='section-subtitle' style='margin-top: 1rem;'>Type your location below to find nearby shelters.</p>", unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Text box + Send button at the top
input_col, btn_col = st.columns([5, 1])
with input_col:
    user_message = st.text_input(
        "Your location or question",
        placeholder="e.g  אני בזידנגוף 50  ",
        key="chat_text_input",
        label_visibility="collapsed",
    )
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)  # align with input
    send_clicked = st.button("Send", type="primary")

# Answer block: show only the latest exchange below the textbox
for msg in st.session_state.messages[-2:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Accept input from either the text box (Send) or the native chat input
prompt_input = user_message.strip() if (send_clicked and user_message) else st.chat_input("איפה אתה נמצא? (לדוגמה: אני בזידנגוף 50)")
if prompt_input:
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    if not validation_ok:
        reply = "Cannot search: please add OPENAI_API_KEY to .env and ensure tlv_shelters.csv and tlv_addresses.csv exist."
    else:
        chat_history = []
        for m in st.session_state.messages[:-1]:
            if m["role"] == "user":
                chat_history.append(HumanMessage(content=m["content"]))
            else:
                chat_history.append(AIMessage(content=m["content"]))
        with st.spinner("מחפש מחסות..."):
            try:
                agent_executor = _create_agent_executor()
                result = agent_executor.invoke(
                    {"input": prompt_input, "chat_history": chat_history}
                )
                reply = result.get("output", str(result))
            except Exception as e:
                reply = f"שגיאה: {e}"
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# --- 5. Folium map and Google Maps links (from latest tool-call result) ---
if "latest_map_data" in st.session_state:
    map_data = st.session_state.latest_map_data
    user_lat = map_data.get("user_lat")
    user_lon = map_data.get("user_lon")
    shelters = map_data.get("shelters") or []
    if user_lat is not None and user_lon is not None and shelters:
        col_text, col_map = st.columns([1, 1], gap="large")
        with col_text:
            st.markdown("*Note: GPS might not be accurate due to signal blocking.*")
            for i, shelter in enumerate(shelters):
                lat = shelter.get("lat")
                lon = shelter.get("lon")
                address = shelter.get("address") or shelter.get("t_ktovet") or "Unknown Address"
                if address and address != "Unknown Address" and user_lat is not None and user_lon is not None:
                    destination_encoded = quote(f"{address}, Tel Aviv, Israel", safe="")
                    gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lon}&destination={destination_encoded}&travelmode=walking"
                    st.markdown(f"**Option {i+1}:** [📍 Navigate to {address} (Google Maps)]({gmaps_url})")
                elif lat is not None and lon is not None:
                    gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lon}&destination={lat},{lon}&travelmode=walking"
                    st.markdown(f"**Option {i+1}:** [📍 Navigate to {address} (Google Maps)]({gmaps_url})")
        with col_map:
            if isinstance(shelters, list) and len(shelters) > 0 and "lat" in shelters[0] and "lon" in shelters[0]:
                map_obj = build_satellite_map(user_lat, user_lon, shelters)
                st_folium(map_obj, use_container_width=True, height=400)

# --- 6. KPI widgets at the very bottom (secondary info) ---
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

# --- Emergency & System Info at bottom of page ---
with st.expander("🛡️ מידע חירום ואמינות המערכת (Emergency & System Info)", expanded=False):
    st.markdown("<h4 style='text-align: center; color: #1d1d1f;'>📞 מוקדי חירום (Emergency Lines)</h4>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #515154;'>**104** – פיקוד העורף (Home Front Command)<br>**100** – משטרה (Police)<br>**101** – מד\"א (Ambulance)</p>",
        unsafe_allow_html=True,
    )
    st.info("💡 **Why trust this app?** System operates with a fully offline geocoding fallback to bypass GPS spoofing and network crashes.")

# --- Connection status footer ---
st.markdown(
    """
    <div style='text-align: center; color: #86868b; font-size: 12px; margin-top: 2rem;'>
        <p style='margin: 0;'>Dataset — This dashboard is connected to Tel Aviv Municipality emergency shelters data.</p>
        <p style='margin: 0.25rem 0 0 0;'>🟢 Connected: Tel Aviv Emergency Shelters DB</p>
    </div>
    """,
    unsafe_allow_html=True,
)