"""
Data layer for the Tel Aviv Offline-First Tactical Shelter Locator.

Handles loading of municipal CSVs, offline address lookup with smart fallback,
and geodesic distance calculations. All paths resolve to the project's data/ directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from geopy.distance import geodesic

# Project data directory (next to this file)
_DATA_DIR = Path(__file__).resolve().parent / "data"


def _data_path(filename: str) -> str:
    """Return the full path for a dataset file under data/."""
    return str(_DATA_DIR / filename)


@st.cache_data(show_spinner=False)
def load_shelters_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the Tel Aviv emergency shelters dataset from CSV.

    Returns a DataFrame with columns including lat, lon, address fields, and size.
    Cached per Streamlit run to avoid repeated disk I/O.
    """
    path = path or _data_path("tlv_shelters.csv")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_addresses_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the Tel Aviv municipal address database (offline geocoding).

    Returns a DataFrame with Hebrew street names, house numbers, and Lat/Lon.
    Cached per Streamlit run.
    """
    path = path or _data_path("tlv_addresses.csv")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_streets_from_addresses(path: str | None = None) -> list[str]:
    """
    Return a sorted, unique list of Hebrew street names from the address database.

    Used for autocomplete or validation. Numeric street names are sorted last.
    """
    path = path or _data_path("tlv_addresses.csv")
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


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first column name from df that exists in candidates.

    Used to support multiple possible column names across different CSV schemas.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def lookup_address_offline(
    df_addresses: pd.DataFrame,
    street_name: str,
    house_number: str,
) -> tuple[float | None, float | None, str | None, bool, str | None]:
    """
    Resolve a Hebrew street + house number to coordinates using the local address DB.

    Returns (lat, lon, display_address, used_fallback, fallback_closest_num).
    If no exact match, falls back to the closest numeric house number on that street.
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


def find_nearest_shelters_from_coords(
    user_lat: float,
    user_lon: float,
    df: pd.DataFrame,
) -> tuple[list[dict], str | None]:
    """
    Compute the three closest shelters to (user_lat, user_lon) using geodesic distance.

    Returns (list of dicts with address, distance_m, lat, lon, size, ...) or ([], error_message).
    """
    if "lat" not in df.columns or "lon" not in df.columns:
        return [], "Shelter dataset does not include lat/lon. Regenerate data using get_shelters.py."

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
        ["Full_Address", "FULL_ADDRESS", "full_address", "Address", "address", "כתובת", "LOCATION", "location"],
    )
    size_col = pick_col(
        df,
        ["shetach_mr", "Size", "size", "CAPACITY", "capacity", "גודל"],
    )

    out: list[dict] = []
    for _, row in top.iterrows():
        out.append({
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
        })
    return out, None


def get_data_dir() -> Path:
    """Return the project data directory path (for existence checks)."""
    return _DATA_DIR


def build_satellite_map(
    user_lat: float,
    user_lon: float,
    top_3: list[dict],
):
    """
    Build an interactive Folium map centered on the user with a blue user marker
    and red markers for each of the 3 closest shelters (shield icon + tooltip).
    """
    import folium
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
