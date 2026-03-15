from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
from pyproj import Transformer


URL = (
    "https://gisn.tel-aviv.gov.il/arcgis/rest/services/IView2/MapServer/592/query"
    "?where=1%3D1&outFields=*&f=json&returnGeometry=true&outSR=2039"
)


def looks_like_itm_epsg_2039(x: float, y: float) -> bool:
    """
    Heuristic check for Israeli Transverse Mercator (ITM), EPSG:2039.
    Typical ranges: x ~ 120k–280k, y ~ 500k–900k.
    """
    return 100_000 <= x <= 300_000 and 400_000 <= y <= 1_000_000


def itm_to_wgs84_lat_lon(x: float, y: float) -> Tuple[float, float]:
    """
    Convert ITM (EPSG:2039) x/y to WGS84 latitude/longitude (EPSG:4326).
    Returns (lat, lon).
    """
    transformer = Transformer.from_crs("EPSG:2039", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lat), float(lon)


def parse_feature(feature: Dict[str, Any]) -> Dict[str, Any]:
    attrs = feature.get("attributes") or {}
    geom = feature.get("geometry") or {}

    x = geom.get("x")
    y = geom.get("y")

    record: Dict[str, Any] = dict(attrs)
    record["x_coord"] = x
    record["y_coord"] = y

    lat = None
    lon = None
    try:
        if x is not None and y is not None:
            x_f = float(x)
            y_f = float(y)
            if looks_like_itm_epsg_2039(x_f, y_f):
                lat, lon = itm_to_wgs84_lat_lon(x_f, y_f)
            else:
                # If the service returns WGS84 already (rare for this endpoint),
                # treat (x,y) as (lon,lat).
                if -180.0 <= x_f <= 180.0 and -90.0 <= y_f <= 90.0:
                    lon = x_f
                    lat = y_f
    except Exception:
        lat = None
        lon = None

    record["lat"] = lat
    record["lon"] = lon
    return record


def fetch_shelters() -> List[Dict[str, Any]]:
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    features = payload.get("features", []) or []
    return [parse_feature(f) for f in features]


def main() -> None:
    print("Fetching data from Tel Aviv Municipality GIS API...")
    shelters = fetch_shelters()
    df = pd.DataFrame(shelters)

    # Ensure consistent column order for location fields
    preferred = ["lat", "lon", "x_coord", "y_coord"]
    cols = preferred + [c for c in df.columns if c not in preferred]
    df = df[cols]

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "tlv_shelters.csv")
    df.to_csv(out_path, index=False)
    print(f"Success! Saved {len(df)} shelters to {out_path}")


if __name__ == "__main__":
    main()