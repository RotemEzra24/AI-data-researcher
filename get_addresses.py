import os
import time
import pandas as pd
import requests

print("Downloading complete offline address database for Tel Aviv (Layer 527)...")
url = "https://gisn.tel-aviv.gov.il/arcgis/rest/services/IView2/MapServer/527/query"

all_addresses = []
offset = 0
record_count = 2000 # מושכים במנות כדי לא להפיל את השרת

while True:
    print(f"Fetching records {offset} to {offset + record_count}...")
    params = {
        "where": "1=1",
        "outFields": "*",
        "outSR": "4326", # מבקשים קואורדינטות GPS עולמיות
        "f": "json",
        "resultOffset": offset,
        "resultRecordCount": record_count
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    features = data.get('features', [])
    if not features:
        break
        
    for feature in features:
        attr = feature.get('attributes', {})
        geom = feature.get('geometry', {})
        
        # חילוץ הקואורדינטות והוספה לתכונות
        if geom:
            attr['Longitude'] = geom.get('x')
            attr['Latitude'] = geom.get('y')
            
        all_addresses.append(attr)
        
    offset += len(features)
    
    # אם קיבלנו פחות מ-2000 רשומות, סימן שהגענו לסוף
    if len(features) < record_count:
        break
    
    time.sleep(0.5) # השהייה קטנה כדי לא להציף את השרת של העירייה

df = pd.DataFrame(all_addresses)
os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "tlv_addresses.csv")
df.to_csv(out_path, index=False)
print(f"Success! Downloaded {len(df)} addresses to {out_path}")