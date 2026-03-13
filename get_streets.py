import requests
import pandas as pd
import json

print("Fetching 100% of Tel Aviv streets using official City Code (5000)...")

# שימוש בפילטר של סמל יישוב במקום חיפוש טקסטואלי רגיל
url = 'https://data.gov.il/api/3/action/datastore_search?resource_id=a7296d1a-f8c9-4b70-96c2-6ebb4352f8e3&limit=10000'

# הוספת הפילטר בצורה בטוחה
params = {
    'filters': json.dumps({"סמל_ישוב": 5000})
}

response = requests.get(url, params=params)
data = response.json()

records = data['result']['records']

# חילוץ הרחובות וניקוי רווחים מיותרים
streets = []
for record in records:
    street_name = record['שם_רחוב'].strip()
    # סינון רחובות ריקים או ללא שם רשמי
    if street_name and street_name != "לא רשמי":
        streets.append(street_name)

# הסרת כפילויות וסידור לפי א'-ב'
streets = sorted(list(set(streets)))

# שמירה לקובץ
df_streets = pd.DataFrame(streets, columns=['Street_Name'])
df_streets.to_csv("tlv_streets.csv", index=False)

print(f"✅ Success! Saved {len(streets)} official streets to tlv_streets.csv")