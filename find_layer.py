import requests

print("🕵️ Scanning Tel Aviv GIS Server for Address layers...")
url = "https://gisn.tel-aviv.gov.il/arcgis/rest/services/IView2/MapServer?f=json"

response = requests.get(url)
data = response.json()
layers = data.get('layers', [])

# חיפוש שכבות שקשורות לכתובות, בניינים או רחובות
print("--- Results ---")
for layer in layers:
    name = layer.get('name', '')
    if 'כתוב' in name or 'בתים' in name or 'בניינים' in name or 'רחוב' in name:
        print(f"Layer ID: {layer['id']} | Name: {name}")