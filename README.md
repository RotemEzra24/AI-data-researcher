# Tel Aviv Offline-First Tactical Shelter Locator

A production-grade GenAI data-engineering demo: a **LangChain tool-calling agent** that finds the three nearest emergency shelters from a user’s location. The app uses **local municipal CSVs** for offline geocoding and geodesic distance, so it works without external geocoding APIs and can bypass GPS spoofing via manual address entry. The UI is built with **Streamlit** and follows a clear separation between data, agent logic, and frontend.

## Architecture

- **Streamlit frontend (`app.py`)** — Page config, session state, chat UI, Folium map, expanders. No business or data logic.
- **Data layer (`data_engine.py`)** — Pandas-based loading of `data/tlv_shelters.csv` and `data/tlv_addresses.csv`, offline address lookup with a smart fallback (closest house number on street), and geodesic distance to rank shelters. Optional Folium satellite map builder.
- **Agent layer (`agent_logic.py`)** — LangChain tools `geocode_address` and `find_shelters`, system prompt, and cached `AgentExecutor`. The agent turns natural language (e.g. “אני בדיזנגוף 50” or raw lat/lon) into tool calls; the UI renders the list and map from tool results.

Secrets (e.g. `OPENAI_API_KEY`) are read from the environment or Streamlit secrets only; nothing is hardcoded.

## Run locally

1. **Clone and install**

   ```bash
   git clone <repo-url>
   cd ai-data-researcher
   pip install -r requirements.txt
   ```

2. **Data**

   - Place municipal CSVs in the `data/` folder:
     - `data/tlv_shelters.csv` — shelters with `lat`, `lon`, address, and optional size.
     - `data/tlv_addresses.csv` — addresses for offline geocoding (e.g. `t_rechov`, `ms_bayit`, `Latitude`, `Longitude`).
   - You can generate them (if you have the sources) with:
     - `python get_shelters.py` → writes `data/tlv_shelters.csv`
     - `python get_addresses.py` → writes `data/tlv_addresses.csv`

3. **Secrets**

   - Create a `.env` in the project root (or use [Streamlit secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)):
     ```env
     OPENAI_API_KEY=sk-...
     ```
   - Do **not** commit `.env` or `.streamlit/secrets.toml` (they are in `.gitignore`).

4. **Start the app**

   ```bash
   streamlit run app.py
   ```

   Open the URL shown in the terminal (e.g. `http://localhost:8501`). Type an address or use “Share location” (browser geolocation); the agent will geocode, find the three closest shelters, and show the list plus an interactive map and Google Maps links.

## Project layout

```text
.
├── app.py              # Streamlit UI only
├── agent_logic.py      # LangChain tools + AgentExecutor
├── data_engine.py      # Data loading, geocoding, distance, map
├── data/
│   ├── tlv_shelters.csv
│   └── tlv_addresses.csv
├── get_shelters.py     # Script to fetch shelters into data/
├── get_addresses.py    # Script to fetch addresses into data/
├── requirements.txt
├── .env                # Not committed; add OPENAI_API_KEY
└── README.md
```

## License

Use and adapt as needed; data is from Tel Aviv Municipality. Ensure compliance with their terms when redistributing or using the datasets.
