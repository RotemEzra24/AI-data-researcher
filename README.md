# Tel Aviv Emergency Shelter Locator

A web application that helps users find the closest emergency shelters in Tel Aviv. 

This project uses a LangChain tool-calling agent to process natural language inputs (e.g., "I'm at 50 Dizengoff") or raw GPS coordinates. To ensure reliability during GPS spoofing or network instability, the application performs all geocoding and geodesic distance calculations locally using municipal datasets.

## Architecture

The codebase is strictly modular, separating the frontend, AI logic, and data processing:

* `app.py`: The Streamlit frontend. Handles the chat interface, session state, layout, and interactive Folium map rendering.
* `agent_logic.py`: The LangChain agent setup. Defines the system prompt and the custom tools (`geocode_address` and `find_shelters`).
* `data_engine.py`: The data layer. Uses Pandas to load local CSVs, perform offline address matching (with a mathematical fallback for approximate house numbers), and calculate routing distances.

## Local Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/RotemEzra24/AI-data-researcher.git](https://github.com/RotemEzra24/AI-data-researcher.git)
   cd AI-data-researcher
   pip install -r requirements.txt



2. **Data**

 ##  Prepare the Data
*The application requires Tel Aviv municipality datasets. Place the following files in a data/ folder in the root directory:

*data/tlv_shelters.csv (contains latitude, longitude, and address)

*data/tlv_addresses.csv (contains the municipal address grid)

*Note: You can fetch the latest data using the provided scraping scripts (python get_shelters.py and python get_addresses.py).

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
├── app.py              # Streamlit UI
├── agent_logic.py      # LangChain tools and AgentExecutor
├── data_engine.py      # Data loading, geocoding, and map logic
├── data/               # Local CSV directory
│   ├── tlv_shelters.csv
│   └── tlv_addresses.csv
├── get_shelters.py     # Script to fetch shelter data
├── get_addresses.py    # Script to fetch address data
├── requirements.txt
├── .env                # Not committed
└── README.md
```

## License

Use and adapt as needed; data is from Tel Aviv Municipality. Ensure compliance with their terms when redistributing or using the datasets.
