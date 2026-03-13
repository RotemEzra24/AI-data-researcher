import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from prompts import build_agent_prompt

# --- 1. Page Configuration ---
st.set_page_config(page_title="Data Intelligence", page_icon="", layout="wide")
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
    Encapsulates the LangChain pandas agent configuration and interaction
    for this application.
    """

    def __init__(self, df: pd.DataFrame, is_car_data: bool) -> None:
        self.df = df
        self.is_car_data = is_car_data
        self._agent = self._create_agent(df)

    @staticmethod
    def _create_agent(df: pd.DataFrame):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            max_iterations=20,
            agent_type="openai-tools",
        )
        return agent

    def answer_question(self, user_question: str) -> str:
        """
        Build a rich system prompt and return the agent's textual answer.
        Plot creation is handled implicitly via the prompt rules.
        """
        agent_prompt = build_agent_prompt(
            user_question=user_question,
            df=self.df,
            is_car_data=self.is_car_data,
        )
        response = self._agent.invoke(agent_prompt)
        return response.get("output", "")

# --- 3. Hero Section ---
st.markdown(
    """
    <div class="hero-card">
        <h1>Data Intelligence.</h1>
        <p class="hero-subtitle">
            Upload a dataset, surface hidden patterns, and turn raw numbers into a clear narrative.
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
            By default the app loads a built-in Car Market dataset. You can optionally upload your own CSV to override it.
        </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload CSV Dataset (optional)", type="csv")

# Load default embedded dataset if no upload is provided
if uploaded_file is None:
    default_path = "israel_car_market_prices.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.caption("Using built-in sample dataset: Israeli Car Market.")
    else:
        st.error("No dataset available. Please upload a CSV file.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()
else:
    df = pd.read_csv(uploaded_file)
    st.caption("Using your uploaded dataset.")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- 4. Smart Insights Engine (Business Logic) ---
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Market Highlights</div>
        <div class="section-subtitle">Instantly calculated KPIs based on your dataset.</div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

# Check if this is our Car Market dataset
is_car_data = all(col in df.columns for col in ['Manufacturer', 'Model', 'Current_Price_ILS', 'Original_Price_ILS'])

if is_car_data:
    # 1. Total Market Value
    total_market_value = df['Current_Price_ILS'].sum() / 1_000_000  # In Millions
    with col1:
        premium_widget(
            "Market Volume",
            f"₪{total_market_value:.1f}M",
            f"Across {len(df):,} listings",
            "linear-gradient(135deg, #FF9A9E 0%, #FECFEF 99%, #FECFEF 100%)",
        )

    # 2. Best Selling Car (Most Listings)
    top_car_series = df.groupby(['Manufacturer', 'Model']).size().idxmax()
    top_car_name = f"{top_car_series[0]} {top_car_series[1]}"
    with col2:
        premium_widget(
            "Most Popular",
            top_car_name,
            "Highest number of listings",
            "linear-gradient(135deg, #667EEA 0%, #764BA2 100%)",
        )

    # 3. Best Investment (Highest Value Retention)
    # Calculate retention: Current Price / Original Price
    df['Value_Retention'] = df['Current_Price_ILS'] / df['Original_Price_ILS']
    # Group by model, get the mean retention, find the highest
    best_investment_series = df.groupby(['Manufacturer', 'Model'])['Value_Retention'].mean().idxmax()
    best_investment_name = f"{best_investment_series[0]} {best_investment_series[1]}"
    best_retention_val = df.groupby(['Manufacturer', 'Model'])['Value_Retention'].mean().max() * 100

    with col3:
        premium_widget(
            "Best Investment",
            best_investment_name,
            f"Retains {best_retention_val:.1f}% of value",
            "linear-gradient(135deg, #43E97B 0%, #38F9D7 100%)",
        )
else:
    # Generic dataset fallback
    with col1:
        premium_widget(
            "Total Records",
            f"{len(df):,}",
            "Rows in dataset",
            "linear-gradient(135deg, #667EEA 0%, #764BA2 100%)",
        )
    with col2:
        premium_widget(
            "Features",
            f"{len(df.columns)}",
            "Columns in dataset",
            "linear-gradient(135deg, #FF9A9E 0%, #FECFEF 100%)",
        )
    with col3:
        premium_widget(
            "File Size",
            f"{(df.memory_usage(deep=True).sum() / (1024*1024)):.2f} MB",
            "Memory footprint",
            "linear-gradient(135deg, #43E97B 0%, #38F9D7 100%)",
        )

st.markdown("</div>", unsafe_allow_html=True)

# --- 5. Visual Overview (Always-On Charts) ---
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Visual overview</div>
        <div class="section-subtitle">Clean, high-level views so anyone can immediately understand the distribution of your data.</div>
    """,
    unsafe_allow_html=True,
)

if is_car_data:
    viz_col1, viz_col2 = st.columns(2)

    # Price distribution
    with viz_col1:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.hist(df["Current_Price_ILS"], bins=30, color="#4F8EF7", alpha=0.9)
        ax.set_title("Distribution of current prices (ILS)", fontsize=11)
        ax.set_xlabel("Current price (ILS)")
        ax.set_ylabel("Number of listings")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # Mileage vs price
    with viz_col2:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.scatter(
            df["Mileage_km"],
            df["Current_Price_ILS"],
            alpha=0.25,
            s=10,
            color="#34C759",
        )
        ax.set_title("Mileage vs. current price", fontsize=11)
        ax.set_xlabel("Mileage (km)")
        ax.set_ylabel("Current price (ILS)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
else:
    # Generic preview and first numeric distribution
    st.dataframe(df.head(10), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        target_col = numeric_cols[0]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df[target_col].dropna(), bins=30, color="#4F8EF7", alpha=0.9)
        ax.set_title(f"Distribution of {target_col}", fontsize=11)
        ax.set_xlabel(str(target_col))
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- 6. AI Agent Initialization ---
data_agent = DataResearchAgent(df=df, is_car_data=is_car_data)

# --- 7. AI Research Chat ---
st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Ask the data</div>
        <div class="section-subtitle">Pose analytical questions or request concise, executive-level summaries.</div>
    """,
    unsafe_allow_html=True
)
user_question = st.text_input("", placeholder="For example: Which fuel type loses value fastest over time? Plot it.")

if st.button("Generate AI Insight") and user_question:
    with st.spinner("AI is analyzing the data..."):
        try:
            answer_text = data_agent.answer_question(user_question)

            st.markdown(
                """
                <div style='background: white; border-left: 4px solid #0071e3; padding: 20px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-top: 18px;'>
                """,
                unsafe_allow_html=True,
            )

            st.write(answer_text)

            if os.path.exists("temp_plot.png"):
                st.image("temp_plot.png", use_container_width=True)
                os.remove("temp_plot.png")

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("</div>", unsafe_allow_html=True)