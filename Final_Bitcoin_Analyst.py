import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import xgboost as xgb
from scipy.stats import pearsonr
# import matplotlib.pyplot as plt # Not directly used for st plots
# import seaborn as sns # Not directly used for st plots
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import base64
from io import BytesIO
import os
import re

# --- Agno and AI Tool Imports (from App 2) ---
from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
import yfinance as yf

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Load Environment Variables (for AI Agents) ---
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Bitcoin Intelligence Hub",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Combined Custom CSS ---
st.markdown(
    """
<style>
    /* --- Base Dark Theme (from App 2) --- */
    .main {
        background-color: #0f1216; /* Dark background */
        color: #f8f8f2; /* Light text */
    }
    .sidebar .sidebar-content {
        background-color: #191f28; /* Darker sidebar */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f7931a !important; /* Bitcoin Orange for headers */
    }
    .stButton>button {
        background-color: #f7931a;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #e07908; /* Darker orange on hover */
    }
    /* Styling for st.expander, st.metric (Streamlit internal classes, may be fragile) */
    .css-1r6slb0, .st-expander { /* Expander content area */
        background-color: #1e2530 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid #2a313f !important;
    }
    .st-expander header button{
        color: #f7931a !important;
    }
    .css-1xarl3l, .stMetric { /* st.metric container */
        background-color: #1e2530;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #f7931a;
        margin-bottom: 10px; /* Added margin */
    }
    .stMetric label { color: #a0a0a0 !important; }
    .stMetric value { color: #f8f8f2 !important; }
    /* Metric delta colors are handled by Streamlit's default positive/negative classes */
    .stMetric .st-emotion-cache-1xarl3l span[data-testid="stMetricDelta"] > div:nth-child(1) { /* Positive delta */
        color: #28a745 !important;
    }
    .stMetric .st-emotion-cache-1xarl3l span[data-testid="stMetricDelta"] > div:nth-child(2) { /* Negative delta */
         color: #dc3545 !important;
    }


    /* --- Styles from App 1 (Predictive Model) adapted for Dark Theme --- */
    .main-header-predictive {
        font-size: 2.2rem; /* Slightly smaller to fit mode context */
        font-weight: 700;
        color: #f7931a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header-predictive {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f7931a;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .info-text-predictive {
        font-size: 1rem;
        color: #d0d0d0; /* Lighter text for dark theme */
        margin-bottom: 1rem;
    }
    .highlight-predictive {
        color: #f7931a; /* Bitcoin Orange */
        font-weight: 600;
    }
    .prediction-up-predictive {
        font-size: 1.5rem; /* Adjusted from App 1's second CSS */
        font-weight: 700;
        color: #28a745; /* Green */
    }
    .prediction-down-predictive {
        font-size: 1.5rem; /* Adjusted from App 1's second CSS */
        font-weight: 700;
        color: #dc3545; /* Red */
    }
    .custom-metric-container-predictive {
        background-color: #1e2530; /* Dark background */
        border-radius: 8px; /* Slightly adjusted */
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2); /* Adjusted shadow for dark */
        border-left: 3px solid #f7931a; /* Bitcoin orange accent */
        margin-bottom: 10px;
    }
    .custom-metric-container-predictive h5 {
        font-size: 0.9rem; /* Adjusted */
        margin-bottom: 0.5rem;
        color: #b0b0b0; /* Lighter gray for label */
    }
    .custom-metric-container-predictive .value-highlight { /* New class for the value */
        font-size: 1.6rem; /* Adjusted */
        font-weight: 700;
        color: #f8f8f2; /* White/very light text for value */
    }
    .footer-predictive {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #888888; /* Medium Gray - Okay for dark theme */
    }
    /* Ensure tabs also fit dark theme */
    .stTabs [data-baseweb="tab-list"] {
gap: 24px;
}
.stTabs [data-baseweb="tab"] {
height: 50px;
        white-space: pre-wrap;
background-color: transparent;
border-radius: 4px 4px 0px 0px;
gap: 1px;
padding-top: 10px;
padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e2530; /* Active tab background */
        color: #f7931a; /* Active tab text color */
    }
    /* Dataframe styling */
    .stDataFrame {
        background-color: #1e2530;
    }
    /* Chat input */
    .stChatInput {
        background-color: #191f28;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Helper Functions ---

# From App 1: For CSV download
def get_download_link(df, filename, text):
    """Generate a link to download the dataframe as a CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# From App 2: For fixing LLM text formatting
def fix_formatting(text):
    """Fix common text formatting issues in LLM-generated responses."""
    if not text:
        return text
    text = re.sub(r"(\d[,\d]*)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z]),([a-zA-Z])", r"\1, \2", text)
    text = re.sub(r"\.([a-zA-Z])", r". \1", text)
    text = re.sub(r"(while|for|or|and)(the|a|to|in|is|at)", r"\1 \2", text)
    text = re.sub(r"([0-9])or even([0-9])", r"\1 or even \2", text)
    return text

# From App 1: For trading strategy (if uncommented)
def calculate_max_drawdown(returns_series):
    """Calculate maximum drawdown percentage"""
    rolling_max = returns_series.cummax()
    drawdown = (returns_series / rolling_max - 1) * 100
    max_drawdown = drawdown.min()
    return abs(max_drawdown) # Return positive value for drawdown


# --- AI Agent Definitions (from App 2) ---
def process_agent_output(agent_response):
    """Process agent output to fix formatting issues"""
    if hasattr(agent_response, "content"):
        agent_response.content = fix_formatting(agent_response.content)
    return agent_response

if GROQ_API_KEY and GOOGLE_API_KEY:
    bitcoin_news_agent = Agent(
        name="Bitcoin News Agent",
        role="Search for the latest Bitcoin news and market sentiment",
        model=Groq(id="llama3-8b-8192", api_key=GROQ_API_KEY),
        tools=[DuckDuckGoTools()],
        instructions=[
            "Focus only on Bitcoin-related news",
            "Prioritize latest developments, price movements, and market sentiment",
            "Include sources for all information",
            "Present information in a clear, organized format",
            "Always add spaces between numbers and words (e.g., '50,000 level' not '50,000level')",
            "Always add spaces after commas and periods",
            "Use proper spacing in price ranges (e.g., '$45,000 to $50,000' not '$45,000to$50,000')",
        ],
        show_tool_calls=False, # Keep it cleaner for UI
        markdown=True,
    )

    bitcoin_analysis_agent = Agent(
        name="Bitcoin Analysis Agent",
        model=Gemini(id="gemini-1.5-flash-latest", api_key=GOOGLE_API_KEY),
        tools=[
            YFinanceTools(
                stock_price=True, stock_fundamentals=True, company_news=True
            ),
        ],
        instructions=[
            "Focus only on Bitcoin (BTC-USD) technical and fundamental analysis",
            "Use tables and bullet points to display data clearly",
            "Provide price targets and support/resistance levels",
            "Summarize trading volume and market capitalization trends",
            "Always add spaces between numbers and words (e.g., '50,000 level' not '50,000level')",
            "Always add spaces after commas and periods",
            "Use proper spacing in price ranges (e.g., '$45,000 to $50,000' not '$45,000to$50,000')",
        ],
        show_tool_calls=False,
        markdown=True,
    )

    bitcoin_chatbot = Agent(
        name="Bitcoin Chatbot",
        model=Gemini(id="gemini-1.5-flash-latest", api_key=GOOGLE_API_KEY),
        tools=[DuckDuckGoTools(), YFinanceTools(stock_price=True)],
        instructions=[
            "Provide clear and concise answers to Bitcoin-related questions",
            "Use friendly, conversational tone",
            "Include factual data when available",
            "Correct misconceptions about Bitcoin when appropriate",
            "Always add spaces between numbers and words (e.g., '50,000 level' not '50,000level')",
            "Always add spaces after commas and periods",
            "Use proper spacing in price ranges (e.g., '$45,000 to $50,000' not '$45,000to$50,000')",
        ],
        show_tool_calls=False,
        markdown=True,
    )

    bitcoin_intel_agent = Agent(
        team=[bitcoin_analysis_agent, bitcoin_news_agent],
        model=Groq(id="llama3-8b-8192", api_key=GROQ_API_KEY),
        instructions=[
            "Focus only on Bitcoin analysis",
            "First provide technical and price analysis using the Bitcoin Analysis Agent.",
            "Then summarize the latest news and developments using the Bitcoin News Agent.",
            "Include sources for all information if provided by the agents.",
            "Conclude with a brief overall outlook based on the combined information.",
            "Always add spaces between numbers and words (e.g., '50,000 level' not '50,000level').",
            "Always add spaces after commas and periods.",
            "Ensure proper spacing between all words and numbers.",
            "Use proper spacing in price ranges (e.g., '$45,000 to $50,000' not '$45,000to$50,000')",
        ],
        show_tool_calls=False,
        markdown=True,
    )
else:
    bitcoin_news_agent = None
    bitcoin_analysis_agent = None
    bitcoin_chatbot = None
    bitcoin_intel_agent = None
    st.sidebar.warning("API keys for Groq or Google not found. AI features will be disabled.")


# --- Bitcoin Data Fetching for Live Charts (from App 2) ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_bitcoin_data_live(timeframe_str="1mo"):
    ticker = "BTC-USD"
    period_map = {
        "1d": "1d", "5d": "5d", "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
        "1y": "1y", "ytd": "ytd", "max": "max"
    }
    interval_map = { # Adjusted intervals for yfinance
        "1d": "2m", "5d": "15m", "1mo": "90m", "3mo": "1d", "6mo": "1d",
        "1y": "1wk", "ytd": "1wk", "max": "1wk"
    }
    data = yf.download(ticker, period=period_map.get(timeframe_str, "1mo"),
                       interval=interval_map.get(timeframe_str, "90m"))
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data


# --- Predictive Modeling Functions (from App 1) ---

# 1. LOAD AND PREPARE DATA
@st.cache_data()
def load_data_predictive(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error(
            "Unsupported file format. Please upload a CSV or Excel file."
        )
        return None
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    return df

@st.cache_data()
def feature_engineering_predictive(df):
    data = df.copy()
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Dataset must contain columns: {', '.join(required_cols)}")
        return pd.DataFrame() # Return empty df if required cols are missing

    for window in [5, 10, 20, 50, 100]:
        data[f"MA_{window}"] = data["Close"].rolling(window=window).mean()
        data[f"Volume_MA_{window}"] = (
            data["Volume"].rolling(window=window).mean()
        )
    for window in [1, 3, 5, 10]:
        data[f"Price_Change_{window}d"] = data["Close"].pct_change(window)
        data[f"Volume_Change_{window}d"] = data["Volume"].pct_change(window)
    for window in [5, 10, 20]:
        data[f"Volatility_{window}d"] = (
            data["Close"].rolling(window).std()
        )

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean() # Added min_periods
        avg_loss = loss.rolling(window=period, min_periods=1).mean() # Added min_periods
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data["RSI_14"] = calculate_rsi(data["Close"], 14)
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    market_columns = ["SP500_Index", "Gold_USD", "DXY_Index", "US10Y_Rate"]
    existing_columns = [col for col in market_columns if col in data.columns]
    if "SP500_Index" in existing_columns:
        data["SP500_Return_1d"] = data["SP500_Index"].pct_change()
        data["BTC_SP500_Ratio"] = data["Close"] / data["SP500_Index"]
    if "Gold_USD" in existing_columns:
        data["Gold_Return_1d"] = data["Gold_USD"].pct_change()
        data["BTC_Gold_Ratio"] = data["Close"] / data["Gold_USD"]
    if "DXY_Index" in existing_columns:
        data["DXY_Return_1d"] = data["DXY_Index"].pct_change()
    if "US10Y_Rate" in existing_columns:
        data["Rate_Change_1d"] = data["US10Y_Rate"].diff()

    if "Date" in data.columns:
        data["Day_of_Week"] = data["Date"].dt.dayofweek
        data["Month"] = data["Date"].dt.month
        data["Year"] = data["Date"].dt.year

    data["Daily_Range"] = data["High"] - data["Low"]
    data["Daily_Range_Ratio"] = data["Daily_Range"] / data["Open"]
    data["Price_Position"] = (data["Close"] - data["Low"]) / (
        data["High"] - data["Low"] + 1e-6 # Avoid division by zero
    )

    hash_columns = ["Hash_Rate", "Transactions_per_Block"]
    existing_hash_columns = [
        col for col in hash_columns if col in data.columns
    ]
    if "Hash_Rate" in existing_hash_columns:
        data["Hash_Rate_Change"] = data["Hash_Rate"].pct_change()
    if "Transactions_per_Block" in existing_hash_columns:
        data["Trans_per_Block_Change"] = data[
            "Transactions_per_Block"
        ].pct_change()

    for lag in [1, 2, 3, 5, 7]:
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
        data[f"Volume_Lag_{lag}"] = data["Volume"].shift(lag)
    data["Target"] = data["Close"].shift(-1)
    data["Return_1d"] = data["Close"].pct_change()
    data["Direction"] = (data["Return_1d"] > 0).astype(int)
    data["Target_Direction"] = (data["Target"] > data["Close"]).astype(int)
    data = data.dropna()
    return data

def prepare_data_for_modeling_predictive(
    df, test_size=0.2, scaler_type="standard"
):
    data = df.copy()
    X = data.drop(
        ["Date", "Target", "Target_Direction", "Direction", "Open", "High", "Low", "Close"],
        axis=1,
        errors='ignore' # Ignore if some columns are not present (e.g. from sample data)
    )
    y = data["Target"]
    feature_names = X.columns.tolist()
    split_idx = int(len(data) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=feature_names, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

@st.cache_data()
def feature_importance_analysis_predictive(data, target="Target"):
    features = data.drop(
        ["Date", "Target", "Target_Direction", "Direction"], axis=1, errors='ignore'
    )
    correlations = []
    for col in features.columns:
        if pd.api.types.is_numeric_dtype(features[col]) and target in data.columns and pd.api.types.is_numeric_dtype(data[target]):
            # Ensure no NaNs in correlation calculation
            valid_indices = features[col].notna() & data[target].notna()
            if len(features[col][valid_indices]) > 1 and len(data[target][valid_indices]) > 1: # Pearsonr needs at least 2 samples
                corr, _ = pearsonr(features[col][valid_indices], data[target][valid_indices])
                correlations.append((col, corr))
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    return correlations

# 2. BUILD BASE MODELS
def create_base_models_predictive(train_mode="fast"):
    if train_mode == "fast":
        models = {
            "xgboost": xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
            ),
            "ridge": Ridge(alpha=1.0, random_state=42),
            "lasso": Lasso(alpha=0.01, random_state=42),
            "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
        }
        return models
    else: # hyperparameter_tuning
        param_grids = {
            "xgboost": {
                "n_estimators": [100, 150], "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5], "subsample": [0.8, 0.9],
                "colsample_bytree": [0.8, 0.9],
            },
            "random_forest": {
                "n_estimators": [100, 150], "max_depth": [7, 10],
                "min_samples_split": [2, 5], "min_samples_leaf": [1, 2],
            },
            "ridge": {"alpha": [0.1, 1.0, 10.0]},
            "lasso": {"alpha": [0.001, 0.01, 0.1]},
            "elastic_net": {
                "alpha": [0.01, 0.1], "l1_ratio": [0.3, 0.5, 0.7]
            },
            "gradient_boosting": {
                "n_estimators": [100, 150], "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5], "subsample": [0.8, 0.9],
            },
        }
        models = {
            "xgboost": xgb.XGBRegressor(random_state=42),
            "random_forest": RandomForestRegressor(random_state=42),
            "ridge": Ridge(random_state=42),
            "lasso": Lasso(random_state=42),
            "elastic_net": ElasticNet(random_state=42),
            "gradient_boosting": GradientBoostingRegressor(random_state=42),
        }
        return models, param_grids

def tune_model_predictive(model, param_grid, X_train, y_train, model_name):
    tscv = TimeSeriesSplit(n_splits=3) # Reduced splits for faster tuning in app
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=tscv,
        scoring="neg_mean_squared_error", n_jobs=-1, verbose=0,
    )
    with st.spinner(f"Tuning {model_name}... This might take a moment."):
        grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, -grid_search.best_score_

# 3. IMPLEMENT STACKING
def train_base_models_predictive(models, X_train, y_train, cv=3, progress_bar=None): # Reduced CV for app
    n_train = X_train.shape[0]
    n_models = len(models)
    oof_preds = np.zeros((n_train, n_models))
    trained_models = {}
    model_metrics = {}
    tscv = TimeSeriesSplit(n_splits=cv)

    for model_idx, (model_name, model) in enumerate(models.items()):
        if progress_bar:
            progress_bar.progress(
                (model_idx) / len(models), text=f"Training {model_name}..."
            )
        model_oof_preds = np.zeros(n_train)
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_cv_train, y_cv_train)
            model_oof_preds[val_idx] = model.predict(X_cv_val)

        oof_preds[:, model_idx] = model_oof_preds
        final_model = model.__class__(**model.get_params())
        final_model.fit(X_train, y_train)
        trained_models[model_name] = final_model

        mse = mean_squared_error(y_train, model_oof_preds)
        model_metrics[model_name] = {
            "RMSE": np.sqrt(mse), "MAE": mean_absolute_error(y_train, model_oof_preds),
            "R²": r2_score(y_train, model_oof_preds),
        }
    if progress_bar:
        progress_bar.progress(1.0, text="Base model training complete!")
    oof_df = pd.DataFrame(oof_preds, columns=list(models.keys()), index=X_train.index)
    return trained_models, oof_df, model_metrics

def train_meta_model_predictive(oof_preds, y_train, meta_model_type="ridge"):
    if meta_model_type == "xgboost":
        meta_model = xgb.XGBRegressor(
            n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42 # Faster
        )
    elif meta_model_type == "elastic_net":
        meta_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    else: # Default to ridge
        meta_model = Ridge(alpha=1.0, random_state=42)

    meta_model.fit(oof_preds, y_train)
    importances = getattr(meta_model, 'feature_importances_', getattr(meta_model, 'coef_', None))
    model_weights = {}
    if importances is not None:
        for model_name, importance in zip(oof_preds.columns, importances):
            model_weights[model_name] = importance
    return meta_model, model_weights

def generate_test_predictions_predictive(trained_models, X_test):
    n_test = X_test.shape[0]
    n_models = len(trained_models)
    test_preds = np.zeros((n_test, n_models))
    for model_idx, (model_name, model) in enumerate(trained_models.items()):
        test_preds[:, model_idx] = model.predict(X_test)
    test_preds_df = pd.DataFrame(
        test_preds, columns=list(trained_models.keys()), index=X_test.index
    )
    return test_preds_df

# 4. EVALUATE
def evaluate_ensemble_predictive(meta_model, test_preds_df, y_test, model_names, original_data=None):
    ensemble_preds = meta_model.predict(test_preds_df)
    ensemble_metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, ensemble_preds)),
        "MAE": mean_absolute_error(y_test, ensemble_preds),
        "R²": r2_score(y_test, ensemble_preds),
    }
    individual_metrics = {}
    for model_name in model_names:
        model_preds = test_preds_df[model_name]
        individual_metrics[model_name] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, model_preds)),
            "MAE": mean_absolute_error(y_test, model_preds),
            "R²": r2_score(y_test, model_preds),
        }
    results = pd.DataFrame({"Actual": y_test, "Predicted": ensemble_preds})
    if original_data is not None and 'Close' in original_data.columns:
        # Align indices before joining
        original_data_aligned = original_data.loc[y_test.index]
        results = results.join(original_data_aligned[['Close']])
        if 'Close' in results.columns: # Check if join was successful
            results["Actual_Return"] = (results["Actual"] - results["Close"]) / (results["Close"] + 1e-9)
            results["Predicted_Return"] = (results["Predicted"] - results["Close"]) / (results["Close"] + 1e-9)
            results["Actual_Direction"] = (results["Actual_Return"] > 0).astype(int)
            results["Predicted_Direction"] = (results["Predicted_Return"] > 0).astype(int)
            direction_accuracy = (
                results["Actual_Direction"] == results["Predicted_Direction"]
            ).mean()
            ensemble_metrics["Direction_Accuracy"] = direction_accuracy
    return results, ensemble_preds, ensemble_metrics, individual_metrics

def predict_next_day_predictive(trained_models, meta_model, latest_data, scaler_object, feature_names):
    if latest_data.empty or not all(f in latest_data.columns for f in feature_names):
        st.error("Insufficient data or missing features for next day prediction.")
        return {
            'current_price': 0, 'predicted_price': 0, 'change_amount': 0,
            'change_percent': 0, 'direction': "N/A", 'base_predictions': {}
        }

    latest_features = latest_data[feature_names].iloc[-1:]
    latest_scaled = scaler_object.transform(latest_features) # Use the passed scaler_object
    base_predictions = {}
    for model_name, model in trained_models.items():
        base_predictions[model_name] = model.predict(latest_scaled)[0]

    meta_input = pd.DataFrame([base_predictions])
    final_prediction = meta_model.predict(meta_input)[0]
    current_close = latest_data["Close"].iloc[-1] if "Close" in latest_data else 0

    change_amount = final_prediction - current_close
    change_percent = (change_amount / (current_close  + 1e-9)) * 100
    direction = "UP" if change_amount > 0 else "DOWN"

    return {
        "current_price": current_close, "predicted_price": final_prediction,
        "change_amount": change_amount, "change_percent": change_percent,
        "direction": direction, "base_predictions": base_predictions,
    }

# --- Plotting Functions for Predictive Model (from App 1, adapted for dark theme) ---
def plot_price_history_predictive(data):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Date"], y=data["Close"], mode="lines", name="Bitcoin Price",
            line=dict(color="#f7931a", width=2),
            hovertemplate="<b>Date</b>: %{x}<br><b>Price</b>: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=data["Date"], y=data["Volume"], name="Volume",
            marker_color="rgba(135, 206, 235, 0.5)", opacity=0.7, yaxis="y2",
            hovertemplate="<b>Date</b>: %{x}<br><b>Volume</b>: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Bitcoin Price History with Trading Volume",
        xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Price (USD)", gridcolor="rgba(255,255,255,0.1)", tickformat="$,.0f"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')),
        hovermode="x unified", template="plotly_dark", height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),
            ])
        ),
    )
    return fig

def plot_feature_importance_predictive(correlations):
    if not correlations:
        return go.Figure().update_layout(title="No feature correlations to display.", template="plotly_dark")
    top_features = correlations[:20]
    features = [item[0] for item in top_features]
    corr_values = [item[1] for item in top_features]
    colors = ["#f7931a" if c > 0 else "#6c757d" for c in corr_values] # Orange for pos, gray for neg
    fig = go.Figure()
    fig.add_trace(go.Bar(y=features, x=corr_values, orientation="h", marker=dict(color=colors)))
    fig.update_layout(
        title="Top 20 Features by Correlation with Target",
        xaxis=dict(title="Correlation Coefficient", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Feature", autorange="reversed", gridcolor="rgba(255,255,255,0.1)"), # Reversed for top-to-bottom
        template="plotly_dark", height=600,
        margin=dict(l=150, r=20, t=60, b=20), # Increased left margin for feature names
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def plot_model_weights_predictive(model_weights):
    if not model_weights:
        return go.Figure().update_layout(title="No model weights to display.", template="plotly_dark")
    sorted_weights = sorted(model_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    models = [item[0] for item in sorted_weights]
    weights = [item[1] for item in sorted_weights]
    colors = ["#f7931a" if w > 0 else "#6c757d" for w in weights]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=models, x=weights, orientation="h", marker=dict(color=colors)))
    fig.update_layout(
        title="Model Weights in Ensemble",
        xaxis=dict(title="Weight", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Model", autorange="reversed", gridcolor="rgba(255,255,255,0.1)"),
        template="plotly_dark", height=400,
        margin=dict(l=150, r=20, t=60, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def plot_prediction_results_predictive(results):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results.index, y=results["Actual"], mode="lines", name="Actual Price",
            line=dict(color="#17a2b8", width=2), # Cyan for actual
            hovertemplate="<b>Date</b>: %{x}<br><b>Actual</b>: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results.index, y=results["Predicted"], mode="lines", name="Predicted Price",
            line=dict(color="#f7931a", width=2), # Orange for predicted
            hovertemplate="<b>Date</b>: %{x}<br><b>Predicted</b>: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Actual vs Predicted Bitcoin Prices",
        xaxis=dict(title="Date", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Price (USD)", gridcolor="rgba(255,255,255,0.1)", tickformat="$,.0f"),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')),
        hovermode="x unified", template="plotly_dark", height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def plot_error_distribution_predictive(results):
    if 'Actual' not in results.columns or 'Predicted' not in results.columns:
        return go.Figure().update_layout(title="Not enough data for error distribution.", template="plotly_dark")
    results["Error"] = results["Actual"] - results["Predicted"]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=results["Error"], nbinsx=30, marker_color="#f7931a", opacity=0.7)
    )
    fig.update_layout(
        title="Prediction Error Distribution",
        xaxis=dict(title="Error (USD)", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.1)"),
        template="plotly_dark", height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def plot_prediction_scatter_predictive(results):
    if 'Actual' not in results.columns or 'Predicted' not in results.columns:
        return go.Figure().update_layout(title="Not enough data for scatter plot.", template="plotly_dark")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results["Actual"], y=results["Predicted"], mode="markers",
            marker=dict(color="#f7931a", size=8, opacity=0.7, line=dict(width=1, color="#f8f8f2")),
            hovertemplate="<b>Actual</b>: $%{x:,.2f}<br><b>Predicted</b>: $%{y:,.2f}<extra></extra>",
        )
    )
    min_val = min(results["Actual"].min(), results["Predicted"].min())
    max_val = max(results["Actual"].max(), results["Predicted"].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val], mode="lines",
            name="Perfect Prediction", line=dict(color="#f8f8f2", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Actual vs Predicted Price Scatter Plot",
        xaxis=dict(title="Actual Price (USD)", gridcolor="rgba(255,255,255,0.1)", tickformat="$,.0f"),
        yaxis=dict(title="Predicted Price (USD)", gridcolor="rgba(255,255,255,0.1)", tickformat="$,.0f"),
        template="plotly_dark", height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# --- Main Application ---
def run_hub():
    # --- Sidebar ---
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 10px; background-color: #1e2530; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #f7931a; margin-bottom: 0px;">₿itcoin Hub</h2>
    </div>
    """, unsafe_allow_html=True)

    # Fetch current Bitcoin price for sidebar
    try:
        btc_info = yf.Ticker("BTC-USD").info
        current_price = btc_info.get('regularMarketPrice')
        previous_close = btc_info.get('previousClose')

        if current_price is not None and previous_close is not None:
            price_change = current_price - previous_close
            price_change_pct = (price_change / previous_close) * 100
            # Streamlit's st.metric handles delta color automatically based on positive/negative
            st.sidebar.metric(
                label="BTC Current Price",
                value=f"${current_price:,.2f}",
                delta=f"{price_change_pct:.2f}%"
            )
        else:
            st.sidebar.text("Live price unavailable.")
    except Exception as e:
        st.sidebar.text(f"Error fetching live price.") # Simplified error for sidebar


    st.sidebar.markdown("### Select Mode")
    app_mode = st.sidebar.radio(
        "Choose a tool:",
        (
            "Predictive Modeling & Forecasting",
            "Live Market Charts",
            "AI Market Analysis",
            "AI News Feed",
            "Bitcoin AI Chat",
        ),
        label_visibility="collapsed"
    )
    st.sidebar.markdown("---")

    # --- Predictive Modeling Mode ---
    if app_mode == "Predictive Modeling & Forecasting":
        st.markdown("<h1 class='main-header-predictive'>Bitcoin Price Prediction</h1>", unsafe_allow_html=True)

        # Sidebar controls for this mode
        st.sidebar.markdown("### Predictive Model Controls")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Bitcoin dataset (CSV or Excel)", type=["csv", "xlsx", "xls"]
        )
        use_sample_data = st.sidebar.checkbox(
            "Use sample data", value=True if uploaded_file is None else False
        )

        st.sidebar.markdown("#### Model Configuration")
        train_mode_predictive = st.sidebar.radio("Training Mode", ["fast", "hyperparameter_tuning"], key="pred_train_mode")
        test_size_predictive = st.sidebar.slider("Test Data Size (%)", 10, 40, 20, key="pred_test_size")
        # This key 'pred_scaler' is for the user's CHOICE of scaler type (string)
        scaler_type_predictive = st.sidebar.selectbox("Feature Scaling", ["standard", "minmax"], key="pred_scaler")
        meta_model_type_predictive = st.sidebar.selectbox("Meta-Model", ["ridge", "elastic_net", "xgboost"], key="pred_meta")

        st.sidebar.markdown("#### Visualization Options")
        show_correlation_predictive = st.sidebar.checkbox("Show Feature Correlations", value=True, key="pred_corr")

        tabs_predictive = st.tabs(["Overview", "Data & Price Analysis", "Model Training & Performance", "Predictions"])

        with tabs_predictive[0]: # Overview
            st.markdown("<h2 class='sub-header-predictive'>System Overview</h2>", unsafe_allow_html=True)
            st.markdown("""
            <p class='info-text-predictive'>
            This section provides an advanced ensemble machine learning model for Bitcoin price prediction.
            You can upload your own data or use sample data to train models and forecast prices.
            </p>
            **Key Features:**
            - Historical price analysis and visualization.
            - Automated feature engineering.
            - Ensemble prediction with multiple base models (Stacking).
            - Optional hyperparameter tuning for base models.
            - Next-day price forecasting.

            **Instructions:**
            1.  Use the sidebar to upload your Bitcoin price dataset (CSV/Excel with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume') or select 'Use sample data'.
            2.  Configure model parameters in the sidebar (Training Mode, Test Size, Scaler, Meta-Model).
            3.  Navigate through the tabs:
                *   **Data & Price Analysis**: View raw data, price history, and feature correlations.
                *   **Model Training & Performance**: Train the models and evaluate their performance.
                *   **Predictions**: See next-day forecasts and individual model contributions.
            """, unsafe_allow_html=True)
            st.image("https://www.simplilearn.com/ice9/free_resources_article_thumb/bitcoin-mining-1.jpg", caption="Bitcoin Trading", use_container_width=True)

        df_predictive = None
        if use_sample_data:
            @st.cache_data
            def load_sample_data_predictive():
                dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="D")
                n = len(dates)
                np.random.seed(42)
                # More realistic price generation
                price = 20000
                prices = []
                for _ in range(n):
                    price += np.random.normal(0, price * 0.02) # Daily volatility of 2%
                    price = max(1000, price) # Floor price
                    prices.append(price)
                close_prices = np.array(prices)

                open_prices = close_prices * np.random.normal(loc=1.0, scale=0.005, size=n)
                high_prices = np.maximum(close_prices, open_prices) * np.random.normal(loc=1.01, scale=0.01, size=n)
                low_prices = np.minimum(close_prices, open_prices) * np.random.normal(loc=0.99, scale=0.01, size=n)
                volume = np.random.lognormal(mean=22, sigma=0.5, size=n)

                df = pd.DataFrame({
                    "Date": dates, "Open": open_prices, "High": high_prices,
                    "Low": low_prices, "Close": close_prices, "Volume": volume,
                })
                return df
            df_predictive = load_sample_data_predictive()
            st.sidebar.info("Using sample Bitcoin data.")
        elif uploaded_file is not None:
            df_predictive = load_data_predictive(uploaded_file)
            if df_predictive is None:
                st.error("Error loading data. Please check the file format and contents.")

        if df_predictive is not None and not df_predictive.empty:
            required_cols_check = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df_predictive.columns for col in required_cols_check):
                st.error(f"Uploaded data is missing one or more required columns: {', '.join(required_cols_check)}")
                df_predictive = None # Invalidate df

        if df_predictive is not None and not df_predictive.empty:
            with st.spinner("Performing feature engineering..."):
                data_featured_predictive = feature_engineering_predictive(df_predictive)

            if data_featured_predictive.empty:
                st.error("Feature engineering failed. Please check your data.")
            else:
                with tabs_predictive[1]: # Data & Price Analysis
                    st.markdown("<h2 class='sub-header-predictive'>Data & Price Analysis</h2>", unsafe_allow_html=True)
                    with st.expander("Preview Raw Data"):
                        st.dataframe(df_predictive.head())
                        st.text(f"Dataset Shape: {df_predictive.shape}")

                    st.plotly_chart(plot_price_history_predictive(df_predictive), use_container_width=True)

                    st.markdown("<h3 class='sub-header-predictive'>Price Statistics (from loaded data)</h3>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    latest_close = df_predictive["Close"].iloc[-1]
                    prev_close = df_predictive["Close"].iloc[-2] if len(df_predictive) > 1 else latest_close
                    delta_pct = ((latest_close - prev_close) / (prev_close + 1e-9)) * 100

                    col1.metric(label="Latest Price", value=f"${latest_close:,.2f}", delta=f"{delta_pct:.2f}%")
                    col2.metric(label="All-Time High", value=f"${df_predictive['Close'].max():,.2f}")
                    col3.metric(label="All-Time Low", value=f"${df_predictive['Close'].min():,.2f}")

                    fig_dist = px.histogram(
                        df_predictive, x="Close", nbins=50, title="Bitcoin Price Distribution",
                        labels={"Close": "Price (USD)"}, color_discrete_sequence=["#f7931a"],
                    )
                    fig_dist.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_dist, use_container_width=True)

                    if show_correlation_predictive:
                        st.markdown("<h3 class='sub-header-predictive'>Feature Correlation Analysis</h3>", unsafe_allow_html=True)
                        correlations_predictive = feature_importance_analysis_predictive(data_featured_predictive)
                        st.plotly_chart(plot_feature_importance_predictive(correlations_predictive), use_container_width=True)

                with tabs_predictive[2]: # Model Training & Performance
                    st.markdown("<h2 class='sub-header-predictive'>Model Training & Evaluation</h2>", unsafe_allow_html=True)
                    if st.button("Start Model Training", key="start_training_btn"):
                        # scaler_type_predictive is the STRING from the selectbox (e.g., "standard")
                        X_train, X_test, y_train, y_test, fitted_scaler_object, feature_names = prepare_data_for_modeling_predictive(
                            data_featured_predictive, test_size=test_size_predictive / 100, scaler_type=scaler_type_predictive
                        )

                        models_to_train = {}
                        if train_mode_predictive == "hyperparameter_tuning":
                            with st.spinner("Initializing models for hyperparameter tuning..."):
                                base_models, param_grids = create_base_models_predictive(train_mode_predictive)
                                tuned_models_dict = {}
                                tuning_progress = st.progress(0, text="Starting hyperparameter tuning...")
                                for i, (name, model) in enumerate(base_models.items()):
                                    tuning_progress.progress((i) / len(base_models), text=f"Tuning {name}...")
                                    best_model, _, _ = tune_model_predictive(model, param_grids[name], X_train, y_train, name)
                                    tuned_models_dict[name] = best_model
                                tuning_progress.progress(1.0, text="Hyperparameter tuning complete!")
                                models_to_train = tuned_models_dict
                        else:
                            models_to_train = create_base_models_predictive(train_mode_predictive)

                        progress_bar_train = st.progress(0, text="Starting base model training...")
                        trained_models, oof_df, model_metrics_train = train_base_models_predictive(
                            models_to_train, X_train, y_train, progress_bar=progress_bar_train
                        )

                        with st.spinner("Training ensemble meta-model..."):
                            meta_model, model_weights = train_meta_model_predictive(oof_df, y_train, meta_model_type_predictive)

                        with st.spinner("Generating test predictions and evaluating..."):
                            test_preds_df = generate_test_predictions_predictive(trained_models, X_test)
                            results_df, _, ensemble_metrics, individual_metrics = evaluate_ensemble_predictive(
                                meta_model, test_preds_df, y_test, list(trained_models.keys()), original_data=data_featured_predictive
                            )

                        st.markdown("<h3 class='sub-header-predictive'>Ensemble Model Performance</h3>", unsafe_allow_html=True)
                        m_cols = st.columns(4)
                        m_cols[0].markdown(f"<div class='custom-metric-container-predictive'><h5>RMSE</h5><p class='value-highlight'>${ensemble_metrics['RMSE']:.2f}</p></div>", unsafe_allow_html=True)
                        m_cols[1].markdown(f"<div class='custom-metric-container-predictive'><h5>MAE</h5><p class='value-highlight'>${ensemble_metrics['MAE']:.2f}</p></div>", unsafe_allow_html=True)
                        m_cols[2].markdown(f"<div class='custom-metric-container-predictive'><h5>R²</h5><p class='value-highlight'>{ensemble_metrics['R²']:.4f}</p></div>", unsafe_allow_html=True)
                        if "Direction_Accuracy" in ensemble_metrics:
                            m_cols[3].markdown(f"<div class='custom-metric-container-predictive'><h5>Direction Acc.</h5><p class='value-highlight'>{ensemble_metrics['Direction_Accuracy']*100:.2f}%</p></div>", unsafe_allow_html=True)

                        st.markdown("<h4 class='sub-header-predictive' style='font-size:1.2rem; margin-top:1.5rem;'>Base Model Comparison (on Test Set)</h4>", unsafe_allow_html=True)
                        metrics_display_df = pd.DataFrame(individual_metrics).T.reset_index().rename(columns={'index':'Model'})
                        st.dataframe(metrics_display_df.style.format({'RMSE': '${:.2f}', 'MAE': '${:.2f}', 'R²': '{:.4f}'}))

                        st.plotly_chart(plot_model_weights_predictive(model_weights), use_container_width=True)
                        st.markdown("<h3 class='sub-header-predictive'>Prediction Visualizations</h3>", unsafe_allow_html=True)
                        st.plotly_chart(plot_prediction_results_predictive(results_df), use_container_width=True)
                        st.plotly_chart(plot_prediction_scatter_predictive(results_df), use_container_width=True)
                        st.plotly_chart(plot_error_distribution_predictive(results_df), use_container_width=True)

                        st.session_state["pred_trained_models"] = trained_models
                        st.session_state["pred_meta_model"] = meta_model
                        # Store the FITTED scaler object with a NEW key
                        st.session_state["fitted_scaler_object"] = fitted_scaler_object
                        st.session_state["pred_feature_names"] = feature_names
                        st.session_state["pred_data_featured"] = data_featured_predictive
                        st.session_state["pred_models_ready"] = True

                        st.markdown(get_download_link(results_df, "bitcoin_prediction_results.csv", "Download prediction results as CSV"), unsafe_allow_html=True)
                    else:
                        st.info("Click 'Start Model Training' to begin the modeling process with the current settings.")

                with tabs_predictive[3]: # Predictions
                    st.markdown("<h2 class='sub-header-predictive'>Next Day Price Prediction</h2>", unsafe_allow_html=True)
                    if st.session_state.get("pred_models_ready", False):
                        pred_info = predict_next_day_predictive(
                            st.session_state["pred_trained_models"],
                            st.session_state["pred_meta_model"],
                            st.session_state["pred_data_featured"],
                            st.session_state["fitted_scaler_object"], # Use the new key for the FITTED scaler
                            st.session_state["pred_feature_names"]
                        )
                        p_cols = st.columns(3)
                        p_cols[0].metric("Current Price (from data)", f"${pred_info['current_price']:,.2f}")
                        p_cols[1].metric("Predicted Next Day", f"${pred_info['predicted_price']:,.2f}", delta=f"{pred_info['change_percent']:.2f}%")

                        direction_html = ""
                        if pred_info['direction'] == "UP":
                            direction_html = f"<p class='prediction-up-predictive'>⬆️ BULLISH</p>"
                        elif pred_info['direction'] == "DOWN":
                            direction_html = f"<p class='prediction-down-predictive'>⬇️ BEARISH</p>"
                        else:
                            direction_html = f"<p>N/A</p>"
                        p_cols[2].markdown(f"<div class='custom-metric-container-predictive'><h5>Prediction</h5>{direction_html}</div>", unsafe_allow_html=True)

                        st.markdown("<h4 class='sub-header-predictive' style='font-size:1.2rem; margin-top:1.5rem;'>Base Model Contributions to Next Day Prediction</h4>", unsafe_allow_html=True)
                        base_preds_df = pd.DataFrame({
                            "Model": list(pred_info["base_predictions"].keys()),
                            "Predicted Price": list(pred_info["base_predictions"].values())
                        })
                        base_preds_df["Difference"] = base_preds_df["Predicted Price"] - pred_info["current_price"]
                        base_preds_df["% Change"] = (base_preds_df["Difference"] / (pred_info["current_price"] + 1e-9)) * 100
                        st.dataframe(base_preds_df.sort_values("Predicted Price", ascending=False).style.format({
                            "Predicted Price": "${:.2f}", "Difference": "${:.2f}", "% Change": "{:.2f}%"
                        }))
                    else:
                        st.info("Train models in the 'Model Training & Performance' tab to see predictions.")
        elif not use_sample_data and uploaded_file is None:
             with tabs_predictive[1]: st.info("Please upload a dataset or select 'Use sample data' in the sidebar to begin.")
             with tabs_predictive[2]: st.info("Data needed for model training.")
             with tabs_predictive[3]: st.info("Data needed for predictions.")


    # --- Live Market Charts Mode ---
    elif app_mode == "Live Market Charts":
        st.header("₿itcoin Live Market Charts")
        st.sidebar.markdown("### Chart Settings")
        timeframe_live = st.sidebar.selectbox(
            "Select Timeframe",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd", "max"], index=2, # Default 1mo
            key="live_timeframe"
        )
        chart_type_live = st.sidebar.selectbox(
            "Chart Type", ["Candlestick", "Line", "Area"], key="live_chart_type"
        )

        with st.spinner(f"Loading Bitcoin price data for {timeframe_live}..."):
            data_live = fetch_bitcoin_data_live(timeframe_live)
            if data_live.empty:
                st.error("Could not fetch live Bitcoin data. Please try again later.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Period High", f"${data_live['High'].max():,.2f}")
                m2.metric("Period Low", f"${data_live['Low'].min():,.2f}")
                m3.metric("Avg. Volume", f"{data_live['Volume'].mean():,.0f}")

                if chart_type_live == "Candlestick":
                    fig_live = go.Figure(data=[go.Candlestick(
                        x=data_live.index, open=data_live["Open"], high=data_live["High"],
                        low=data_live["Low"], close=data_live["Close"],
                        increasing_line_color="#28a745", decreasing_line_color="#dc3545",
                    )])
                elif chart_type_live == "Line":
                    fig_live = px.line(data_live, x=data_live.index, y="Close")
                    fig_live.update_traces(line_color="#f7931a", line_width=2)
                else:  # Area
                    fig_live = px.area(data_live, x=data_live.index, y="Close")
                    fig_live.update_traces(line_color="#f7931a", fill="tozeroy", fillcolor="rgba(247, 147, 26, 0.3)")

                fig_live.update_layout(
                    title=f"Bitcoin Price - {timeframe_live.upper()}",
                    template="plotly_dark", height=550,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)", rangeslider_visible=True if chart_type_live == "Candlestick" else False),
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(fig_live, use_container_width=True)

                volume_fig_live = go.Figure()
                volume_fig_live.add_trace(go.Bar(
                    x=data_live.index, y=data_live["Volume"],
                    marker_color="#f7931a", marker_line_width=0, opacity=0.7
                ))
                volume_fig_live.update_layout(
                    title="Trading Volume", template="plotly_dark", height=250,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(volume_fig_live, use_container_width=True)

    # --- AI Market Analysis ---
    elif app_mode == "AI Market Analysis":
        st.header("🤖 AI Bitcoin Market Analysis")
        if bitcoin_intel_agent:
            if st.button("Get Latest AI Analysis", key="ai_analysis_btn"):
                with st.spinner("AI is analyzing Bitcoin markets... This may take a moment."):
                    prompt = """
                    Provide a detailed analysis of Bitcoin's (BTC-USD) current market position.
                    Include:
                    1.  Current price action and recent trends (last 7 days).
                    2.  Key support and resistance levels.
                    3.  Summary of trading volume and market capitalization trends.
                    4.  Brief overview of relevant on-chain metrics if accessible (e.g., transaction volume, active addresses - conceptual if direct data unavailable).
                    5.  Current market sentiment based on news and general discourse.
                    6.  A concise overall outlook (bullish, bearish, neutral) with justification.
                    Format clearly with headings and bullet points.
                    IMPORTANT: Always add spaces between numbers and words (e.g., write '50,000 level' not '50,000level').
                    Always include spaces after commas, and between all price values, words, and currency symbols.
                    """
                    try:
                        response = bitcoin_intel_agent.run(prompt)
                        processed_response = process_agent_output(response)
                        st.markdown(processed_response.content)
                    except Exception as e:
                        st.error(f"⚠️ Error generating AI analysis: {e}")
            else:
                st.info("Click the button to generate an up-to-date AI market analysis for Bitcoin.")
        else:
            st.error("AI Market Analysis agent is not available. Check API key configuration.")


    # --- AI News Feed ---
    elif app_mode == "AI News Feed":
        st.header("📰 AI Bitcoin News Feed")
        if bitcoin_news_agent:
            if st.button("Fetch Latest Bitcoin News", key="ai_news_btn"):
                with st.spinner("Fetching the latest Bitcoin news via AI..."):
                    prompt = """
                    Find the 5 most important and recent Bitcoin news headlines from the last 24-48 hours.
                    For each story, provide:
                    - A concise summary.
                    - The source of the news (if available).
                    - The date of publication (if available).
                    Focus on news related to price movements, regulatory changes, institutional adoption, major project updates, or significant market sentiment shifts.
                    Use the keywords 'latest Bitcoin news' for your search.
                    IMPORTANT: Always add spaces between numbers and words.
                    """
                    try:
                        response = bitcoin_news_agent.run(prompt)
                        processed_response = process_agent_output(response)
                        st.markdown(processed_response.content)
                    except Exception as e:
                        st.error(f"⚠️ Error fetching AI news: {e}")
            else:
                st.info("Click the button to fetch the latest Bitcoin news curated by AI.")
        else:
            st.error("AI News Feed agent is not available. Check API key configuration.")

    # --- Bitcoin AI Chat ---
    elif app_mode == "Bitcoin AI Chat":
        st.header("💬 Chat with Bitcoin AI")
        if bitcoin_chatbot:
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = [{"role": "assistant", "content": "Hi! How can I help you with Bitcoin today?"}]

            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_question = st.chat_input("Ask anything about Bitcoin...")
            if user_question:
                st.session_state.chat_messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("AI is thinking..."):
                        enhanced_question = f"""
                        User question about Bitcoin: {user_question}
                        Please provide a helpful and informative answer.
                        IMPORTANT: Always add spaces between numbers and words.
                        """
                        try:
                            response = bitcoin_chatbot.run(enhanced_question)
                            processed_response = process_agent_output(response)
                            ai_answer = processed_response.content
                        except Exception as e:
                            ai_answer = f"Sorry, I encountered an error: {e}"

                        st.markdown(ai_answer)
                        st.session_state.chat_messages.append({"role": "assistant", "content": ai_answer})
        else:
            st.error("Bitcoin AI Chatbot is not available. Check API key configuration.")

    # --- Footer ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #888888; font-size: 0.9em;">
        <p>Bitcoin Intelligence Hub<br>Powered by Streamlit & AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    run_hub()
