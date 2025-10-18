import streamlit as st
import pandas as pd
import json
import threading 
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ta  # Technical Analysis library
import base64 # For encoding HTML for download

# Supabase imports
from supabase import create_client, Client
from kiteconnect import KiteConnect # Moved import to top for consistency

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Kite Connect - Advanced Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect")
st.markdown("A comprehensive platform for fetching market data, performing ML-driven analysis, risk assessment, and live data streaming.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252 # Commonly accepted general figure for equity markets. Can be adjusted for specific regions.
DEFAULT_EXCHANGE = "NSE"

# Initialize session state variables if they don't exist
if "kite_access_token" not in st.session_state:
    st.session_state["kite_access_token"] = None
if "kite_login_response" not in st.session_state:
    st.session_state["kite_login_response"] = None
if "instruments_df" not in st.session_state:
    st.session_state["instruments_df"] = pd.DataFrame()
if "historical_data" not in st.session_state:
    st.session_state["historical_data"] = pd.DataFrame()
if "last_fetched_symbol" not in st.session_state:
    st.session_state["last_fetched_symbol"] = None
if "user_session" not in st.session_state:
    st.session_state["user_session"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "saved_indexes" not in st.session_state:
    st.session_state["saved_indexes"] = []
if "current_calculated_index_data" not in st.session_state: # To store current CSV's index data
    st.session_state["current_calculated_index_data"] = pd.DataFrame() # Initialize as empty DataFrame
if "current_calculated_index_history" not in st.session_state: # To store historical index values for plotting
    st.session_state["current_calculated_index_history"] = pd.DataFrame() # Initialize as empty DataFrame
if "last_comparison_df" not in st.session_state:
    st.session_state["last_comparison_df"] = pd.DataFrame()
if "last_comparison_metrics" not in st.session_state: # To store metrics for factsheet
    st.session_state["last_comparison_metrics"] = {}
if "last_facts_data" not in st.session_state: # To store data for factsheet download
    st.session_state["last_facts_data"] = None
if "last_factsheet_html_data" not in st.session_state: # To store HTML for factsheet download
    st.session_state["last_factsheet_html_data"] = None
if "current_market_data" not in st.session_state: # For market data snapshot
    st.session_state["current_market_data"] = None
if "holdings_data" not in st.session_state:
    st.session_state["holdings_data"] = None
if "historical_data_NIFTY" not in st.session_state:
    st.session_state["historical_data_NIFTY"] = pd.DataFrame()
# Modified session state for factsheet constituent selection
if "factsheet_selected_constituents_index_names" not in st.session_state:
    st.session_state["factsheet_selected_constituents_index_names"] = []


# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    supabase_conf = secrets.get("supabase", {})

    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")
    if not supabase_conf.get("url") or not supabase_conf.get("anon_key"):
        errors.append("Supabase credentials (url, anon_key)")

    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
        st.info("Example `secrets.toml`:\n```toml\n[kite]\napi_key=\"YOUR_KITE_API_KEY\"\napi_secret=\"YOUR_KITE_SECRET\"\nredirect_uri=\"http://localhost:8501\"\n\n[supabase]\nurl=\"YOUR_SUPABASE_URL\"\nanon_key=\"YOUR_SUPABASE_ANON_KEY\"\n```")
        st.stop()
    return kite_conf, supabase_conf

KITE_CREDENTIALS, SUPABASE_CREDENTIALS = load_secrets()

# --- Supabase Client Initialization ---
@st.cache_resource(ttl=3600) # Cache for 1 hour to prevent re-initializing on every rerun
def init_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)

supabase: Client = init_supabase_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["anon_key"])

# --- KiteConnect Client Initialization (Unauthenticated for login URL) ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
login_url = kite_unauth_client.login_url()


# --- Utility Functions ---

# Helper to create an authenticated KiteConnect instance
def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None

# Moved _refresh_supabase_session here, outside any `with` block
def _refresh_supabase_session():
    try:
        session_data = supabase.auth.get_session()
        if session_data and session_data.user:
            st.session_state["user_session"] = session_data
            st.session_state["user_id"] = session_data.user.id
        else:
            st.session_state["user_session"] = None
            st.session_state["user_id"] = None
    except Exception: # Catch any error during session fetching
        st.session_state["user_session"] = None
        st.session_state["user_id"] = None

@st.cache_data(ttl=86400, show_spinner="Loading instruments...") # Cache for 24 hours
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    """Returns pandas.DataFrame of instrument data, using an internally created Kite instance."""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        # Return an empty DataFrame with an error indicator
        return pd.DataFrame({"_error": ["Kite not authenticated to load instruments."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        if 'tradingsymbol' in df.columns and 'name' in df.columns:
            df = df[['instrument_token', 'tradingsymbol', 'name', 'exchange']]
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [f"Failed to load instruments for {exchange or 'all exchanges'}: {e}"]})


@st.cache_data(ttl=60) # Cache LTP for 1 minute
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    """Fetches LTP for a symbol, using an internally created Kite instance."""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return {"_error": "Kite not authenticated to fetch LTP."}
    
    exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
    try:
        ltp_data = kite_instance.ltp([exchange_symbol])
        return ltp_data.get(exchange_symbol)
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=3600) # Cache historical data for 1 hour
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date: datetime.date, to_date: datetime.date, interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    """Fetches historical data for a symbol, using an internally created Kite instance."""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated to fetch historical data."]})

    # Load instruments for token lookup (this calls the *cached* load_instruments_cached)
    instruments_df = load_instruments_cached(api_key, access_token, exchange)
    if "_error" in instruments_df.columns:
        return pd.DataFrame({"_error": [instruments_df.loc[0, '_error']]}) # Access the error message correctly

    token = find_instrument_token(instruments_df, symbol, exchange)
    if not token:
        return pd.DataFrame({"_error": [f"Instrument token not found for {symbol} on {exchange}."]})

    from_datetime = datetime.combine(from_date, datetime.min.time())
    to_datetime = datetime.combine(to_date, datetime.max.time())
    try:
        data = kite_instance.historical_data(token, from_date=from_datetime, to_date=to_datetime, interval=interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=['close'], inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [str(e)]})


def find_instrument_token(df: pd.DataFrame, tradingsymbol: str, exchange: str = DEFAULT_EXCHANGE) -> int | None:
    if df.empty:
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None


# No caching for this as it modifies df and generates many dynamic columns
def add_technical_indicators(df: pd.DataFrame, sma_short=10, sma_long=50, rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9, bb_window=20, bb_std_dev=2) -> pd.DataFrame:
    if df.empty or 'close' not in df.columns:
        st.warning("Insufficient data or missing 'close' column for indicator calculation.")
        return pd.DataFrame()

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    df_copy['SMA_Short'] = ta.trend.sma_indicator(df_copy['close'], window=sma_short)
    df_copy['SMA_Long'] = ta.trend.sma_indicator(df_copy['close'], window=sma_long)
    df_copy['RSI'] = ta.momentum.rsi(df_copy['close'], window=rsi_window)
    
    macd_obj = ta.trend.MACD(df_copy['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df_copy['MACD'] = macd_obj.macd()
    df_copy['MACD_signal'] = macd_obj.macd_signal()
    df_copy['MACD_hist'] = macd_obj.macd_diff() 
    
    bollinger = ta.volatility.BollingerBands(df_copy['close'], window=bb_window, window_dev=bb_std_dev)
    df_copy['Bollinger_High'] = bollinger.bollinger_hband()
    df_copy['Bollinger_Low'] = bollinger.bollinger_lband()
    df_copy['Bollinger_Mid'] = bollinger.bollinger_mavg()
    df_copy['Bollinger_Width'] = bollinger.bollinger_wband()
    
    df_copy['Daily_Return'] = df_copy['close'].pct_change() * 100
    df_copy['Lag_1_Close'] = df_copy['close'].shift(1)
    
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    return df_copy

def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    if returns_series.empty or len(returns_series) < 2:
        return {}
    
    # Ensure returns are not already in percentage form for cumulative calculation
    # Using 0.1 as a threshold, if mean absolute return is > 0.1 (e.g., 10%), it's likely percentage.
    # Convert to decimal for calculations.
    daily_returns_decimal = returns_series / 100.0 if returns_series.abs().mean() > 0.1 else returns_series

    # Ensure no NaN or infinite values in returns before cumulative product
    daily_returns_decimal = daily_returns_decimal.replace([np.inf, -np.inf], np.nan).dropna()
    if daily_returns_decimal.empty:
        return {} # Not enough valid data

    cumulative_returns = (1 + daily_returns_decimal).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100 if not cumulative_returns.empty else 0

    num_periods = len(daily_returns_decimal)
    
    # Annualized Return (Geometric Mean for stability)
    # Expm1 and Log1p for numerical stability with small returns
    if num_periods > 0 and (1 + daily_returns_decimal > 0).all(): # Ensure values are > -1 to prevent log(negative)
        # Calculate geometric mean daily return
        geometric_mean_daily_return = np.expm1(np.log1p(daily_returns_decimal).mean())
        # Annualize the geometric mean daily return
        annualized_return = ((1 + geometric_mean_daily_return) ** TRADING_DAYS_PER_YEAR - 1) * 100
    else:
        annualized_return = 0

    daily_volatility = daily_returns_decimal.std()
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR) * 100 if daily_volatility is not None else 0

    # Ensure risk_free_rate is decimal for calculation (e.g., 5% = 0.05)
    risk_free_rate_decimal = risk_free_rate / 100.0
    sharpe_ratio = (annualized_return / 100 - risk_free_rate_decimal) / (annualized_volatility / 100) if annualized_volatility != 0 else np.nan

    # Max Drawdown
    if not cumulative_returns.empty:
        peak = cumulative_returns.expanding(min_periods=1).max()
        # Handle division by zero if peak is 0 (occurs if all values are 0 or negative relative to start)
        # In such cases, drawdown is 0 if no gains were made from which to draw down.
        drawdown_values = []
        for i in range(len(cumulative_returns)):
            if peak.iloc[i] > 0:
                drawdown_values.append((cumulative_returns.iloc[i] - peak.iloc[i]) / peak.iloc[i])
            else: # If peak is 0 or negative, assume no meaningful drawdown from positive value.
                drawdown_values.append(0) # Or np.nan, depending on desired strictness

        drawdown = pd.Series(drawdown_values, index=cumulative_returns.index)
        max_drawdown = drawdown.min() * 100 
    else:
        max_drawdown = 0

    # Sortino Ratio
    # Calculate downside deviation relative to risk-free rate
    # Only consider returns below the risk-free rate for downside deviation
    downside_returns = daily_returns_decimal[daily_returns_decimal < risk_free_rate_decimal]
    
    downside_std_dev_daily = downside_returns.std()
    annualized_downside_std_dev = downside_std_dev_daily * np.sqrt(TRADING_DAYS_PER_YEAR) if not np.isnan(downside_std_dev_daily) else np.nan

    sortino_ratio = (annualized_return / 100 - risk_free_rate_decimal) / (annualized_downside_std_dev) if annualized_downside_std_dev != 0 and not np.isnan(annualized_downside_std_dev) else np.nan

    # Round results for display and consistency
    return {
        "Total Return (%)": round(total_return, 4),
        "Annualized Return (%)": round(annualized_return, 4),
        "Annualized Volatility (%)": round(annualized_volatility, 4),
        "Sharpe Ratio": round(sharpe_ratio, 4),
        "Max Drawdown (%)": round(max_drawdown, 4),
        "Sortino Ratio": round(sortino_ratio, 4)
    }

@st.cache_data(ttl=3600, show_spinner="Calculating historical index values...")
def _calculate_historical_index_value(api_key: str, access_token: str, constituents_df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    """
    Calculates the historical value of a custom index based on its constituents and weights.
    Returns a DataFrame with 'date' and 'index_value'.
    """
    if constituents_df.empty:
        return pd.DataFrame({"_error": ["No constituents provided for historical index calculation."]})

    all_historical_closes = {}
    
    # Use a single progress bar for all fetches
    progress_bar_placeholder = st.empty()
    progress_text_placeholder = st.empty()
    
    # Ensure instruments are loaded for symbol to token lookup
    if st.session_state["instruments_df"].empty:
        st.session_state["instruments_df"] = load_instruments_cached(api_key, access_token, exchange)
        if "_error" in st.session_state["instruments_df"].columns:
            return pd.DataFrame({"_error": [st.session_state["instruments_df"].loc[0, '_error']]})

    for i, row in constituents_df.iterrows():
        symbol = row['symbol']
        progress_text_placeholder.text(f"Fetching historical data for {symbol} ({i+1}/{len(constituents_df)})...")
        
        # Use the cached historical data function
        hist_df = get_historical_data_cached(api_key, access_token, symbol, start_date, end_date, "day", exchange)
        
        if isinstance(hist_df, pd.DataFrame) and "_error" not in hist_df.columns and not hist_df.empty:
            all_historical_closes[symbol] = hist_df['close']
        else:
            error_msg = hist_df.get('_error', ['Unknown error'])[0] if isinstance(hist_df, pd.DataFrame) else 'Unknown error'
            st.warning(f"Could not fetch historical data for {symbol}. Skipping for historical calculation. Error: {error_msg}")
        progress_bar_placeholder.progress((i + 1) / len(constituents_df))

    progress_text_placeholder.empty()
    progress_bar_placeholder.empty()

    if not all_historical_closes:
        return pd.DataFrame({"_error": ["No historical data available for any constituent to build index."]})

    combined_closes = pd.DataFrame(all_historical_closes)
    
    # Forward-fill and then back-fill any missing daily prices to be more robust
    combined_closes = combined_closes.ffill().bfill()
    combined_closes.dropna(how='all', inplace=True) # Drop rows where all are still NaN

    if combined_closes.empty:
        return pd.DataFrame({"_error": ["Insufficient common historical data for index calculation after cleaning."]})

    # Calculate daily weighted prices
    weights_series = constituents_df.set_index('symbol')['Weights']
    
    # Align columns and weights
    common_symbols = weights_series.index.intersection(combined_closes.columns)
    if common_symbols.empty:
        return pd.DataFrame({"_error": ["No common symbols between historical data and constituent weights."]})

    aligned_combined_closes = combined_closes[common_symbols]
    aligned_weights = weights_series[common_symbols]

    # Apply weights
    weighted_closes = aligned_combined_closes.mul(aligned_weights, axis=1)

    # Sum the weighted prices for each day to get the index value
    index_history_series = weighted_closes.sum(axis=1)

    # Normalize the index to a base value (e.g., 100 on the first common date)
    if not index_history_series.empty:
        first_valid_index = index_history_series.first_valid_index()
        if first_valid_index is not None:
            base_value = index_history_series[first_valid_index]
            if base_value != 0:
                index_history_df = pd.DataFrame({
                    "index_value": (index_history_series / base_value) * 100
                })
                index_history_df.index.name = 'date' # Ensure index name for later merging/plotting
                return index_history_df.dropna() # Drop any remaining NaNs
            else:
                return pd.DataFrame({"_error": ["First day's index value is zero, cannot normalize."]})
    return pd.DataFrame({"_error": ["Error in calculating or normalizing historical index values."]})

# Function to generate factsheet as multi-section CSV (includes historical data)
def generate_factsheet_csv_content(
    factsheet_constituents_df_final: pd.DataFrame, # Changed name for clarity
    factsheet_history_df_final: pd.DataFrame,     # Changed name for clarity
    last_comparison_df: pd.DataFrame,
    last_comparison_metrics: dict,
    current_live_value: float,
    index_name: str = "Custom Index",
    ai_agent_embed_snippet: str = None # Added for consistency, though won't be in CSV
) -> str:
    """Generates a comprehensive factsheet as a multi-section CSV string, including historical data."""
    content = []
    
    # --- Factsheet Header ---
    content.append(f"Factsheet for {index_name}\n")
    content.append(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    content.append("\n--- Index Overview ---\n")
    if current_live_value > 0 and not factsheet_constituents_df_final.empty:
        content.append(f"Current Live Calculated Index Value,₹{current_live_value:,.2f}\n")
    else:
        content.append("Current Live Calculated Index Value,N/A (Constituent data not available or comparison report only)\n")
    
    # --- Constituents ---
    content.append("\n--- Constituents ---\n")
    if not factsheet_constituents_df_final.empty:
        const_export_df = factsheet_constituents_df_final.copy()
        # Ensure these columns exist before formatting
        if 'Last Price' not in const_export_df.columns:
            const_export_df['Last Price'] = np.nan
        if 'Weighted Price' not in const_export_df.columns:
            const_export_df['Weighted Price'] = np.nan
        
        const_export_df['Last Price'] = const_export_df['Last Price'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        const_export_df['Weighted Price'] = const_export_df['Weighted Price'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        
        content.append(const_export_df[['symbol', 'Name', 'Weights', 'Last Price', 'Weighted Price']].to_csv(index=False))
    else:
        content.append("No constituent data available.\n")

    # --- Historical Performance (CSV ONLY) ---
    content.append("\n--- Historical Performance (Normalized to 100) ---\n")
    if not factsheet_history_df_final.empty:
        content.append(factsheet_history_df_final.to_csv())
    else:
        content.append("No historical performance data available.\n")

    # --- Performance Metrics ---
    content.append("\n--- Performance Metrics ---\n")
    if last_comparison_metrics:
        metrics_df = pd.DataFrame(last_comparison_metrics).T
        metrics_df = metrics_df.applymap(lambda x: f"{x:.4f}" if pd.notna(x) and isinstance(x, (int, float)) else x) # Format to 4 decimal places for precision
        content.append(metrics_df.to_csv())
    else:
        content.append("No performance metrics available (run a comparison first).\n")

    # --- Comparison Data ---
    content.append("\n--- Comparison Data (Normalized to 100) ---\n")
    if not last_comparison_df.empty:
        content.append(last_comparison_df.to_csv())
    else:
        content.append("No comparison data available.\n")

    return "".join(content)

# Function to generate factsheet as HTML (without historical time series)
def generate_factsheet_html_content(
    factsheet_constituents_df_final: pd.DataFrame, # Changed name for clarity
    factsheet_history_df_final: pd.DataFrame,     # Changed name for clarity
    last_comparison_df: pd.DataFrame,
    last_comparison_metrics: dict,
    current_live_value: float,
    index_name: str = "Custom Index",
    ai_agent_embed_snippet: str = None # New argument for AI agent snippet
) -> str:
    """Generates a comprehensive factsheet as an HTML string, including visualizations but NOT raw historical data."""
    html_content_parts = []

    # Basic HTML structure and styling
    html_content_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Invsion Connect Factsheet</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #1a1a1a; color: #e0e0e0; }
            .container { max-width: 900px; margin: auto; padding: 20px; background-color: #2b2b2b; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
            h1, h2, h3, h4 { color: #f0f0f0; border-bottom: 2px solid #444; padding-bottom: 5px; margin-top: 20px; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th, td { border: 1px solid #444; padding: 8px; text-align: left; }
            th { background-color: #3a3a3a; }
            .metric { font-size: 1.1em; margin-bottom: 5px; }
            .plotly-graph { margin-top: 20px; border: 1px solid #444; border-radius: 5px; overflow: hidden; }
            .info-box { background-color: #334455; border-left: 5px solid #6699cc; padding: 10px; margin-top: 10px; border-radius: 4px; }
            .warning-box { background-color: #554433; border-left: 5px solid #cc9966; padding: 10px; margin-top: 10px; border-radius: 4px; }
            .ai-agent-section { margin-top: 30px; padding: 15px; background-color: #333344; border-radius: 8px; }
            .ai-agent-section h3 { color: #add8e6; border-bottom: 1px solid #555; padding-bottom: 5px; }
            @media print {
                body { background-color: #fff; color: #000; }
                .container { box-shadow: none; border: 1px solid #eee; background-color: #fff; }
                h1, h2, h3, h4 { color: #000; border-bottom-color: #ccc; }
                th, td { border-color: #ccc; }
                .plotly-graph { border: none; }
                .ai-agent-section { display: none; /* Hide AI agent in print view if not desired */ }
            }
        </style>
    </head>
    <body>
        <div class="container">
    """)

    # --- Factsheet Header ---
    html_content_parts.append(f"<h1>Invsion Connect Factsheet: {index_name}</h1>")
    html_content_parts.append(f"<p><strong>Generated On:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html_content_parts.append("<h2>Index Overview</h2>")
    
    # FIX: Only display live value if constituent data is present AND value is calculated.
    if current_live_value > 0 and not factsheet_constituents_df_final.empty:
        html_content_parts.append(f"<p class='metric'><strong>Current Live Calculated Index Value:</strong> ₹{current_live_value:,.2f}</p>")
    else:
        # This is the line that generates the message shown in the image
        html_content_parts.append("<p class='warning-box'>Current Live Calculated Index Value: N/A (Constituent data not available or comparison report only)</p>")

    # --- Constituents ---
    html_content_parts.append("<h3>Constituents</h3>")
    if not factsheet_constituents_df_final.empty:
        const_display_df = factsheet_constituents_df_final.copy()
        
        if 'Name' not in const_display_df.columns:
            const_display_df['Name'] = const_display_df['symbol'] 

        # Ensure 'Last Price' and 'Weighted Price' exist before formatting
        if 'Last Price' in const_display_df.columns:
            const_display_df['Last Price'] = const_display_df['Last Price'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        else:
            const_display_df['Last Price'] = "N/A" # Default if column is missing
        
        if 'Weighted Price' in const_display_df.columns:
            const_display_df['Weighted Price'] = const_display_df['Weighted Price'].apply(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
        else:
            const_display_df['Weighted Price'] = "N/A" # Default if column is missing

        html_content_parts.append(const_display_df[['symbol', 'Name', 'Weights', 'Last Price', 'Weighted Price']].to_html(index=False, classes='table'))

        # Index Composition Pie Chart
        fig_pie = go.Figure(data=[go.Pie(labels=const_display_df['Name'], values=const_display_df['Weights'], hole=.3)])
        fig_pie.update_layout(title_text='Constituent Weights', height=400, template="plotly_dark")
        html_content_parts.append("<h3>Index Composition</h3>")
        # include_plotlyjs='cdn' ensures Plotly.js is loaded for this chart
        html_content_parts.append(f"<div class='plotly-graph'>{fig_pie.to_html(full_html=False, include_plotlyjs='cdn')}</div>") 
    else:
        html_content_parts.append("<p class='warning-box'>No constituent data available for this index.</p>")
    
    # --- Performance Metrics ---
    html_content_parts.append("<h3>Performance Metrics Summary</h3>")
    if last_comparison_metrics:
        metrics_df = pd.DataFrame(last_comparison_metrics).T
        metrics_html = metrics_df.style.format("{:.4f}").to_html(classes='table') # Format to 4 decimal places for precision
        html_content_parts.append(metrics_html)
    else:
        html_content_parts.append("<p class='warning-box'>No performance metrics available (run a comparison first).</p>")

    # --- Comparison Data (Chart Only) ---
    html_content_parts.append("<h3>Cumulative Performance Comparison (Normalized to 100)</h3>")
    if not last_comparison_df.empty:
        fig_comparison = go.Figure()
        for col in last_comparison_df.columns:
            fig_comparison.add_trace(go.Scatter(x=last_comparison_df.index, y=last_comparison_df[col], mode='lines', name=col))
        
        # Adjust title based on whether it's a specific index report or a generic comparison
        chart_title = "Multi-Index & Benchmark Performance"
        if index_name != "Consolidated Report" and index_name != "Comparison Report" and index_name != "Combined Index Constituents Report": # Added "Combined Index Constituents Report"
            chart_title = f"{index_name} vs Benchmarks Performance"

        fig_comparison.update_layout(
            title_text=chart_title,
            xaxis_title="Date",
            yaxis_title="Normalized Value (Base 100)",
            height=600,
            template="plotly_dark",
            hovermode="x unified"
        )
        # include_plotlyjs='cdn' ensures Plotly.js is loaded for this chart
        html_content_parts.append(f"<div class='plotly-graph'>{fig_comparison.to_html(full_html=False, include_plotlyjs='cdn')}</div>") 
    else:
        html_content_parts.append("<p class='warning-box'>No comparison data available.</p>")

    # --- Optional: Historical Performance Chart for the main index (if applicable and not too large) ---
    # We will only show this if a SINGLE index is selected for constituents.
    # If multiple are chosen for constituents, it becomes ambiguous.
    # Updated condition to check for a single selected index in factsheet_selected_constituents_index_names
    if (len(st.session_state["factsheet_selected_constituents_index_names"]) == 1 and 
        not factsheet_history_df_final.empty and 
        factsheet_history_df_final.shape[0] < 730): 
        html_content_parts.append("<h3>Index Historical Performance (Normalized to 100)</h3>")
        fig_hist_index = go.Figure(data=[go.Scatter(x=factsheet_history_df_final.index, y=factsheet_history_df_final['index_value'], mode='lines', name=index_name)])
        fig_hist_index.update_layout(title_text=f"{index_name} Historical Performance", template="plotly_dark", height=400)
        # include_plotlyjs='cdn' ensures Plotly.js is loaded for this chart
        html_content_parts.append(f"<div class='plotly-graph'>{fig_hist_index.to_html(full_html=False, include_plotlyjs='cdn')}</div>")
    elif not factsheet_history_df_final.empty and len(st.session_state["factsheet_selected_constituents_index_names"]) == 1:
        html_content_parts.append(f"<p class='info-box'>Historical performance chart for {index_name} is too large for the HTML factsheet. Please refer to the CSV download.</p>")
    elif len(st.session_state["factsheet_selected_constituents_index_names"]) > 1:
         html_content_parts.append(f"<p class='info-box'>Historical performance chart for individual index constituents is not shown when multiple indexes are selected for the constituents section. Please refer to the CSV download for full historical data or the comparison chart above.</p>")


    # --- AI Agent Embed Snippet ---
    if ai_agent_embed_snippet:
        html_content_parts.append("""
            <div class="ai-agent-section">
                <h3>Embedded AI Agent</h3>
        """)
        html_content_parts.append(ai_agent_embed_snippet) # FIX: Changed from 'ai_agent_snippet' to 'ai_agent_embed_snippet'
        html_content_parts.append("</div>")

    html_content_parts.append("""
        <div class="info-box">
            <p><strong>Note:</strong> Raw historical time series data (tables) is intentionally excluded from this HTML/PDF factsheet to keep it concise and visually focused. For the full historical data, please download the CSV factsheet.</p>
            <p>To convert this HTML file to PDF, open it in your web browser (e.g., Chrome, Firefox) and use the browser's "Print" function. Then select "Save as PDF" from the printer options.</p>
        </div>
        </div>
    </body>
    </html>
    """)
    return "".join(html_content_parts)


# --- Sidebar: Kite Login ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    st.write("Click to open Kite login. You'll be redirected back with a `request_token`.")
    st.markdown(f"[🔗 Open Kite login]({login_url})")

    # Handle request_token from URL
    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        st.info("Received request_token — exchanging for access token...")
        try:
            data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
            st.session_state["kite_access_token"] = data.get("access_token")
            st.session_state["kite_login_response"] = data
            st.sidebar.success("Kite Access token obtained.")
            st.query_params.clear() # Clear request_token from URL
            st.rerun() # Rerun to refresh UI
        except Exception as e:
            st.sidebar.error(f"Failed to generate Kite session: {e}")

    if st.session_state["kite_access_token"]:
        # Only render the logout button if authenticated
        # Using a more unique key based on session state to avoid conflicts on reruns
        if st.sidebar.button("Logout from Kite", key=f"kite_logout_btn_{st.session_state['kite_access_token'][:5]}"):
            st.session_state["kite_access_token"] = None
            st.session_state["kite_login_response"] = None
            st.session_state["instruments_df"] = pd.DataFrame() # Clear cached instruments
            st.success("Logged out from Kite. Please login again.")
            st.rerun()
        st.success("Kite Authenticated ✅")
    else:
        st.info("Not authenticated with Kite yet.")


# --- Sidebar: Supabase Authentication ---
with st.sidebar:
    st.markdown("### 2. Supabase User Account")
    
    # Call the globally defined _refresh_supabase_session
    _refresh_supabase_session()

    if st.session_state["user_session"]:
        st.success(f"Logged into Supabase as: {st.session_state['user_session'].user.email}")
        # Only render the logout button if authenticated
        if st.button("Logout from Supabase", key=f"supabase_logout_btn_{st.session_state['user_id']}"): # Dynamic key
            try:
                supabase.auth.sign_out()
                _refresh_supabase_session() # Update session state immediately
                st.sidebar.success("Logged out from Supabase.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error logging out: {e}")
    else:
        # The form key here needs to be static, as this block is entered only when not logged in.
        # The previous error likely came from a rerun causing this block to be processed again
        # while the form was somehow still registered, perhaps due to multiple 'with' blocks or
        # an unexpected interaction with rerun. Using a static key in this 'else' branch is standard.
        with st.form("supabase_auth_form_logged_out_static_key"):
            st.markdown("##### Email/Password Login/Sign Up")
            email = st.text_input("Email", key="supabase_email_input", help="Your email for Supabase authentication.")
            password = st.text_input("Password", type="password", key="supabase_password_input", help="Your password for Supabase authentication.")
            
            col_auth1, col_auth2 = st.columns(2)
            with col_auth1:
                login_submitted = st.form_submit_button("Login")
            with col_auth2:
                signup_submitted = st.form_submit_button("Sign Up")

            if login_submitted:
                if email and password:
                    try:
                        with st.spinner("Logging in..."):
                            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        _refresh_supabase_session()
                        st.success("Login successful! Welcome.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")
                else:
                    st.warning("Please enter both email and password for login.")
            
            if signup_submitted:
                if email and password:
                    try:
                        with st.spinner("Signing up..."):
                            response = supabase.auth.sign_up({"email": email, "password": password})
                        _refresh_supabase_session()
                        st.success("Sign up successful! Please check your email to confirm your account.")
                        st.info("After confirming your email, you can log in.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")
                else:
                    st.warning("Please enter both email and password for sign up.")

    st.markdown("---")
    st.markdown("### 3. Quick Data Access (Kite)")
    if st.session_state["kite_access_token"]:
        current_k_client_for_sidebar = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

        if st.button("Fetch Current Holdings", key="sidebar_fetch_holdings_btn"):
            try:
                holdings = current_k_client_for_sidebar.holdings() # Direct call
                st.session_state["holdings_data"] = pd.DataFrame(holdings)
                st.success(f"Fetched {len(holdings)} holdings.")
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")
        if st.session_state.get("holdings_data") is not None and not st.session_state["holdings_data"].empty:
            with st.expander("Show Holdings"):
                st.dataframe(st.session_state["holdings_data"])
    else:
        st.info("Login to Kite to access quick data.")


# --- Authenticated KiteConnect client (used by main tabs) ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])


# --- Main UI - Tabs for modules ---
tabs = st.tabs(["Market & Historical", "Custom Index"]) # Removed "Dashboard"
tab_market, tab_custom_index = tabs # Updated assignment

# --- Tab Logic Functions ---
# render_dashboard_tab function removed as per request.

def render_market_historical_tab(kite_client: KiteConnect | None, api_key: str | None, access_token: str | None):
    st.header("Market Data & Historical Candles")
    if not kite_client:
        st.info("Login first to fetch market data.")
        return
    if not api_key or not access_token: # Additional check for cached functions
        st.info("Kite authentication details required for cached data access.")
        return

    st.subheader("Current Market Data Snapshot")
    col_market_quote1, col_market_quote2 = st.columns([1, 2])
    with col_market_quote1:
        q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="market_exchange_tab")
        q_symbol = st.text_input("Tradingsymbol", value="NIFTY 50", key="market_symbol_tab") # Default to NIFTY 50 for quick demo
        if st.button("Get Market Data", key="get_market_data_btn"):
            ltp_data = get_ltp_price_cached(api_key, access_token, q_symbol, q_exchange) # Use cached LTP
            if ltp_data and "_error" not in ltp_data:
                st.session_state["current_market_data"] = ltp_data
                st.success(f"Fetched LTP for {q_symbol}.")
            else:
                st.error(f"Market data fetch failed for {q_symbol}: {ltp_data.get('_error', 'Unknown error')}")
    with col_market_quote2:
        if st.session_state.get("current_market_data"):
            st.markdown("##### Latest Quote Details")
            st.json(st.session_state["current_market_data"])
        else:
            st.info("Market data will appear here.")

    st.markdown("---")
    st.subheader("Historical Price Data")
    with st.expander("Load Instruments for Symbol Lookup (Recommended)"):
        exchange_for_lookup = st.selectbox("Exchange to load instruments", ["NSE", "BSE", "NFO"], key="hist_inst_load_exchange_selector")
        if st.button("Load Instruments into Cache", key="load_inst_cache_btn"):
            df_instruments = load_instruments_cached(api_key, access_token, exchange_for_lookup) # Use cached instruments
            if not df_instruments.empty and "_error" not in df_instruments.columns:
                st.session_state["instruments_df"] = df_instruments
                st.success(f"Loaded {len(df_instruments)} instruments.")
            else:
                st.error(f"Failed to load instruments: {df_instruments.get('_error', 'Unknown error')}")


    col_hist_controls, col_hist_plot = st.columns([1, 2])
    with col_hist_controls:
        hist_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO"], key="hist_ex_tab_selector")
        hist_symbol = st.text_input("Tradingsymbol", value="NIFTY 50", key="hist_sym_tab_input") # Default to NIFTY 50 for quick demo
        from_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=90), key="from_dt_tab_input")
        to_date = st.date_input("To Date", value=datetime.now().date(), key="to_dt_tab_input")
        interval = st.selectbox("Interval", ["minute", "5minute", "30minute", "day", "week", "month"], index=3, key="hist_interval_selector")

        if st.button("Fetch Historical Data", key="fetch_historical_data_btn"):
            with st.spinner(f"Fetching {interval} historical data for {hist_symbol}..."):
                df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, hist_exchange) # Use cached historical
                if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                    st.session_state["historical_data"] = df_hist
                    st.session_state["last_fetched_symbol"] = hist_symbol
                    st.success(f"Fetched {len(df_hist)} records for {hist_symbol}.")
                else:
                    st.error(f"Historical fetch failed: {df_hist.get('_error', 'Unknown error')}")

    with col_hist_plot:
        if not st.session_state.get("historical_data", pd.DataFrame()).empty:
            df = st.session_state["historical_data"]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candlestick'), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='blue'), row=2, col=1)
            fig.update_layout(title_text=f"Historical Price & Volume for {st.session_state['last_fetched_symbol']}", xaxis_rangeslider_visible=False, height=600, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Historical chart will appear here.")


def render_custom_index_tab(kite_client: KiteConnect | None, supabase_client: Client, api_key: str | None, access_token: str | None):
    st.header("📊 Custom Index Creation, Benchmarking & Export")
    st.markdown("Create your own weighted index, analyze its historical performance, compare it against benchmarks, and calculate key financial metrics.")
    
    if not kite_client:
        st.info("Login to Kite first to fetch live and historical prices for index constituents.")
        return
    if not st.session_state["user_id"]:
        st.info("Login with your Supabase account in the sidebar to save and load custom indexes.")
        return
    if not api_key or not access_token:
        st.info("Kite authentication details required for data access.")
        return

    # Helper function to load historical data for a given index/symbol
    @st.cache_data(ttl=3600, show_spinner="Fetching historical data for comparison...")
    def _fetch_and_normalize_data_for_comparison(
        name: str,
        data_type: str, # "custom_index" or "benchmark"
        comparison_start_date: datetime.date, # User-defined start date for comparison
        comparison_end_date: datetime.date,   # User-defined end date for comparison
        constituents_df: pd.DataFrame = None, # For custom index
        symbol: str = None, # For benchmark
        exchange: str = DEFAULT_EXCHANGE,
        api_key: str = None,
        access_token: str = None
    ) -> pd.DataFrame:
        """
        Fetches and normalized historical data for a custom index or a benchmark symbol
        within a specified comparison date range.
        Returns a DataFrame with 'date', 'normalized_value', and 'raw_values'.
        """
        hist_df = pd.DataFrame()
        if data_type == "custom_index":
            if constituents_df is None or constituents_df.empty:
                return pd.DataFrame({"_error": [f"No constituents for custom index {name}."]})
            # Always recalculate for the exact comparison range to ensure consistency
            hist_df = _calculate_historical_index_value(api_key, access_token, constituents_df, comparison_start_date, comparison_end_date, exchange)
            if "_error" in hist_df.columns:
                return hist_df
            data_series = hist_df['index_value']
        elif data_type == "benchmark":
            if symbol is None:
                return pd.DataFrame({"_error": [f"No symbol for benchmark {name}."]})
            hist_df = get_historical_data_cached(api_key, access_token, symbol, comparison_start_date, comparison_end_date, "day", exchange)
            if "_error" in hist_df.columns:
                return hist_df
            data_series = hist_df['close']
        else:
            return pd.DataFrame({"_error": ["Invalid data_type for comparison."]})

        if data_series.empty:
            return pd.DataFrame({"_error": [f"No historical data for {name} within the selected range."]})

        # Normalize to 100 on the first available date within the fetched series
        first_valid_index = data_series.first_valid_index()
        if first_valid_index is not None and data_series[first_valid_index] != 0:
            normalized_series = (data_series / data_series[first_valid_index]) * 100
            return pd.DataFrame({'normalized_value': normalized_series, 'raw_values': data_series}).rename_axis('date')
        return pd.DataFrame({"_error": [f"Could not normalize {name} (first value is zero or no valid data in range)."]})


    # Helper function to render an index's details, charts, and export options
    def display_single_index_details(index_name: str, constituents_df: pd.DataFrame, index_history_df: pd.DataFrame, index_id: str | None = None, is_recalculated_live=False):
        st.markdown(f"#### Details for Index: **{index_name}** {'(Recalculated Live)' if is_recalculated_live else ''}")
        
        st.subheader("Constituents and Current Live Value")
        
        live_quotes = {}
        symbols_for_ltp = [sym for sym in constituents_df["symbol"]]
        
        # Load instruments for symbol to token lookup if not already loaded
        if st.session_state["instruments_df"].empty:
            with st.spinner("Loading instruments for live price lookup..."):
                st.session_state["instruments_df"] = load_instruments_cached(api_key, access_token, DEFAULT_EXCHANGE)
        
        if "_error" in st.session_state["instruments_df"].columns:
            st.warning(f"Could not load instruments for live price lookup: {st.session_state['instruments_df'].loc[0, '_error']}")
        else:
            if symbols_for_ltp:
                try:
                    kc_client = get_authenticated_kite_client(api_key, access_token)
                    if kc_client:
                        instrument_identifiers = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols_for_ltp]
                        # Use uncached LTP batch call for the live price check
                        ltp_data_batch = kc_client.ltp(instrument_identifiers)
                        
                        for sym in symbols_for_ltp:
                            key = f"{DEFAULT_EXCHANGE}:{sym}"
                            if key in ltp_data_batch:
                                live_quotes[sym] = ltp_data_batch[key].get("last_price", np.nan)
                            else:
                                live_quotes[sym] = np.nan
                    else:
                        st.warning("Kite client not available for batch LTP fetch (internal error).")
                except Exception as e:
                    st.error(f"Error fetching batch LTP: {e}. Live prices might be partial.")

        # Ensure 'Name' column exists for display
        if 'Name' not in constituents_df.columns:
            if not st.session_state["instruments_df"].empty:
                instrument_names = st.session_state["instruments_df"].set_index('tradingsymbol')['name'].to_dict()
                constituents_df['Name'] = constituents_df['symbol'].map(instrument_names).fillna(constituents_df['symbol'])
            else:
                constituents_df['Name'] = constituents_df['symbol'] # Fallback if no names found

        constituents_df_display = constituents_df.copy()
        constituents_df_display["Last Price"] = constituents_df_display["symbol"].map(live_quotes)
        constituents_df_display["Weighted Price"] = constituents_df_display["Last Price"] * constituents_df_display["Weights"]
        current_live_value = constituents_df_display["Weighted Price"].sum()

        st.dataframe(constituents_df_display[['symbol', 'Name', 'Weights', 'Last Price', 'Weighted Price']].style.format({
            "Weights": "{:.4f}",
            "Last Price": "₹{:,.2f}",
            "Weighted Price": "₹{:,.2f}"
        }), use_container_width=True)
        st.success(f"Current Live Calculated Index Value: **₹{current_live_value:,.2f}**")

        st.markdown("---")
        st.subheader("Index Composition")
        fig_pie = go.Figure(data=[go.Pie(labels=constituents_df_display['Name'], values=constituents_df_display['Weights'], hole=.3)])
        fig_pie.update_layout(title_text='Constituent Weights', height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        st.subheader("Export Options")
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            csv_constituents = constituents_df_display[['symbol', 'Name', 'Weights', 'Last Price', 'Weighted Price']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Constituents to CSV",
                data=csv_constituents,
                file_name=f"{index_name}_constituents.csv",
                mime="text/csv",
                key=f"export_constituents_{index_id or index_name}"
            )
        with col_export2:
            if not index_history_df.empty:
                csv_history = index_history_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Export Historical Performance to CSV",
                    data=csv_history,
                    file_name=f"{index_name}_historical_performance.csv",
                    mime="text/csv",
                    key=f"export_history_{index_id or index_name}"
                )
            else:
                st.info("No historical data to export for this index.")

    # --- Section: Index Creation ---
    st.markdown("---")
    st.subheader("1. Create New Index from CSV")
    st.markdown("Upload a CSV file containing your desired index constituents. The CSV must have columns: `symbol`, `Name`, `Weights` (case-sensitive).")
    st.markdown("The `symbol` column contains the ticker (e.g., INFY) used to fetch quotes.")
    st.code("symbol,Name,Weights\nINFY,Infosys,0.3\nRELIANCE,Reliance Industries,0.5\nHDFCBANK,HDFC Bank,0.2")

    uploaded_file = st.file_uploader("Upload CSV with index constituents", type=["csv"], key="index_upload_csv")
    
    if uploaded_file:
        try:
            df_constituents_new = pd.read_csv(uploaded_file)
            required_cols = {"symbol", "Name", "Weights"}
            if not required_cols.issubset(set(df_constituents_new.columns)):
                st.error(f"CSV must contain columns: `symbol`, `Name`, `Weights`. Missing: {required_cols - set(df_constituents_new.columns)}")
                return

            df_constituents_new["Weights"] = pd.to_numeric(df_constituents_new["Weights"], errors='coerce')
            df_constituents_new.dropna(subset=["Weights", "symbol"], inplace=True)
            
            if df_constituents_new.empty:
                st.error("No valid constituents found in the CSV. Ensure 'symbol' and numeric 'Weights' columns are present.")
                return

            total_weights = df_constituents_new["Weights"].sum()
            if total_weights <= 0:
                st.error("Sum of weights must be positive.")
                return
            df_constituents_new["Weights"] = df_constituents_new["Weights"] / total_weights # Normalize weights
            st.info(f"Loaded {len(df_constituents_new)} constituents. Weights have been normalized to sum to 1.")

            st.subheader("Configure Historical Calculation for New Index")
            hist_start_date = st.date_input("Historical Start Date", value=datetime.now().date() - timedelta(days=365), key="new_index_hist_start_date")
            hist_end_date = st.date_input("Historical End Date", value=datetime.now().date(), key="new_index_hist_end_date")

            if st.button("Calculate and Analyze New Index", key="calculate_new_index_btn"):
                if hist_start_date >= hist_end_date:
                    st.error("Historical start date must be before end date.")
                else:
                    index_history_df_new = _calculate_historical_index_value(api_key, access_token, df_constituents_new, hist_start_date, hist_end_date, DEFAULT_EXCHANGE)
                
                    if not index_history_df_new.empty and "_error" not in index_history_df_new.columns:
                        st.session_state["current_calculated_index_data"] = df_constituents_new
                        st.session_state["current_calculated_index_history"] = index_history_df_new
                        st.success("Historical index values calculated successfully.")
                        # Set newly calculated index as the default selection for factsheet constituents
                        st.session_state["factsheet_selected_constituents_index_names"] = ["Newly Calculated Index"] 
                    else:
                        st.error(f"Failed to calculate historical index values for new index: {index_history_df_new.get('_error', ['Unknown error'])[0]}")
                        st.session_state["current_calculated_index_data"] = pd.DataFrame() # Ensure it's a DataFrame
                        st.session_state["current_calculated_index_history"] = pd.DataFrame()
                        st.session_state["factsheet_selected_constituents_index_names"] = [] # Clear selection if calculation fails
                        
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}.")

    # --- After calculation for a new index ---
    # Retrieve current_calculated_index_data and history, ensuring they are DataFrames
    current_calculated_index_data_df = st.session_state.get("current_calculated_index_data", pd.DataFrame())
    current_calculated_index_history_df = st.session_state.get("current_calculated_index_history", pd.DataFrame())

    if not current_calculated_index_data_df.empty and not current_calculated_index_history_df.empty:
        
        # Calculate current_live_value to display and for the factsheet
        constituents_df_for_live = current_calculated_index_data_df.copy()
        live_quotes = {}
        symbols_for_ltp = [sym for sym in constituents_df_for_live["symbol"]] if not constituents_df_for_live.empty else []

        if not st.session_state["instruments_df"].empty and symbols_for_ltp:
            try:
                kc_client = get_authenticated_kite_client(api_key, access_token)
                if kc_client:
                    instrument_identifiers = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols_for_ltp]
                    ltp_data_batch = kc_client.ltp(instrument_identifiers)
                    for sym in symbols_for_ltp:
                        key = f"{DEFAULT_EXCHANGE}:{sym}"
                        live_quotes[sym] = ltp_data_batch.get(key, {}).get("last_price", np.nan)
            except Exception as e:
                st.warning(f"Error fetching batch LTP for live value calculation: {e}. Live prices might be partial.")
        
        # Add Last Price and Weighted Price columns to constituents_df_for_live for consistent factsheet generation
        constituents_df_for_live["Last Price"] = constituents_df_for_live["symbol"].map(live_quotes)
        constituents_df_for_live["Weighted Price"] = constituents_df_for_live["Last Price"] * constituents_df_for_live["Weights"]
        current_live_value_for_factsheet_display = constituents_df_for_live["Weighted Price"].sum() if not constituents_df_for_live["Weighted Price"].empty else 0.0

        display_single_index_details("Newly Calculated Index", constituents_df_for_live, current_calculated_index_history_df, index_id="new_index")
        
        st.markdown("---")
        st.subheader("Save Newly Created Index")
        index_name_to_save = st.text_input("Enter a unique name for this index to save it:", value="MyCustomIndex", key="new_index_save_name")
        if st.button("Save New Index to DB", key="save_new_index_to_db_btn"):
            if index_name_to_save and st.session_state["user_id"]:
                try:
                    with st.spinner("Saving index..."):
                        check_response = supabase_client.table("custom_indexes").select("id").eq("user_id", st.session_state["user_id"]).eq("index_name", index_name_to_save).execute()
                        if check_response.data:
                            st.warning(f"An index named '{index_name_to_save}' already exists. Please choose a different name.")
                        else:
                            history_df_to_save = current_calculated_index_history_df.reset_index()
                            history_df_to_save['date'] = history_df_to_save['date'].dt.strftime('%Y-%m-%dT%H:%M:%S') 

                            index_data = {
                                "user_id": st.session_state["user_id"],
                                "index_name": index_name_to_save,
                                # Save the original constituents without live price columns
                                "constituents": current_calculated_index_data_df[['symbol', 'Name', 'Weights']].to_dict(orient='records'),
                                "historical_performance": history_df_to_save.to_dict(orient='records')
                            }
                            response = supabase_client.table("custom_indexes").insert(index_data).execute()
                            st.success(f"Index '{index_name_to_save}' saved successfully!")
                            # Clear session state to prevent re-saving and force reload
                            st.session_state["saved_indexes"] = [] 
                            st.session_state["current_calculated_index_data"] = pd.DataFrame() # Reset to empty DataFrame
                            st.session_state["current_calculated_index_history"] = pd.DataFrame() # Reset to empty DataFrame
                            st.session_state["factsheet_selected_constituents_index_names"] = [] # Reset factsheet selection
                            st.rerun()
                except Exception as e:
                    st.error(f"Error saving new index: {e}")
            else:
                st.warning("Please enter an index name and ensure you are logged into Supabase.")
    
    st.markdown("---")
    st.subheader("2. Load & Manage Saved Indexes")
    if st.button("Load My Indexes from DB", key="load_my_indexes_db_btn"):
        try:
            with st.spinner("Loading indexes..."):
                response = supabase_client.table("custom_indexes").select("id, index_name, constituents, historical_performance").eq("user_id", st.session_state["user_id"]).execute()
            if response.data:
                st.session_state["saved_indexes"] = response.data
                st.success(f"Loaded {len(response.data)} indexes.")
            else:
                st.session_state["saved_indexes"] = []
                st.info("No saved indexes found for your account.")
        except Exception as e: st.error(f"Error loading indexes: {e}")
    
    # Ensure saved_indexes is a list (can be empty)
    saved_indexes = st.session_state.get("saved_indexes", [])
    if saved_indexes:
        index_names_from_db = [idx['index_name'] for idx in saved_indexes]
        
        selected_custom_indexes_names = st.multiselect(
            "Select saved custom indexes to include in comparison:", 
            options=index_names_from_db, 
            key="select_saved_indexes_for_comparison"
        )

        st.markdown("---")
        st.subheader("3. Configure & Run Multi-Index & Benchmark Comparison")
        
        col_comp_dates, col_comp_bench = st.columns(2)
        with col_comp_dates:
            comparison_start_date = st.date_input("Comparison Start Date", value=datetime.now().date() - timedelta(days=365), key="comparison_start_date")
            comparison_end_date = st.date_input("Comparison End Date", value=datetime.now().date(), key="comparison_end_date")
            if comparison_start_date >= comparison_end_date:
                st.error("Comparison start date must be before end date.")
                # To ensure comparison_start_date and comparison_end_date are always defined
                comparison_start_date = datetime.now().date() - timedelta(days=365)
                comparison_end_date = datetime.now().date()


        with col_comp_bench:
            benchmark_symbols_str = st.text_area(
                "Enter External Benchmark Symbols (comma-separated, e.g., NIFTY 50,BANKNIFTY,NIFTY BANK)",
                value="NIFTY 50", # Default to one benchmark
                height=80,
                key="comparison_benchmark_symbols_input"
            )
            external_benchmark_symbols = [s.strip().upper() for s in benchmark_symbols_str.split(',') if s.strip()]
            comparison_exchange = st.selectbox("Exchange for External Benchmarks", ["NSE", "BSE", "NFO"], key="comparison_bench_exchange_select")
        
        if st.button("Run Multi-Index & Benchmark Comparison", key="run_multi_comparison_btn"):
            if not selected_custom_indexes_names and not external_benchmark_symbols:
                st.warning("Please select at least one custom index or enter at least one benchmark symbol for comparison.")
            else:
                all_normalized_data = {}
                all_performance_metrics = {}
                
                # Load instruments once for all lookups in this tab
                if st.session_state["instruments_df"].empty:
                    with st.spinner("Loading instruments for comparison lookup..."):
                        st.session_state["instruments_df"] = load_instruments_cached(api_key, access_token, DEFAULT_EXCHANGE)
                
                if "_error" in st.session_state["instruments_df"].columns:
                    st.error(f"Failed to load instruments for comparison lookup: {st.session_state['instruments_df'].loc[0, '_error']}")
                    return

                # Fetch data for selected custom indexes
                for index_name in selected_custom_indexes_names:
                    db_index_data = next((idx for idx in saved_indexes if idx['index_name'] == index_name), None) # Use local `saved_indexes`
                    if db_index_data:
                        constituents_df = pd.DataFrame(db_index_data['constituents'])
                        
                        normalized_df_result = _fetch_and_normalize_data_for_comparison(
                            name=index_name,
                            data_type="custom_index",
                            comparison_start_date=comparison_start_date,
                            comparison_end_date=comparison_end_date,
                            constituents_df=constituents_df,
                            api_key=api_key,
                            access_token=access_token
                        )
                        if "_error" not in normalized_df_result.columns:
                            all_normalized_data[index_name] = normalized_df_result['normalized_value']
                            # Use raw values for performance metrics calculation
                            all_performance_metrics[index_name] = calculate_performance_metrics(normalized_df_result['raw_values'].pct_change().dropna() * 100)
                        else:
                            st.error(f"Error processing custom index {index_name}: {normalized_df_result.loc[0, '_error']}")

                # Fetch data for external benchmarks
                for symbol in external_benchmark_symbols:
                    normalized_df_result = _fetch_and_normalize_data_for_comparison(
                        name=symbol,
                        data_type="benchmark",
                        comparison_start_date=comparison_start_date,
                        comparison_end_date=comparison_end_date,
                        symbol=symbol,
                        exchange=comparison_exchange,
                        api_key=api_key,
                        access_token=access_token
                    )
                    if "_error" not in normalized_df_result.columns:
                        all_normalized_data[symbol] = normalized_df_result['normalized_value']
                        # Use raw values for performance metrics calculation
                        all_performance_metrics[symbol] = calculate_performance_metrics(normalized_df_result['raw_values'].pct_change().dropna() * 100)
                    else:
                        st.error(f"Error processing benchmark {symbol}: {normalized_df_result.loc[0, '_error']}")

                if all_normalized_data:
                    # Combine all normalized series into a single DataFrame
                    combined_comparison_df = pd.DataFrame(all_normalized_data)
                    combined_comparison_df.dropna(how='all', inplace=True) # Drop rows where all are NaN
                    
                    if not combined_comparison_df.empty:
                        st.session_state["last_comparison_df"] = combined_comparison_df
                        st.session_state["last_comparison_metrics"] = all_performance_metrics
                        st.success("Comparison data generated successfully.")
                    else:
                        st.warning("No common or sufficient data found for comparison. Please check selected indexes/benchmarks and date range.")
                else:
                    st.info("No data selected or fetched for comparison.")

        # Ensure last_comparison_df is a DataFrame
        last_comparison_df = st.session_state.get("last_comparison_df", pd.DataFrame())

        if not last_comparison_df.empty:
            st.markdown("#### Cumulative Performance Comparison (Normalized to 100)")
            fig_comparison = go.Figure()
            for col in last_comparison_df.columns:
                fig_comparison.add_trace(go.Scatter(x=last_comparison_df.index, y=last_comparison_df[col], mode='lines', name=col))
            
            fig_comparison.update_layout(
                title_text="Multi-Index & Benchmark Performance",
                xaxis_title="Date",
                yaxis_title="Normalized Value (Base 100)",
                height=600,
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

            st.markdown("#### Performance Metrics Summary")
            metrics_df = pd.DataFrame(st.session_state["last_comparison_metrics"]).T
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True) # Display metrics with higher precision


        st.markdown("---")
        st.subheader("5. Generate and Download Consolidated Factsheet")
        st.info("This will generate a factsheet. If a new index is calculated or a single saved index is selected, it will create a detailed report for that index. Otherwise, it will generate a comparison-only factsheet if comparison data is available.")
        
        # --- Factsheet data preparation logic ---
        factsheet_constituents_df_final = pd.DataFrame()
        factsheet_history_df_final = pd.DataFrame() # Will only be populated if a single index is selected for constituents
        factsheet_index_name_final = "Consolidated Report" # Default for a general comparison report
        current_live_value_for_factsheet_final = 0.0
        
        # UI for selecting which index's constituents to show in the factsheet
        available_constituents_for_factsheet = ["None"] # Keep "None" as an option
        if not current_calculated_index_data_df.empty:
            available_constituents_for_factsheet.append("Newly Calculated Index")
        if saved_indexes: # Add all saved index names
            available_constituents_for_factsheet.extend(index_names_from_db)
        
        st.markdown("---")
        st.subheader("Factsheet Content Selection")
        
        # Use st.multiselect instead of st.selectbox
        selected_constituents_for_factsheet = st.multiselect(
            "Select which custom index(es) constituents and live value to include in the factsheet:",
            options=available_constituents_for_factsheet,
            default=st.session_state["factsheet_selected_constituents_index_names"], # Use previously selected as default
            key="factsheet_constituents_selector"
        )
        st.session_state["factsheet_selected_constituents_index_names"] = selected_constituents_for_factsheet

        all_constituents_dfs = []
        
        if selected_constituents_for_factsheet:
            # If "Newly Calculated Index" is selected, add its data
            if "Newly Calculated Index" in selected_constituents_for_factsheet:
                if not current_calculated_index_data_df.empty:
                    all_constituents_dfs.append(current_calculated_index_data_df.copy())
            
            # Add data for other selected saved indexes
            for index_name in selected_constituents_for_factsheet:
                if index_name == "Newly Calculated Index": # Already handled if present
                    continue
                selected_db_index_data = next((idx for idx in saved_indexes if idx['index_name'] == index_name), None)
                if selected_db_index_data:
                    all_constituents_dfs.append(pd.DataFrame(selected_db_index_data['constituents']).copy())

            if all_constituents_dfs:
                # Combine all selected constituents and aggregate weights if duplicates exist
                factsheet_constituents_df_final = pd.concat(all_constituents_dfs, ignore_index=True)
                
                # Aggregate weights for duplicate symbols
                factsheet_constituents_df_final = factsheet_constituents_df_final.groupby(['symbol', 'Name'])['Weights'].sum().reset_index()
                factsheet_constituents_df_final['Weights'] = factsheet_constituents_df_final['Weights'] / factsheet_constituents_df_final['Weights'].sum() # Re-normalize combined weights

                # Determine the factsheet index name
                if len(selected_constituents_for_factsheet) == 1:
                    factsheet_index_name_final = selected_constituents_for_factsheet[0]
                    # If only one index selected, try to get its historical data for the chart
                    if factsheet_index_name_final == "Newly Calculated Index":
                         factsheet_history_df_final = current_calculated_index_history_df.copy()
                    else: # A single saved index
                        selected_db_index_data = next((idx for idx in saved_indexes if idx['index_name'] == factsheet_index_name_final), None)
                        if selected_db_index_data and selected_db_index_data.get('historical_performance'):
                            history_from_db = pd.DataFrame(selected_db_index_data['historical_performance'])
                            if not history_from_db.empty:
                                history_from_db['date'] = pd.to_datetime(history_from_db['date'])
                                history_from_db.set_index('date', inplace=True)
                                history_from_db.sort_index(inplace=True)
                                factsheet_history_df_final = history_from_db
                else:
                    factsheet_index_name_final = "Combined Index Constituents Report"

                # Fetch live prices for the combined constituents
                live_quotes_for_factsheet_final = {}
                symbols_for_ltp_for_factsheet_final = [sym for sym in factsheet_constituents_df_final["symbol"]]
                if not st.session_state["instruments_df"].empty and symbols_for_ltp_for_factsheet_final:
                    try:
                        kc_client = get_authenticated_kite_client(api_key, access_token)
                        if kc_client:
                            instrument_identifiers = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols_for_ltp_for_factsheet_final]
                            ltp_data_batch_for_factsheet_final = kc_client.ltp(instrument_identifiers)
                            for sym in symbols_for_ltp_for_factsheet_final:
                                key = f"{DEFAULT_EXCHANGE}:{sym}"
                                live_quotes_for_factsheet_final[sym] = ltp_data_batch_for_factsheet_final.get(key, {}).get("last_price", np.nan)
                    except Exception as e:
                        st.warning(f"Error fetching batch LTP for factsheet live value: {e}. Live prices might be partial.")
                
                # Ensure 'Name' column exists
                if 'Name' not in factsheet_constituents_df_final.columns and not st.session_state["instruments_df"].empty:
                    instrument_names_for_factsheet_final = st.session_state["instruments_df"].set_index('tradingsymbol')['name'].to_dict()
                    factsheet_constituents_df_final['Name'] = factsheet_constituents_df_final['symbol'].map(instrument_names_for_factsheet_final).fillna(factsheet_constituents_df_final['symbol'])
                elif 'Name' not in factsheet_constituents_df_final.columns:
                    factsheet_constituents_df_final['Name'] = factsheet_constituents_df_final['symbol']

                factsheet_constituents_df_final["Last Price"] = factsheet_constituents_df_final["symbol"].map(live_quotes_for_factsheet_final)
                factsheet_constituents_df_final["Weighted Price"] = factsheet_constituents_df_final["Last Price"] * factsheet_constituents_df_final["Weights"]
                current_live_value_for_factsheet_final = factsheet_constituents_df_final["Weighted Price"].sum() if not factsheet_constituents_df_final["Weighted Price"].empty else 0.0
            else: # If all_constituents_dfs is empty after filtering
                factsheet_constituents_df_final = pd.DataFrame()
                factsheet_history_df_final = pd.DataFrame()
                factsheet_index_name_final = "Comparison Report" if not last_comparison_df.empty else "Consolidated Report"
                current_live_value_for_factsheet_final = 0.0
        else: # "None" or nothing selected
            factsheet_constituents_df_final = pd.DataFrame()
            factsheet_history_df_final = pd.DataFrame()
            factsheet_index_name_final = "Comparison Report" if not last_comparison_df.empty else "Consolidated Report"
            current_live_value_for_factsheet_final = 0.0 # Reset live value if no constituents chosen for display

        # AI Agent Embed Snippet input
        ai_agent_snippet_input = st.text_area(
            "Optional: Paste HTML snippet for an embedded AI Agent (e.g., iframe code)",
            height=150,
            key="ai_agent_embed_snippet_input",
            value="<iframe\n  src=\"https://etlas-v5.vercel.app/chat-agent?id=93dee35f-0ebe-42f6-beef-9a1abd1a6f12\"\n  width=\"400\"\n  height=\"600\"\n  frameborder=\"0\"\n  style=\"border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);\"\n></iframe>"
        )

        col_factsheet_download_options_1, col_factsheet_download_options_2 = st.columns(2)

        with col_factsheet_download_options_1:
            if st.button("Generate & Download Factsheet (CSV)", key="generate_download_factsheet_csv_btn"):
                # Check if there's any data at all to put into factsheet
                if not factsheet_constituents_df_final.empty or not factsheet_history_df_final.empty or not last_comparison_df.empty:
                    factsheet_csv_content = generate_factsheet_csv_content(
                        factsheet_constituents_df_final=factsheet_constituents_df_final,
                        factsheet_history_df_final=factsheet_history_df_final, # Include history in CSV
                        last_comparison_df=last_comparison_df,
                        last_comparison_metrics=st.session_state.get("last_comparison_metrics", {}),
                        current_live_value=current_live_value_for_factsheet_final,
                        index_name=factsheet_index_name_final,
                        ai_agent_embed_snippet=None # CSV doesn't support HTML embeds
                    )
                    st.session_state["last_facts_data"] = factsheet_csv_content.encode('utf-8')
                    st.download_button(
                        label="Download CSV Factsheet",
                        data=st.session_state["last_facts_data"],
                        file_name=f"InvsionConnect_Factsheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="factsheet_download_button_final_csv",
                        help="Includes constituents, historical data, comparison data, and metrics."
                    )
                    st.success("CSV Factsheet generated and ready for download!")
                else:
                    st.warning("No data available to generate a factsheet. Please calculate a new index, load a saved index, or run a comparison first.")

        with col_factsheet_download_options_2:
            if st.button("Generate & Download Factsheet (HTML/PDF)", key="generate_download_factsheet_html_btn"):
                # Check if there's any data at all to put into factsheet
                if not factsheet_constituents_df_final.empty or not factsheet_history_df_final.empty or not last_comparison_df.empty:
                    factsheet_html_content = generate_factsheet_html_content(
                        factsheet_constituents_df_final=factsheet_constituents_df_final,
                        factsheet_history_df_final=factsheet_history_df_final, # Pass history to potentially show chart
                        last_comparison_df=last_comparison_df,
                        last_comparison_metrics=st.session_state.get("last_comparison_metrics", {}),
                        current_live_value=current_live_value_for_factsheet_final,
                        index_name=factsheet_index_name_final,
                        ai_agent_embed_snippet=ai_agent_snippet_input if ai_agent_snippet_input.strip() else None # Pass the user's snippet
                    )
                    st.session_state["last_factsheet_html_data"] = factsheet_html_content.encode('utf-8')

                    # For direct HTML download
                    st.download_button(
                        label="Download HTML Factsheet",
                        data=st.session_state["last_factsheet_html_data"],
                        file_name=f"InvsionConnect_Factsheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        key="factsheet_download_button_final_html",
                        help="Includes charts for performance and composition, and optional embedded AI agent. Open in browser to Print to PDF."
                    )
                    st.success("HTML Factsheet generated and ready for download! (Open in browser, then 'Print to PDF')")
                else:
                    st.warning("No data available to generate a factsheet. Please calculate a new index, load a saved index, or run a comparison first.")

        st.markdown("---")
        st.subheader("6. View/Delete Individual Saved Indexes")
        
        # Ensure `index_names_from_db` is available here
        index_names_from_db_for_selector = [idx['index_name'] for idx in saved_indexes] if saved_indexes else []

        selected_index_to_manage = st.selectbox(
            "Select a single saved index to view details or delete:", 
            ["--- Select ---"] + index_names_from_db_for_selector, 
            key="select_single_saved_index_to_manage"
        )
        # st.session_state['selected_index_to_manage'] = selected_index_to_manage # Store for factsheet logic - this is now handled by factsheet_constituents_selector

        selected_db_index_data = None
        if selected_index_to_manage != "--- Select ---":
            selected_db_index_data = next((idx for idx in saved_indexes if idx['index_name'] == selected_index_to_manage), None)
            if selected_db_index_data:
                loaded_constituents_df = pd.DataFrame(selected_db_index_data['constituents'])
                loaded_historical_performance_raw = selected_db_index_data.get('historical_performance')

                loaded_historical_df = pd.DataFrame()
                is_recalculated_live = False

                if loaded_historical_performance_raw:
                    try:
                        loaded_historical_df = pd.DataFrame(loaded_historical_performance_raw)
                        loaded_historical_df['date'] = pd.to_datetime(loaded_historical_df['date'])
                        loaded_historical_df.set_index('date', inplace=True)
                        loaded_historical_df.sort_index(inplace=True)
                        if loaded_historical_df.empty or 'index_value' not in loaded_historical_df.columns:
                            raise ValueError("Loaded historical data is invalid.")
                    except Exception:
                        st.warning(f"Saved historical data for '{selected_index_to_manage}' is invalid or outdated. Recalculating live...")
                        loaded_historical_df = pd.DataFrame() # Clear invalid data
                
                if loaded_historical_df.empty:
                    min_date = (datetime.now().date() - timedelta(days=365))
                    max_date = datetime.now().date()
                    recalculated_historical_df = _calculate_historical_index_value(api_key, access_token, loaded_constituents_df, min_date, max_date, DEFAULT_EXCHANGE)
                    
                    if not recalculated_historical_df.empty and "_error" not in recalculated_historical_df.columns:
                        loaded_historical_df = recalculated_historical_df
                        is_recalculated_live = True
                        st.success("Historical data recalculated live successfully.")
                    else:
                        st.error(f"Failed to recalculate historical data: {recalculated_historical_df.get('_error', 'Unknown error')}")

                display_single_index_details(selected_index_to_manage, loaded_constituents_df, loaded_historical_df, selected_db_index_data['id'], is_recalculated_live)
                
                st.markdown("---")
                if st.button(f"Delete Index '{selected_index_to_manage}'", key=f"delete_index_{selected_db_index_data['id']}", type="primary"):
                    try:
                        supabase_client.table("custom_indexes").delete().eq("id", selected_db_index_data['id']).execute()
                        st.success(f"Index '{selected_index_to_manage}' deleted successfully.")
                        st.session_state["saved_indexes"] = [] # Force reload
                        st.rerun()
                    except Exception as e: st.error(f"Error deleting index: {e}")
    else:
        st.info("No saved indexes to manage yet. Load them using the button above.")


# --- Main Application Logic (Tab Rendering) ---
# Global api_key and access_token to pass to tab functions that use cached utility functions.
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

# Removed render_dashboard_tab(k, api_key, access_token)
with tab_market: render_market_historical_tab(k, api_key, access_token)
with tab_custom_index: render_custom_index_tab(k, supabase, api_key, access_token)
