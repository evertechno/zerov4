import streamlit as st
import pandas as pd
import json
import re
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import ta  # Technical Analysis library
import base64
import fitz # PyMuPDF for reading PDFs
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# --- AI Imports ---
try:
    import google.generativeai as genai
    from google.generativeai import types
except ImportError:
    st.error("Google Generative AI library not found. Please install it using `pip install google-generativeai`.")
    st.stop()
    
# --- KiteConnect Imports ---
try:
    from kiteconnect import KiteConnect
except ImportError:
    st.error("KiteConnect library not found. Please install it using `pip install kiteconnect`.")
    st.stop()


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Invsion Connect Pro - Enterprise Portfolio Analytics", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Invsion Connect Pro - Professional Investment Compliance Platform"
    }
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .breach-alert {
        background: #fee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .compliance-pass {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
        margin: 2rem 0;
    }
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3c72;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Invsion Connect Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enterprise-Grade Portfolio Compliance, Risk Analytics & AI-Powered Investment Analysis</p>', unsafe_allow_html=True)

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
BENCHMARK_SYMBOL = "NIFTY 50"

# Regulatory Constants
SEBI_SINGLE_STOCK_LIMIT = 10.0
SEBI_SINGLE_SECTOR_LIMIT = 25.0
SEBI_GROUP_EXPOSURE_LIMIT = 25.0
SEBI_MIN_LIQUID_ASSETS = 5.0

# Initialize session state variables
session_vars = [
    "kite_access_token", "kite_login_response", "instruments_df", "historical_data",
    "last_fetched_symbol", "current_market_data", "holdings_data", "compliance_results_df",
    "advanced_metrics", "ai_analysis_response", "security_level_compliance", "breach_alerts",
    "risk_metrics_detailed", "concentration_analysis", "var_analysis", "stress_test_results",
    "liquidity_analysis", "correlation_analysis", "attribution_analysis", "compliance_history",
    "custom_rule_results" # Added custom_rule_results to session state
]

for var in session_vars:
    if var not in st.session_state:
        if var.endswith('_df') or var.endswith('_analysis') or var.endswith('_results'):
            st.session_state[var] = pd.DataFrame()
        elif var.endswith('_alerts') or var == 'compliance_history' or var == 'custom_rule_results': # Added custom_rule_results
            st.session_state[var] = []
        else:
            st.session_state[var] = None


# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    """Load API credentials from secrets"""
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    gemini_conf = secrets.get("google_gemini", {})
    
    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")
    if not gemini_conf.get("api_key"):
        errors.append("Google Gemini API key")

    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
        st.info("Ensure your `secrets.toml` includes both [kite] and [google_gemini] sections.")
        st.stop()
    return kite_conf, gemini_conf

KITE_CREDENTIALS, GEMINI_CREDENTIALS = load_secrets()
genai.configure(api_key=GEMINI_CREDENTIALS["api_key"])


# --- KiteConnect Client Initialization ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
login_url = kite_unauth_client.login_url()


# --- Enhanced Utility Functions ---

def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    """Get authenticated Kite client"""
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None


@st.cache_data(ttl=86400, show_spinner="Loading instruments...")
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    """Load and cache instrument data"""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: 
        return pd.DataFrame({"_error": ["Kite not authenticated."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if "instrument_token" in df.columns: 
            df["instrument_token"] = df["instrument_token"].astype("int64")
        if 'tradingsymbol' in df.columns and 'name' in df.columns: 
            df = df[['instrument_token', 'tradingsymbol', 'name', 'exchange']]
        return df
    except Exception as e: 
        return pd.DataFrame({"_error": [f"Failed to load instruments: {e}"]})


@st.cache_data(ttl=60)
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    """Get last traded price"""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: 
        return {"_error": "Kite not authenticated."}
    try: 
        return kite_instance.ltp([f"{exchange.upper()}:{symbol.upper()}"]).get(f"{exchange.upper()}:{symbol.upper()}")
    except Exception as e: 
        return {"_error": str(e)}


@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date: datetime.date, 
                               to_date: datetime.date, interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    """Fetch historical market data"""
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: 
        return pd.DataFrame({"_error": ["Kite not authenticated."]})
    
    # Ensure instruments_df is loaded only if kite_instance is valid
    # and access_token exists
    if kite_instance and access_token:
        instruments_df = load_instruments_cached(api_key, access_token)
    else:
        return pd.DataFrame({"_error": ["Kite not authenticated, cannot load instruments."]})

    if "_error" in instruments_df.columns:
        return instruments_df # Propagate error from load_instruments_cached

    token = find_instrument_token(instruments_df, symbol, exchange)
    
    if not token and symbol in ["NIFTY BANK", "NIFTYBANK", "BANKNIFTY", BENCHMARK_SYMBOL, "SENSEX"]:
        index_exchange = "NSE" if symbol not in ["SENSEX"] else "BSE"
        instruments_secondary = load_instruments_cached(api_key, access_token, index_exchange)
        # Ensure instruments_secondary is valid
        if "_error" in instruments_secondary.columns:
            return instruments_secondary
        token = find_instrument_token(instruments_secondary, symbol, index_exchange)
    
    if not token: 
        return pd.DataFrame({"_error": [f"Instrument token not found for {symbol}."]})
    
    try:
        data = kite_instance.historical_data(
            token, 
            from_date=datetime.combine(from_date, datetime.min.time()), 
            to_date=datetime.combine(to_date, datetime.max.time()), 
            interval=interval
        )
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
    """Find instrument token from symbol"""
    if df.empty: 
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None


def add_technical_indicators(df: pd.DataFrame, sma_periods, ema_periods, rsi_window, 
                            macd_fast, macd_slow, macd_signal, bb_window, bb_std_dev) -> pd.DataFrame:
    """Add comprehensive technical indicators"""
    if df.empty or 'close' not in df.columns: 
        return df.copy()
    
    df_copy = df.copy()
    
    # Moving Averages
    for period in sma_periods:
        if period > 0: 
            df_copy[f'SMA_{period}'] = ta.trend.sma_indicator(df_copy['close'], window=period)
    
    for period in ema_periods:
        if period > 0: 
            df_copy[f'EMA_{period}'] = ta.trend.ema_indicator(df_copy['close'], window=period)
    
    # Momentum Indicators
    df_copy['RSI'] = ta.momentum.rsi(df_copy['close'], window=rsi_window)
    
    # MACD
    macd_obj = ta.trend.MACD(df_copy['close'], window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df_copy['MACD'] = macd_obj.macd()
    df_copy['MACD_signal'] = macd_obj.macd_signal()
    df_copy['MACD_hist'] = macd_obj.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df_copy['close'], window=bb_window, window_dev=bb_std_dev)
    df_copy['Bollinger_High'] = bollinger.bollinger_hband()
    df_copy['Bollinger_Low'] = bollinger.bollinger_lband()
    df_copy['Bollinger_Mid'] = bollinger.bollinger_mavg()
    df_copy['Bollinger_Width'] = bollinger.bollinger_wband()
    
    # Returns
    df_copy['Daily_Return'] = df_copy['close'].pct_change() * 100
    
    # Fill missing values
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    
    return df_copy.dropna()


def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """Calculate comprehensive performance metrics"""
    if returns_series.empty or len(returns_series) < 2: 
        return {}
    
    daily_returns_decimal = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    if daily_returns_decimal.empty: 
        return {}
    
    cumulative_returns = (1 + daily_returns_decimal).cumprod() - 1
    total_return = cumulative_returns.iloc[-1] * 100 if not cumulative_returns.empty else 0
    annualized_return = ((1 + daily_returns_decimal.mean()) ** TRADING_DAYS_PER_YEAR - 1) * 100
    annualized_volatility = daily_returns_decimal.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
    
    # Adjusted Sharpe ratio calculation to avoid division by zero
    risk_free_rate_decimal = risk_free_rate / 100.0
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 1e-9 else np.nan # Use small epsilon
    
    if not cumulative_returns.empty:
        max_drawdown = (((1 + cumulative_returns).cummax() - (1 + cumulative_returns)) / \
                       (1 + cumulative_returns).cummax()).max() * 100
    else:
        max_drawdown = np.nan
    
    def round_if_float(x): 
        return round(x, 4) if isinstance(x, (int, float)) and not np.isnan(x) else np.nan
    
    return {
        "Total Return (%)": round_if_float(total_return),
        "Annualized Return (%)": round_if_float(annualized_return),
        "Annualized Volatility (%)": round_if_float(annualized_volatility),
        "Sharpe Ratio": round_if_float(sharpe_ratio),
        "Max Drawdown (%)": round_if_float(max_drawdown)
    }


# --- ENHANCED COMPLIANCE FUNCTIONS ---

def parse_and_validate_rules_enhanced(rules_text: str, portfolio_df: pd.DataFrame) -> List[Dict]:
    """Enhanced rule parser with comprehensive validation capabilities"""
    results = []
    if not rules_text.strip() or portfolio_df.empty: 
        return results
    
    # Prepare aggregations
    sector_weights = portfolio_df.groupby('Industry')['Weight %'].sum()
    stock_weights = portfolio_df.set_index('Symbol')['Weight %']
    rating_weights = portfolio_df.groupby('Rating')['Weight %'].sum() if 'Rating' in portfolio_df.columns else pd.Series()
    asset_class_weights = portfolio_df.groupby('Asset Class')['Weight %'].sum() if 'Asset Class' in portfolio_df.columns else pd.Series()
    market_cap_weights = portfolio_df.groupby('Market Cap')['Weight %'].sum() if 'Market Cap' in portfolio_df.columns else pd.Series()
    
    def check_pass(actual, op, threshold):
        """Check if condition passes"""
        ops = {'>': lambda a, t: a > t, '<': lambda a, t: a < t, '>=': lambda a, t: a >= t,
               '<=': lambda a, t: a <= t, '=': lambda a, t: a == t}
        return ops.get(op, lambda a, t: False)(actual, threshold)
    
    for rule in rules_text.strip().split('\n'):
        rule = rule.strip()
        if not rule or rule.startswith('#'): 
            continue
        
        parts = re.split(r'\s+', rule)
        rule_type = parts[0].upper()
        
        try:
            actual_value = None
            details = ""
            
            if len(parts) < 3:
                results.append({
                    'rule': rule, 'status': 'Error', 'details': 'Invalid format.', 
                    'severity': 'N/A', 'rule_type': rule_type
                })
                continue
            
            op = parts[-2]
            if op not in ['>', '<', '>=', '<=', '=']:
                results.append({
                    'rule': rule, 'status': 'Error', 'details': f"Invalid operator '{op}'.", 
                    'severity': 'N/A', 'rule_type': rule_type
                })
                continue
            
            threshold = float(parts[-1].replace('%', ''))
            
            # Rule Processing Logic
            if rule_type == 'STOCK' and len(parts) == 4:
                symbol = parts[1].upper()
                if symbol in stock_weights.index:
                    actual_value = stock_weights.get(symbol, 0.0)
                    details = f"Actual for {symbol}: {actual_value:.2f}%"
                else:
                    results.append({
                        'rule': rule, 'status': '⚠️ Invalid', 'details': f"Symbol '{symbol}' not found.", 
                        'severity': 'N/A', 'rule_type': rule_type
                    })
                    continue
            
            elif rule_type == 'SECTOR':
                sector_name = ' '.join(parts[1:-2]).upper()
                matching_sector = next((s for s in sector_weights.index if s.upper() == sector_name), None)
                if matching_sector:
                    actual_value = sector_weights.get(matching_sector, 0.0)
                    details = f"Actual for {matching_sector}: {actual_value:.2f}%"
                else:
                    results.append({
                        'rule': rule, 'status': '⚠️ Invalid', 'details': f"Sector '{sector_name}' not found.", 
                        'severity': 'N/A', 'rule_type': rule_type
                    })
                    continue
            
            elif rule_type == 'RATING':
                rating_name = ' '.join(parts[1:-2]).upper()
                actual_value = rating_weights.get(rating_name, 0.0)
                details = f"Actual for {rating_name}: {actual_value:.2f}%"
            
            elif rule_type == 'ASSET_CLASS':
                class_name = ' '.join(parts[1:-2]).upper()
                actual_value = asset_class_weights.get(class_name, 0.0)
                details = f"Actual for {class_name}: {actual_value:.2f}%"
            
            elif rule_type == 'MARKET_CAP':
                cap_name = ' '.join(parts[1:-2]).upper()
                actual_value = market_cap_weights.get(cap_name, 0.0)
                details = f"Actual for {cap_name}: {actual_value:.2f}%"
            
            elif rule_type == 'TOP_N_STOCKS' and len(parts) == 4:
                n = int(parts[1])
                actual_value = portfolio_df.nlargest(n, 'Weight %')['Weight %'].sum()
                details = f"Actual weight of top {n} stocks: {actual_value:.2f}%"
            
            elif rule_type == 'TOP_N_SECTORS' and len(parts) == 4:
                n = int(parts[1])
                actual_value = sector_weights.nlargest(n).sum()
                details = f"Actual weight of top {n} sectors: {actual_value:.2f}%"
            
            elif rule_type == 'COUNT_STOCKS' and len(parts) == 3:
                actual_value = len(portfolio_df)
                details = f"Actual count: {actual_value}"
            
            elif rule_type == 'COUNT_SECTORS' and len(parts) == 3:
                actual_value = portfolio_df['Industry'].nunique()
                details = f"Actual count: {actual_value}"
            
            elif rule_type == 'ISSUER_GROUP':
                group_name = ' '.join(parts[1:-2]).upper()
                if 'Issuer Group' in portfolio_df.columns:
                    actual_value = portfolio_df[portfolio_df['Issuer Group'].str.upper() == group_name]['Weight %'].sum()
                    details = f"Actual for {group_name}: {actual_value:.2f}%"
                else:
                    results.append({
                        'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Issuer Group' not found.", 
                        'severity': 'N/A', 'rule_type': rule_type
                    })
                    continue
            
            elif rule_type == 'MIN_LIQUIDITY' and len(parts) == 4:
                symbol = parts[1].upper()
                if 'Avg Volume (90d)' in portfolio_df.columns:
                    stock_row = portfolio_df[portfolio_df['Symbol'] == symbol]
                    if not stock_row.empty:
                        actual_value = stock_row['Avg Volume (90d)'].values[0]
                        details = f"Actual volume for {symbol}: {actual_value:,.0f}"
                    else:
                        results.append({
                            'rule': rule, 'status': '⚠️ Invalid', 'details': f"Symbol '{symbol}' not found.", 
                            'severity': 'N/A', 'rule_type': rule_type
                        })
                        continue
                else:
                    results.append({
                        'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Avg Volume (90d)' not found.", 
                        'severity': 'N/A', 'rule_type': rule_type
                    })
                    continue
            
            elif rule_type == 'UNRATED_EXPOSURE' and len(parts) == 3:
                if 'Rating' in portfolio_df.columns:
                    unrated_mask = portfolio_df['Rating'].isin(['UNRATED', 'NR', 'NOT RATED', '', 'UNKNOWN'])
                    actual_value = portfolio_df[unrated_mask]['Weight %'].sum()
                    details = f"Actual unrated exposure: {actual_value:.2f}%"
                else:
                    results.append({
                        'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Rating' not found.", 
                        'severity': 'N/A', 'rule_type': rule_type
                    })
                    continue
            
            elif rule_type == 'FOREIGN_EXPOSURE' and len(parts) == 3:
                if 'Country' in portfolio_df.columns:
                    foreign_mask = portfolio_df['Country'].str.upper() != 'INDIA'
                    actual_value = portfolio_df[foreign_mask]['Weight %'].sum()
                    details = f"Actual foreign exposure: {actual_value:.2f}%"
                else:
                    results.append({
                        'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Country' not found.", 
                        'severity': 'N/A', 'rule_type': rule_type
                    })
                    continue
            
            elif rule_type == 'DERIVATIVES_EXPOSURE' and len(parts) == 3:
                if 'Instrument Type' in portfolio_df.columns:
                    deriv_mask = portfolio_df['Instrument Type'].str.upper().isin(['FUTURES', 'OPTIONS', 'SWAPS'])
                    actual_value = portfolio_df[deriv_mask]['Weight %'].sum()
                    details = f"Actual derivatives exposure: {actual_value:.2f}%"
                else:
                    results.append({
                        'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Instrument Type' not found.", 
                        'severity': 'N/A', 'rule_type': rule_type
                    })
                    continue
            
            # Additional rule types
            elif rule_type == 'PORTFOLIO_TURNOVER' and len(parts) == 3:
                if 'Turnover' in portfolio_df.columns: # Assuming 'Turnover' is a column in the uploaded data
                    actual_value = portfolio_df['Turnover'].mean()
                    details = f"Actual average turnover: {actual_value:.2f}%"
                else:
                    results.append({
                        'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Turnover' not found.", 
                        'severity': 'N/A', 'rule_type': rule_type
                    })
                    continue
            
            elif rule_type == 'EXPENSE_RATIO' and len(parts) == 3:
                if 'Expense Ratio' in portfolio_df.columns: # Assuming 'Expense Ratio' is a column in the uploaded data
                    actual_value = portfolio_df['Expense Ratio'].iloc[0] if not portfolio_df.empty else 0
                    details = f"Actual expense ratio: {actual_value:.2f}%"
                else:
                    results.append({
                        'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Expense Ratio' not found.", 
                        'severity': 'N/A', 'rule_type': rule_type
                    })
                    continue
            
            else:
                results.append({
                    'rule': rule, 'status': 'Error', 'details': 'Unrecognized rule format.', 
                    'severity': 'N/A', 'rule_type': rule_type
                })
                continue
            
            # Evaluate compliance
            if actual_value is not None:
                passed = check_pass(actual_value, op, threshold)
                status = "✅ PASS" if passed else "❌ FAIL"
                
                # Determine severity
                if not passed:
                    breach_magnitude = abs(actual_value - threshold)
                    breach_percent = (breach_magnitude / threshold * 100) if threshold != 0 else 100
                    
                    if breach_percent > 50:
                        severity = "🔴 Critical"
                    elif breach_percent > 25:
                        severity = "🟠 High"
                    elif breach_percent > 10:
                        severity = "🟡 Medium"
                    else:
                        severity = "🟢 Low"
                else:
                    severity = "✅ Compliant"
                
                results.append({
                    'rule': rule,
                    'rule_type': rule_type,
                    'status': status,
                    'details': f"{details} | Rule: {op} {threshold}",
                    'severity': severity,
                    'actual_value': actual_value,
                    'threshold': threshold,
                    'breach_amount': actual_value - threshold if not passed else 0,
                    'breach_percent': breach_percent if not passed else 0
                })
        
        except (ValueError, IndexError) as e:
            results.append({
                'rule': rule, 'status': 'Error', 'details': f"Parse error: {e}", 
                'severity': 'N/A', 'rule_type': rule_type
            })
    
    return results


def calculate_security_level_compliance(portfolio_df: pd.DataFrame, rules_config: dict) -> pd.DataFrame:
    """Calculate comprehensive compliance metrics at individual security level"""
    if portfolio_df.empty:
        return pd.DataFrame()
    
    security_compliance = portfolio_df.copy()
    
    # Single stock limit check
    single_stock_limit = rules_config.get('single_stock_limit', SEBI_SINGLE_STOCK_LIMIT)
    security_compliance['Stock Limit Status'] = security_compliance['Weight %'].apply(
        lambda x: '❌ Breach' if x > single_stock_limit else '✅ Compliant'
    )
    security_compliance['Stock Limit Gap (%)'] = single_stock_limit - security_compliance['Weight %']
    security_compliance['Stock Limit Utilization (%)'] = (security_compliance['Weight %'] / single_stock_limit * 100).round(2)
    
    # Liquidity check
    if 'Avg Volume (90d)' in security_compliance.columns:
        min_liquidity = rules_config.get('min_liquidity', 100000)
        security_compliance['Liquidity Status'] = security_compliance['Avg Volume (90d)'].apply(
            lambda x: '✅ Adequate' if x >= min_liquidity else '⚠️ Low' if x >= min_liquidity * 0.5 else '🔴 Critical'
        )
        security_compliance['Liquidity Score'] = (security_compliance['Avg Volume (90d)'] / min_liquidity * 100).clip(0, 100).round(2)
    else: # If Avg Volume is missing, set default values
        security_compliance['Liquidity Status'] = 'N/A - Missing Data'
        security_compliance['Liquidity Score'] = 0.0
    
    # Rating check
    if 'Rating' in security_compliance.columns:
        investment_grade_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
        security_compliance['Rating Category'] = security_compliance['Rating'].apply(
            lambda x: 'Investment Grade' if x in investment_grade_ratings else 
                     'Below Investment Grade' if x not in ['UNKNOWN', 'UNRATED', 'NR', ''] else 'Unrated'
        )
        security_compliance['Rating Compliance'] = security_compliance['Rating Category'].apply(
            lambda x: '✅ Compliant' if x == 'Investment Grade' else '⚠️ Below Threshold' if x == 'Below Investment Grade' else '🔴 Unrated'
        )
    else: # If Rating is missing, set default values
        security_compliance['Rating Category'] = 'N/A - Missing Data'
        security_compliance['Rating Compliance'] = '🔴 Unrated'
    
    # Concentration risk scoring
    security_compliance['Concentration Risk Score'] = security_compliance['Weight %'].apply(
        lambda x: 100 if x > 10 else 80 if x > 8 else 60 if x > 6 else 40 if x > 4 else 20
    )
    security_compliance['Concentration Risk'] = security_compliance['Weight %'].apply(
        lambda x: '🔴 Critical' if x > 10 else '🟠 High' if x > 8 else '🟡 Medium' if x > 5 else '🟢 Low'
    )
    
    # Sector concentration within security - ensure 'Industry' exists
    if 'Industry' in security_compliance.columns:
        sector_totals = security_compliance.groupby('Industry')['Weight %'].sum()
        security_compliance['Sector Weight'] = security_compliance['Industry'].map(sector_totals)
        # Fix: Ensure the result of the division is cast to a float before calling .round()
        security_compliance['% of Sector'] = security_compliance.apply(
            lambda row: (float(row['Weight %']) / row['Sector Weight'] * 100).round(2) if row['Sector Weight'] > 0 else 0.0,
            axis=1
        )
    else:
        security_compliance['Sector Weight'] = 0.0
        security_compliance['% of Sector'] = 0.0
    
    # Overall compliance score (0-100)
    def calculate_compliance_score(row):
        score = 100
        
        # Deduct for stock limit breach
        if row['Stock Limit Status'] == '❌ Breach':
            breach_severity = (row['Weight %'] - single_stock_limit) / single_stock_limit
            score -= min(40, breach_severity * 100)
        
        # Deduct for liquidity issues
        if 'Liquidity Status' in row and row['Liquidity Status'] == '🔴 Critical':
            score -= 25
        elif 'Liquidity Status' in row and row['Liquidity Status'] == '⚠️ Low':
            score -= 15
        
        # Deduct for rating issues
        if 'Rating Compliance' in row and row['Rating Compliance'] == '🔴 Unrated':
            score -= 20
        elif 'Rating Compliance' in row and row['Rating Compliance'] == '⚠️ Below Threshold':
            score -= 10
        
        return max(0, score)
    
    security_compliance['Compliance Score'] = security_compliance.apply(calculate_compliance_score, axis=1)
    security_compliance['Overall Status'] = security_compliance['Compliance Score'].apply(
        lambda x: '🟢 Excellent' if x >= 90 else '🟡 Good' if x >= 75 else '🟠 Fair' if x >= 60 else '🔴 Poor'
    )
    
    return security_compliance


def calculate_advanced_metrics(portfolio_df: pd.DataFrame, api_key: str, access_token: str) -> Dict:
    """Calculate comprehensive advanced risk metrics"""
    if portfolio_df.empty:
        return None # Return None if portfolio_df is empty
    
    symbols = portfolio_df['Symbol'].tolist()
    total_value_sum = portfolio_df['Real-time Value (Rs)'].sum()
    if total_value_sum == 0: # Avoid division by zero if total value is zero
        st.warning("Total portfolio value is zero. Cannot calculate weighted returns.")
        return None
    weights = (portfolio_df['Real-time Value (Rs)'] / total_value_sum).values
    
    from_date = datetime.now().date() - timedelta(days=366)
    to_date = datetime.now().date()
    
    returns_df = pd.DataFrame()
    failed_symbols = []
    
    progress_bar = st.progress(0, "Fetching historical data for advanced analytics...")
    
    for i, symbol in enumerate(symbols):
        hist_data = get_historical_data_cached(api_key, access_token, symbol, from_date, to_date, 'day')
        if not hist_data.empty and "_error" not in hist_data.columns:
            returns_df[symbol] = hist_data['close'].pct_change()
        else:
            failed_symbols.append(symbol)
        progress_bar.progress((i + 1) / len(symbols), f"Processing {symbol}...")
    
    progress_bar.empty()
    
    if failed_symbols:
        st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_symbols[:5])}{'...' if len(failed_symbols) > 5 else ''}")
    
    returns_df.dropna(how='all', inplace=True)
    returns_df.fillna(0, inplace=True)
    
    if returns_df.empty:
        st.error("Insufficient historical data for advanced metrics.")
        return None
    
    # Portfolio returns
    portfolio_returns = returns_df.dot(weights)
    
    # Value at Risk calculations
    var_95 = portfolio_returns.quantile(0.05) if not portfolio_returns.empty else np.nan
    var_99 = portfolio_returns.quantile(0.01) if not portfolio_returns.empty else np.nan
    var_90 = portfolio_returns.quantile(0.10) if not portfolio_returns.empty else np.nan
    
    # Conditional VaR (Expected Shortfall)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if not portfolio_returns.empty else np.nan
    cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean() if not portfolio_returns.empty else np.nan
    
    # Benchmark data
    benchmark_data = get_historical_data_cached(api_key, access_token, BENCHMARK_SYMBOL, from_date, to_date, 'day')
    
    portfolio_beta = None
    alpha = None
    tracking_error = None
    information_ratio = None
    treynor_ratio = None
    jensen_alpha = None
    
    if benchmark_data.empty or '_error' in benchmark_data.columns:
        st.warning(f"⚠️ Benchmark data unavailable. Beta, Alpha, and related metrics will be N/A.")
    else:
        benchmark_returns = benchmark_data['close'].pct_change()
        # Align portfolio and benchmark returns
        aligned_returns = pd.concat([portfolio_returns.rename('portfolio'), benchmark_returns.rename('benchmark')], axis=1, join='inner').dropna()
        
        if not aligned_returns.empty and len(aligned_returns) > 1: # Ensure enough data points for covariance
            # Beta calculation
            covariance = aligned_returns.cov().iloc[0, 1]
            benchmark_variance = aligned_returns['benchmark'].var()
            portfolio_beta = covariance / benchmark_variance if benchmark_variance > 1e-9 else None # Avoid division by zero
            
            # Alpha calculation
            portfolio_annual_return = ((1 + aligned_returns['portfolio'].mean()) ** TRADING_DAYS_PER_YEAR - 1)
            benchmark_annual_return = ((1 + aligned_returns['benchmark'].mean()) ** TRADING_DAYS_PER_YEAR - 1)
            risk_free_rate = 0.06  # 6% assumed
            
            if portfolio_beta is not None:
                alpha = portfolio_annual_return - (risk_free_rate + portfolio_beta * (benchmark_annual_return - risk_free_rate))
                jensen_alpha = alpha  # Jensen's Alpha
            
            # Tracking Error
            tracking_diff = aligned_returns['portfolio'] - aligned_returns['benchmark']
            tracking_error = tracking_diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(tracking_diff) > 1 else None
            
            # Information Ratio
            if tracking_error and tracking_error > 1e-9: # Avoid division by zero
                information_ratio = (portfolio_annual_return - benchmark_annual_return) / tracking_error
            
            # Treynor Ratio
            if portfolio_beta is not None and portfolio_beta > 1e-9: # Avoid division by zero
                treynor_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_beta
        else:
            st.warning("Not enough aligned historical data for Beta, Alpha, and related metrics.")
    
    # Sortino Ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 0
    portfolio_annual_return = ((1 + portfolio_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1)
    sortino_ratio = (portfolio_annual_return - 0.06) / downside_std if downside_std > 1e-9 else np.nan
    
    # Calmar Ratio
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    max_drawdown = ((cumulative_returns.cummax() - cumulative_returns) / (cumulative_returns.cummax() + 1)).max() if not cumulative_returns.empty else np.nan
    calmar_ratio = portfolio_annual_return / max_drawdown if max_drawdown > 1e-9 else np.nan
    
    # Correlation analysis
    correlation_matrix = returns_df.corr() if not returns_df.empty else pd.DataFrame()
    avg_correlation, max_correlation, min_correlation = np.nan, np.nan, np.nan
    if not correlation_matrix.empty:
        # Check if there are off-diagonal elements to compute correlation
        if correlation_matrix.shape[0] > 1:
            upper_triangle_indices = np.triu_indices_from(correlation_matrix.values, k=1)
            if upper_triangle_indices[0].size > 0: # Check if there are any elements in the upper triangle
                avg_correlation = correlation_matrix.values[upper_triangle_indices].mean()
                max_correlation = correlation_matrix.values[upper_triangle_indices].max()
                min_correlation = correlation_matrix.values[upper_triangle_indices].min()
        else: # Only one asset, correlation is not meaningful
            avg_correlation, max_correlation, min_correlation = 1.0, 1.0, 1.0 # Or set to NaN as it's not a portfolio correlation

    # Diversification metrics
    portfolio_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(portfolio_returns) > 1 else np.nan
    weighted_vol = np.sum(weights * returns_df.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if not returns_df.empty else np.nan
    diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 1e-9 else np.nan
    
    # Effective number of holdings
    effective_n = 1 / np.sum(weights ** 2) if len(weights) > 0 else np.nan
    
    # Skewness and Kurtosis
    skewness = portfolio_returns.skew() if len(portfolio_returns) > 1 else np.nan
    kurtosis = portfolio_returns.kurtosis() if len(portfolio_returns) > 1 else np.nan
    
    # Maximum gain/loss
    max_daily_gain = portfolio_returns.max() if not portfolio_returns.empty else np.nan
    max_daily_loss = portfolio_returns.min() if not portfolio_returns.empty else np.nan
    
    # Win rate
    positive_days = (portfolio_returns > 0).sum()
    total_days = len(portfolio_returns)
    win_rate = positive_days / total_days if total_days > 0 else np.nan
    
    # Ulcer Index (measure of downside volatility)
    drawdown_series = (cumulative_returns.cummax() - cumulative_returns) / (cumulative_returns.cummax() + 1) if not cumulative_returns.empty else pd.Series()
    ulcer_index = np.sqrt((drawdown_series ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR) if not drawdown_series.empty else np.nan
    
    # Modified Sharpe Ratio (accounts for skewness and kurtosis)
    sharpe_ratio_calc = (portfolio_annual_return - 0.06) / portfolio_vol if portfolio_vol > 1e-9 else np.nan
    modified_sharpe = np.nan
    if not np.isnan(sharpe_ratio_calc) and not np.isnan(skewness) and not np.isnan(kurtosis):
        modified_sharpe = sharpe_ratio_calc * (1 + (skewness / 6) * sharpe_ratio_calc - ((kurtosis - 3) / 24) * sharpe_ratio_calc ** 2)
    
    return {
        "portfolio_returns": portfolio_returns, # Include portfolio returns for plotting
        "var_95": var_95,
        "var_99": var_99,
        "var_90": var_90,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "beta": portfolio_beta,
        "alpha": alpha,
        "jensen_alpha": jensen_alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "sortino_ratio": sortino_ratio,
        "treynor_ratio": treynor_ratio,
        "calmar_ratio": calmar_ratio,
        "avg_correlation": avg_correlation,
        "max_correlation": max_correlation,
        "min_correlation": min_correlation,
        "diversification_ratio": diversification_ratio,
        "portfolio_volatility": portfolio_vol,
        "effective_n_holdings": effective_n,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "max_daily_gain": max_daily_gain,
        "max_daily_loss": max_daily_loss,
        "win_rate": win_rate,
        "ulcer_index": ulcer_index,
        "sharpe_ratio": sharpe_ratio_calc, # Use the calculated sharpe_ratio_calc
        "modified_sharpe_ratio": modified_sharpe,
        "max_drawdown": max_drawdown # Added Max Drawdown to advanced_metrics
    }


def calculate_concentration_metrics(portfolio_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive concentration risk metrics"""
    
    if portfolio_df.empty:
        return {} # Return empty dict if portfolio is empty
    
    # Herfindahl-Hirschman Index (HHI)
    stock_hhi = (portfolio_df['Weight %'] ** 2).sum()
    sector_hhi = (portfolio_df.groupby('Industry')['Weight %'].sum() ** 2).sum()
    
    # Concentration categories
    def categorize_hhi(hhi):
        if hhi < 1500:
            return "🟢 Low Concentration", "Well diversified"
        elif hhi < 2500:
            return "🟡 Moderate Concentration", "Moderately concentrated"
        else:
            return "🔴 High Concentration", "Highly concentrated"
    
    stock_hhi_category, stock_hhi_desc = categorize_hhi(stock_hhi)
    sector_hhi_category, sector_hhi_desc = categorize_hhi(sector_hhi)
    
    # Gini coefficient (inequality measure)
    sorted_weights = np.sort(portfolio_df['Weight %'].values)
    n = len(sorted_weights)
    if n > 0 and np.sum(sorted_weights) > 0: # Avoid division by zero
        cumsum = np.cumsum(sorted_weights)
        # Using simplified Gini formula: (2 * sum(i * yi) / (n * sum(yi))) - (n+1)/n
        # Or alternatively: 1 - sum_{i=0}^{n-1} (y_{i+1} + y_i) / sum_{i=0}^{n-1} y_i * 1/n
        # More common formula: 1/n * (n+1 - 2 * sum(rank_i * val_i) / sum(val_i))
        # Let's use the provided one with safety checks:
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
    else:
        gini = np.nan
    
    # Effective number of holdings
    weights_normalized = portfolio_df['Weight %'] / 100
    effective_n_stocks = 1 / (weights_normalized ** 2).sum() if not weights_normalized.empty and (weights_normalized ** 2).sum() > 0 else np.nan

    sector_weights = portfolio_df.groupby('Industry')['Weight %'].sum() / 100
    effective_n_sectors = 1 / (sector_weights ** 2).sum() if not sector_weights.empty and (sector_weights ** 2).sum() > 0 else np.nan
    
    # Top N concentration
    top_1 = portfolio_df.nlargest(1, 'Weight %')['Weight %'].sum() if not portfolio_df.empty else 0
    top_3 = portfolio_df.nlargest(3, 'Weight %')['Weight %'].sum() if not portfolio_df.empty else 0
    top_5 = portfolio_df.nlargest(5, 'Weight %')['Weight %'].sum() if not portfolio_df.empty else 0
    top_10 = portfolio_df.nlargest(10, 'Weight %')['Weight %'].sum() if not portfolio_df.empty else 0
    top_20 = portfolio_df.nlargest(20, 'Weight %')['Weight %'].sum() if len(portfolio_df) >= 20 else portfolio_df['Weight %'].sum() if not portfolio_df.empty else 0
    
    # Sector concentration
    sector_analysis = portfolio_df.groupby('Industry')['Weight %'].sum().sort_values(ascending=False)
    top_sector = sector_analysis.iloc[0] if not sector_analysis.empty else 0
    top_3_sectors = sector_analysis.head(3).sum() if not sector_analysis.empty else 0
    
    return {
        "stock_hhi": stock_hhi,
        "stock_hhi_category": stock_hhi_category,
        "stock_hhi_description": stock_hhi_desc,
        "sector_hhi": sector_hhi,
        "sector_hhi_category": sector_hhi_category,
        "sector_hhi_description": sector_hhi_desc,
        "gini_coefficient": gini,
        "effective_n_stocks": effective_n_stocks,
        "effective_n_sectors": effective_n_sectors,
        "top_1_weight": top_1,
        "top_3_weight": top_3,
        "top_5_weight": top_5,
        "top_10_weight": top_10,
        "top_20_weight": top_20,
        "top_sector_weight": top_sector,
        "top_3_sectors_weight": top_3_sectors
    }


def perform_stress_testing(portfolio_df: pd.DataFrame, portfolio_returns: pd.Series, weights: np.ndarray) -> Dict:
    """Perform comprehensive stress testing scenarios"""
    
    scenarios = {}
    
    if portfolio_returns.empty:
        st.warning("Cannot perform stress testing: Insufficient portfolio returns data.")
        return {}
    
    # Historical worst scenarios
    worst_day = portfolio_returns.min() if not portfolio_returns.empty else np.nan
    worst_week = portfolio_returns.rolling(5).sum().min() if len(portfolio_returns) >= 5 else np.nan
    worst_month = portfolio_returns.rolling(21).sum().min() if len(portfolio_returns) >= 21 else np.nan
    
    scenarios['historical'] = {
        'worst_day': worst_day * 100 if not np.isnan(worst_day) else np.nan,
        'worst_week': worst_week * 100 if not np.isnan(worst_week) else np.nan,
        'worst_month': worst_month * 100 if not np.isnan(worst_month) else np.nan
    }
    
    # Hypothetical market crash scenarios
    scenarios['market_crash'] = {
        'moderate_correction_10': -10.0,  # 10% market correction
        'severe_correction_20': -20.0,    # 20% market crash
        'extreme_crash_30': -30.0,         # 30% extreme crash
        'black_swan_50': -50.0             # 50% black swan event
    }
    
    # Sector-specific shocks
    sector_impacts = {}
    if 'Industry' in portfolio_df.columns:
        for sector in portfolio_df['Industry'].unique():
            sector_weight = portfolio_df[portfolio_df['Industry'] == sector]['Weight %'].sum()
            sector_impacts[sector] = {
                'weight': sector_weight,
                'impact_10': sector_weight * -0.10,
                'impact_20': sector_weight * -0.20
            }
    
    scenarios['sector_shocks'] = sector_impacts
    
    # Interest rate scenarios
    scenarios['interest_rate'] = {
        '50bps_hike': 'Estimated -2% to -5% impact on equity valuations',
        '100bps_hike': 'Estimated -5% to -10% impact on equity valuations',
        '50bps_cut': 'Estimated +2% to +5% impact on equity valuations'
    }
    
    return scenarios


def calculate_liquidity_metrics(portfolio_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive liquidity metrics"""
    
    liquidity_metrics = {}
    
    if portfolio_df.empty:
        liquidity_metrics['error'] = 'Portfolio data is empty.'
        return liquidity_metrics
    
    if 'Avg Volume (90d)' in portfolio_df.columns:
        total_value = portfolio_df['Real-time Value (Rs)'].sum()
        
        # Liquidity buckets
        high_liquidity = portfolio_df[portfolio_df['Avg Volume (90d)'] >= 500000]['Real-time Value (Rs)'].sum()
        medium_liquidity = portfolio_df[(portfolio_df['Avg Volume (90d)'] >= 100000) & 
                                       (portfolio_df['Avg Volume (90d)'] < 500000)]['Real-time Value (Rs)'].sum()
        low_liquidity = portfolio_df[portfolio_df['Avg Volume (90d)'] < 100000]['Real-time Value (Rs)'].sum()
        
        liquidity_metrics['high_liquidity_pct'] = (high_liquidity / total_value * 100) if total_value > 0 else 0
        liquidity_metrics['medium_liquidity_pct'] = (medium_liquidity / total_value * 100) if total_value > 0 else 0
        liquidity_metrics['low_liquidity_pct'] = (low_liquidity / total_value * 100) if total_value > 0 else 0
        
        # Average volume weighted by portfolio weight
        liquidity_metrics['weighted_avg_volume'] = (portfolio_df['Avg Volume (90d)'] * 
                                                    portfolio_df['Weight %']).sum() / 100
        
        # Liquidity score (0-100)
        def calculate_liquidity_score(volume):
            if volume >= 1000000:
                return 100
            elif volume >= 500000:
                return 80
            elif volume >= 100000:
                return 60
            elif volume >= 50000:
                return 40
            else:
                return 20
        
        portfolio_df_copy = portfolio_df.copy()
        portfolio_df_copy['Liquidity Score'] = portfolio_df_copy['Avg Volume (90d)'].apply(calculate_liquidity_score)
        liquidity_metrics['portfolio_liquidity_score'] = (portfolio_df_copy['Liquidity Score'] * 
                                                          portfolio_df_copy['Weight %']).sum() / 100
    else:
        liquidity_metrics['error'] = 'Liquidity data not available'
    
    return liquidity_metrics


# --- Sidebar: Enhanced Kite Login ---
with st.sidebar:
    st.markdown("### 🔐 Authentication")
    
    if not st.session_state["kite_access_token"]:
        st.markdown("Connect to your Kite account to access live market data and portfolio analysis.")
        st.link_button("🔗 Connect Kite Account", login_url, use_container_width=True, type="primary")
        st.caption("You'll be redirected to Zerodha for secure authentication")
    
    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        with st.spinner("🔄 Authenticating..."):
            try:
                data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
                st.session_state["kite_access_token"] = data.get("access_token")
                st.session_state["kite_login_response"] = data
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Authentication failed: {e}")
    
    if st.session_state["kite_access_token"]:
        st.success("✅ Kite Connected")
        user_info = st.session_state.get("kite_login_response", {})
        if user_info.get("user_name"):
            st.caption(f"👤 {user_info['user_name']}")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    else:
        st.info("ℹ️ Authentication required for full functionality")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.session_state["kite_access_token"]:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Holdings", key="sidebar_holdings", use_container_width=True):
                k_client = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
                if k_client: # Ensure client is authenticated
                    try:
                        holdings = k_client.holdings()
                        st.session_state["holdings_data"] = pd.DataFrame(holdings)
                        st.success(f"✅ Fetched {len(holdings)} holdings")
                    except Exception as e:
                        st.error(f"❌ Error fetching holdings: {e}")
                else:
                    st.warning("Kite client not authenticated.")
        
        with col2:
            if st.button("💰 Positions", key="sidebar_positions", use_container_width=True):
                k_client = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
                if k_client: # Ensure client is authenticated
                    try:
                        positions = k_client.positions()
                        st.info(f"Net: {len(positions.get('net', []))}, Day: {len(positions.get('day', []))}")
                    except Exception as e:
                        st.error(f"❌ Error fetching positions: {e}")
                else:
                    st.warning("Kite client not authenticated.")
        
        if st.session_state.get("holdings_data") is not None and not st.session_state["holdings_data"].empty:
            with st.expander("📋 Holdings Data"):
                st.dataframe(st.session_state["holdings_data"], use_container_width=True, height=200)
                csv = st.session_state["holdings_data"].to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download CSV",
                    csv,
                    "holdings.csv",
                    "text/csv",
                    use_container_width=True
                )
    else:
        st.caption("Connect Kite account to use quick actions")
    
    st.markdown("---")
    
    # System Info
    st.markdown("### ℹ️ System Info")
    st.caption(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    st.caption(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
    st.caption(f"**Market:** {DEFAULT_EXCHANGE}")
    
    if st.session_state.get("compliance_results_df") is not None and not st.session_state["compliance_results_df"].empty:
        st.caption(f"**Portfolio:** {len(st.session_state['compliance_results_df'])} securities")


# --- Authenticated KiteConnect client ---
# `k` will be None if not authenticated. This is handled by checks within render functions.
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Main Tabs ---
tabs = st.tabs([
    "📈 Market & Technical Analysis", 
    "💼 Investment Compliance Pro", 
    "🤖 AI-Powered Analysis"
])
tab_market, tab_compliance, tab_ai = tabs


# --- TAB 1: Market & Historical Data ---
def render_market_historical_tab(kite_client, api_key, access_token):
    st.header("📈 Market Data & Technical Analysis")
    
    if not kite_client or not access_token: # Check both client and token
        st.info("🔐 Please connect your Kite account to access market data")
        return
    
    # Market Quote Section
    st.subheader("💹 Real-time Market Data")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        q_exchange = st.selectbox("Exchange", ["NSE", "BSE", "NFO", "BFO"], key="market_exchange")
    
    with col2:
        q_symbol = st.text_input("Symbol", value="RELIANCE", key="market_symbol").upper()
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_quote = st.button("🔍 Get Quote", key="get_quote_btn", use_container_width=True, type="primary")
    
    if fetch_quote or st.session_state.get("current_market_data"):
        if fetch_quote:
            # Ensure kite_client is passed to get_ltp_price_cached
            ltp_data = get_ltp_price_cached(api_key, access_token, q_symbol, q_exchange)
            if ltp_data and "_error" not in ltp_data:
                st.session_state["current_market_data"] = ltp_data
            else:
                st.error(f"❌ {ltp_data.get('_error', 'Failed to fetch data')}")
        
        if st.session_state.get("current_market_data"):
            data = st.session_state["current_market_data"]
            cols = st.columns(4)
            cols[0].metric("Last Price", f"₹{data.get('last_price', 0):,.2f}")
            cols[1].metric("Symbol", q_symbol)
            cols[2].metric("Exchange", q_exchange)
            cols[3].metric("Status", "🟢 Live")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Historical Data & Technical Analysis
    st.subheader("📊 Historical Price Data & Technical Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hist_symbol = st.text_input("Symbol for Analysis", value="INFY", key="hist_symbol").upper()
    
    with col2:
        interval = st.selectbox(
            "Interval", 
            ["day", "week", "month", "5minute", "15minute", "30minute", "60minute"],
            key="hist_interval"
        )
    
    with col3:
        from_date = st.date_input(
            "From Date",
            value=datetime.now().date() - timedelta(days=365),
            key="from_date"
        )
    
    with col4:
        to_date = st.date_input(
            "To Date",
            value=datetime.now().date(),
            key="to_date"
        )
    
    if st.button("📥 Fetch Historical Data", key="fetch_hist_btn", type="primary", use_container_width=True):
        with st.spinner(f"🔄 Fetching data for {hist_symbol}..."):
            # Ensure kite_client is passed to get_historical_data_cached
            df_hist = get_historical_data_cached(api_key, access_token, hist_symbol, from_date, to_date, interval, DEFAULT_EXCHANGE)
            if isinstance(df_hist, pd.DataFrame) and "_error" not in df_hist.columns:
                st.session_state["historical_data"] = df_hist
                st.session_state["last_fetched_symbol"] = hist_symbol
                st.success(f"✅ Fetched {len(df_hist)} data points")
            else:
                st.error(f"❌ {df_hist.get('_error', ['Unknown error'])[0]}")
    
    # Check if historical_data is valid before attempting to use it
    if st.session_state.get("historical_data") is not None and not st.session_state["historical_data"].empty and "_error" not in st.session_state["historical_data"].columns:
        df = st.session_state["historical_data"]
        
        with st.expander("⚙️ Technical Indicator Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sma_periods_str = st.text_input("SMA Periods (comma-separated)", "20,50,200")
                ema_periods_str = st.text_input("EMA Periods (comma-separated)", "12,26")
                rsi_window = st.number_input("RSI Window", 5, 50, 14)
            
            with col2:
                macd_fast = st.number_input("MACD Fast", 5, 50, 12)
                macd_slow = st.number_input("MACD Slow", 10, 100, 26)
                macd_signal = st.number_input("MACD Signal", 5, 50, 9)
            
            with col3:
                bb_window = st.number_input("Bollinger Window", 5, 50, 20)
                bb_std_dev = st.number_input("Bollinger Std Dev", 1.0, 4.0, 2.0, 0.5)
                chart_type = st.selectbox("Chart Style", ["Candlestick", "Line", "OHLC"])
        
        sma_periods = [int(p.strip()) for p in sma_periods_str.split(',') if p.strip().isdigit()]
        ema_periods = [int(p.strip()) for p in ema_periods_str.split(',') if p.strip().isdigit()]
        
        df_with_ta = add_technical_indicators(
            df, sma_periods, ema_periods, rsi_window, 
            macd_fast, macd_slow, macd_signal, bb_window, bb_std_dev
        )
        
        # Create comprehensive chart
        st.subheader(f"📊 Technical Analysis: {st.session_state['last_fetched_symbol']}")
        
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.1, 0.2, 0.2],
            subplot_titles=('Price & Indicators', 'Volume', 'RSI', 'MACD')
        )
        
        # Price chart
        if chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=df.index, open=df['open'], high=df['high'], 
                    low=df['low'], close=df['close'], name='Price'
                ),
                row=1, col=1
            )
        elif chart_type == "OHLC":
            fig.add_trace(
                go.Ohlc(
                    x=df.index, open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name='Price'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'),
                row=1, col=1
            )
        
        # Add moving averages
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, period in enumerate(sma_periods):
            if f'SMA_{period}' in df_with_ta.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_ta.index, y=df_with_ta[f'SMA_{period}'],
                        mode='lines', name=f'SMA {period}',
                        line=dict(color=colors[i % len(colors)], width=1)
                    ),
                    row=1, col=1
                )
        
        for i, period in enumerate(ema_periods):
            if f'EMA_{period}' in df_with_ta.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_ta.index, y=df_with_ta[f'EMA_{period}'],
                        mode='lines', name=f'EMA {period}',
                        line=dict(color=colors[(i + len(sma_periods)) % len(colors)], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # Bollinger Bands
        if 'Bollinger_High' in df_with_ta.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_with_ta.index, y=df_with_ta['Bollinger_High'],
                    mode='lines', name='BB Upper', line=dict(width=0.5, color='gray')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_with_ta.index, y=df_with_ta['Bollinger_Low'],
                    mode='lines', name='BB Lower',
                    line=dict(width=0.5, color='gray'),
                    fill='tonexty', fillcolor='rgba(128,128,128,0.2)'
                ),
                row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in df_with_ta.columns:
            fig.add_trace(
                go.Scatter(x=df_with_ta.index, y=df_with_ta['RSI'], mode='lines', name='RSI'),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
        
        # MACD
        if 'MACD' in df_with_ta.columns:
            fig.add_trace(
                go.Bar(x=df_with_ta.index, y=df_with_ta['MACD_hist'], name='MACD Hist', marker_color='orange'),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_with_ta.index, y=df_with_ta['MACD'], mode='lines', name='MACD', line=dict(color='blue')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_with_ta.index, y=df_with_ta['MACD_signal'], mode='lines', name='Signal', line=dict(color='red')),
                row=4, col=1
            )
        
        fig.update_layout(
            height=1000,
            xaxis_rangeslider_visible=False,
            title_text=f"{st.session_state['last_fetched_symbol']} - {interval.upper()} Chart",
            template="plotly_white",
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        metrics = calculate_performance_metrics(df['close'].pct_change(), risk_free_rate=6.0)
        
        if metrics:
            cols = st.columns(5)
            metric_list = list(metrics.items())
            for i, (key, value) in enumerate(metric_list):
                with cols[i % 5]:
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        st.metric(key, f"{value:.2f}")
                    else:
                        st.metric(key, "N/A")
    else:
        st.info("No historical data available to display. Please fetch data first.")


# --- TAB 2: Enhanced Investment Compliance ---
def render_investment_compliance_tab(kite_client, api_key, access_token):
    st.header("💼 Investment Compliance Pro - Enterprise Analytics")
    
    if not kite_client or not access_token: # Check both client and token
        st.warning("🔐 Please connect to Kite to fetch live prices and perform real-time analysis")
        return
    
    # Enhanced UI Layout
    col_upload, col_config = st.columns([3, 2])
    
    with col_upload:
        st.markdown("### 📂 Portfolio Upload")
        uploaded_file = st.file_uploader(
            "Upload Portfolio CSV",
            type="csv",
            help="Required columns: Symbol, Industry, Quantity, Market/Fair Value(Rs. in Lacs). Optional: Rating, Asset Class, Market Cap, Issuer Group, Country, Instrument Type, Avg Volume (90d)"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    with col_config:
        st.markdown("### ⚙️ Compliance Configuration")
        
        with st.expander("🎯 Regulatory Limits", expanded=True):
            single_stock_limit = st.slider(
                "Single Stock Limit (%)",
                1.0, 25.0, SEBI_SINGLE_STOCK_LIMIT, 0.5,
                help="SEBI Default: 10%"
            )
            
            single_sector_limit = st.slider(
                "Single Sector Limit (%)",
                5.0, 50.0, SEBI_SINGLE_SECTOR_LIMIT, 1.0,
                help="SEBI Default: 25%"
            )
            
            top_10_limit = st.slider(
                "Top 10 Holdings Limit (%)",
                20.0, 80.0, 50.0, 5.0
            )
            
            min_holdings = st.number_input(
                "Minimum Holdings Count",
                10, 200, 30, 5
            )
            
            unrated_limit = st.slider(
                "Unrated Securities Limit (%)",
                0.0, 30.0, 10.0, 1.0
            )
    
    # Custom Rules Section
    st.markdown("---")
    st.markdown("### 📋 Custom Compliance Rules")
    
    col_rules, col_guide = st.columns([3, 2])
    
    with col_rules:
        rules_text = st.text_area(
            "Define Custom Rules (one per line)",
            height=250,
            key="compliance_rules",
            value="""# Stock Level Rules
STOCK RELIANCE < 10
STOCK TCS < 8

# Sector Level Rules
SECTOR BANKING < 25
SECTOR IT < 30

# Concentration Rules
TOP_N_STOCKS 5 <= 35
TOP_N_STOCKS 10 <= 50
COUNT_STOCKS >= 30

# Rating Rules
RATING AAA >= 20
UNRATED_EXPOSURE <= 10""",
            help="Define comprehensive compliance rules"
        )
    
    with col_guide:
        st.markdown("#### 📖 Rule Syntax Guide")
        
        with st.expander("View All Rule Types", expanded=False):
            st.markdown("""
            **Stock Rules:**
            - `STOCK [Symbol] <op> [%]`
            - Example: `STOCK RELIANCE < 10`
            
            **Sector Rules:**
            - `SECTOR [Name] <op> [%]`
            - Example: `SECTOR BANKING < 25`
            
            **Concentration:**
            - `TOP_N_STOCKS [N] <op> [%]`
            - `TOP_N_SECTORS [N] <op> [%]`
            
            **Count Rules:**
            - `COUNT_STOCKS <op> [N]`
            - `COUNT_SECTORS <op> [N]`
            
            **Rating/Quality:**
            - `RATING [Rating] <op> [%]`
            - `UNRATED_EXPOSURE <op> [%]`
            
            **Asset/Market Cap:**
            - `ASSET_CLASS [Class] <op> [%]`
            - `MARKET_CAP [Cap] <op> [%]`
            
            **Group/Liquidity:**
            - `ISSUER_GROUP [Group] <op> [%]`
            - `MIN_LIQUIDITY [Symbol] >= [Volume]`
            
            **Operators:** `>`, `<`, `>=`, `<=`, `=`
            """)
    
    # Process Portfolio
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [str(col).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_') for col in df.columns]
            
            # Column mapping
            header_map = {
                'isin': 'ISIN', 'name_of_the_instrument': 'Name', 'symbol': 'Symbol',
                'industry': 'Industry', 'quantity': 'Quantity', 'rating': 'Rating',
                'asset_class': 'Asset Class', 'market_cap': 'Market Cap',
                'issuer_group': 'Issuer Group', 'country': 'Country',
                'instrument_type': 'Instrument Type', 'avg_volume_(90d)': 'Avg Volume (90d)',
                'market_fair_value(rs_in_lacs)': 'Uploaded Value (Lacs)'
            }
            df = df.rename(columns=header_map)
            
            # Ensure mandatory columns exist
            mandatory_cols = ['Symbol', 'Industry', 'Quantity']
            missing_mandatory = [col for col in mandatory_cols if col not in df.columns]
            if missing_mandatory:
                st.error(f"❌ Missing mandatory columns in the uploaded CSV: {', '.join(missing_mandatory)}. Please check your file.")
                st.stop()
            
            # Normalize data
            for col in ['Rating', 'Asset Class', 'Industry', 'Market Cap', 'Issuer Group', 'Country', 'Instrument Type']:
                if col in df.columns:
                    df[col] = df[col].fillna('UNKNOWN').str.strip().str.upper()
            
            st.markdown("---")
            
            col_btn1, col_btn2 = st.columns([1, 3])
            
            with col_btn1:
                analyze_btn = st.button(
                    "🚀 Analyze Portfolio",
                    type="primary",
                    use_container_width=True,
                    key="analyze_portfolio_btn"
                )
            
            if analyze_btn:
                with st.spinner("🔄 Performing comprehensive analysis..."):
                    progress = st.progress(0, "Starting analysis...")
                    
                    # Fetch live prices
                    progress.progress(20, "Fetching live market data...")
                    symbols = df['Symbol'].unique().tolist()
                    ltp_data = kite_client.ltp([f"{DEFAULT_EXCHANGE}:{s}" for s in symbols])
                    prices = {sym: ltp_data.get(f"{DEFAULT_EXCHANGE}:{sym}", {}).get('last_price') for sym in symbols}
                    
                    # Calculate portfolio metrics
                    progress.progress(40, "Calculating portfolio metrics...")
                    df_results = df.copy()
                    df_results['LTP'] = df_results['Symbol'].map(prices)
                    df_results['Real-time Value (Rs)'] = (df_results['LTP'] * pd.to_numeric(df_results['Quantity'], errors='coerce')).fillna(0)
                    total_value = df_results['Real-time Value (Rs)'].sum()
                    df_results['Weight %'] = (df_results['Real-time Value (Rs)'] / total_value * 100) if total_value > 0 else 0
                    
                    # Security-level compliance
                    progress.progress(60, "Evaluating security-level compliance...")
                    rules_config = {
                        'single_stock_limit': single_stock_limit,
                        'single_sector_limit': single_sector_limit,
                        'min_liquidity': 100000
                    }
                    security_compliance = calculate_security_level_compliance(df_results, rules_config)
                    
                    # Concentration analysis
                    progress.progress(70, "Analyzing concentration risk...")
                    concentration_metrics = calculate_concentration_metrics(df_results)

                    # Custom rule validation
                    progress.progress(80, "Validating custom rules...")
                    custom_rule_results = parse_and_validate_rules_enhanced(rules_text, df_results)

                    # Advanced metrics for Risk tab
                    progress.progress(90, "Calculating advanced risk metrics...")
                    # Pass the access token directly
                    advanced_metrics = calculate_advanced_metrics(df_results, api_key, access_token)
                    
                    # Store results
                    st.session_state.compliance_results_df = df_results
                    st.session_state.security_level_compliance = security_compliance
                    st.session_state.concentration_analysis = concentration_metrics
                    st.session_state.custom_rule_results = custom_rule_results # Store custom rule results
                    st.session_state.advanced_metrics = advanced_metrics # Store advanced metrics
                    
                    # Generate breach alerts
                    progress.progress(95, "Generating compliance alerts...")
                    breaches = []
                    
                    # Stock limit breaches
                    if (df_results['Weight %'] > single_stock_limit).any():
                        breach_stocks = df_results[df_results['Weight %'] > single_stock_limit]
                        for _, stock in breach_stocks.iterrows():
                            breaches.append({
                                'type': 'Single Stock Limit',
                                'severity': '🔴 Critical' if stock['Weight %'] > single_stock_limit * 1.5 else '🟠 High',
                                'details': f"{stock['Symbol']} at {stock['Weight %']:.2f}% (Limit: {single_stock_limit}%)",
                                'recommendation': f"Reduce position by {(stock['Weight %'] - single_stock_limit):.2f}%"
                            })
                    
                    # Sector breaches
                    sector_weights = df_results.groupby('Industry')['Weight %'].sum()
                    if (sector_weights > single_sector_limit).any():
                        breach_sectors = sector_weights[sector_weights > single_sector_limit]
                        for sector, weight in breach_sectors.items():
                            breaches.append({
                                'type': 'Sector Limit',
                                'severity': '🟠 High',
                                'details': f"{sector} at {weight:.2f}% (Limit: {single_sector_limit}%)",
                                'recommendation': f"Reduce {sector} exposure by {(weight - single_sector_limit):.2f}%"
                            })
                    
                    # Top 10 concentration
                    top_10_weight = df_results.nlargest(10, 'Weight %')['Weight %'].sum()
                    if top_10_weight > top_10_limit:
                        breaches.append({
                            'type': 'Top 10 Concentration',
                            'severity': '🟡 Medium',
                            'details': f"Top 10 holdings: {top_10_weight:.2f}% (Limit: {top_10_limit}%)",
                            'recommendation': "Increase diversification across portfolio"
                        })

                    # Unrated securities limit
                    if 'Rating' in df_results.columns:
                        unrated_exposure = df_results[df_results['Rating'].isin(['UNRATED', 'NR', 'NOT RATED', '', 'UNKNOWN'])]['Weight %'].sum()
                        if unrated_exposure > unrated_limit:
                            breaches.append({
                                'type': 'Unrated Exposure Limit',
                                'severity': '🟡 Medium',
                                'details': f"Unrated exposure: {unrated_exposure:.2f}% (Limit: {unrated_limit}%)",
                                'recommendation': "Seek ratings for unrated securities or reduce exposure."
                            })
                    
                    st.session_state.breach_alerts = breaches
                    
                    progress.progress(100, "Analysis complete!")
                    time.sleep(0.5)
                    progress.empty()
                    
                    st.success("✅ Comprehensive analysis completed successfully!")
                    
                    if breaches:
                        st.warning(f"⚠️ Found {len(breaches)} compliance breach(es)")
                    else:
                        st.success("🎉 Portfolio is fully compliant with all rules!")
        
        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")
            st.exception(e)
    
    # Display Results
    results_df = st.session_state.get("compliance_results_df", pd.DataFrame())
    
    if not results_df.empty and 'Weight %' in results_df.columns:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Breach Alerts Section
        if st.session_state.get("breach_alerts"):
            st.markdown("### 🚨 Compliance Breach Alerts")
            
            for breach in st.session_state["breach_alerts"]:
                severity_color = {
                    '🔴 Critical': '#ffebee',
                    '🟠 High': '#fff3e0',
                    '🟡 Medium': '#fffde7',
                    '🟢 Low': '#e8f5e9'
                }.get(breach['severity'], '#f5f5f5')
                
                st.markdown(f"""
                <div style="background: {severity_color}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #f44336;">
                    <strong>{breach['severity']} - {breach['type']}</strong><br>
                    {breach['details']}<br>
                    <em>💡 Recommendation: {breach.get('recommendation', 'Review portfolio allocation')}</em>
                </div>
                """, unsafe_allow_html=True)
        
        # Analysis Tabs
        analysis_tabs = st.tabs([
            "📊 Executive Dashboard",
            "🔍 Detailed Analytics",
            "📈 Risk Metrics",
            "⚖️ Rule Validation",
            "🔐 Security Compliance",
            "📉 Concentration Analysis",
            "💧 Liquidity Analysis",
            "📄 Export Report"
        ])
        
        # TAB 1: Executive Dashboard
        with analysis_tabs[0]:
            st.markdown("### 📊 Portfolio Executive Dashboard")
            
            total_value = results_df['Real-time Value (Rs)'].sum()
            
            # KPI Metrics
            kpi_cols = st.columns(6)
            
            kpi_cols[0].metric(
                "Portfolio Value",
                f"₹{total_value:,.0f}",
                help="Total portfolio value at current market prices"
            )
            kpi_cols[1].metric(
                "Holdings",
                f"{len(results_df)}",
                help="Number of unique securities"
            )
            kpi_cols[2].metric(
                "Sectors",
                f"{results_df['Industry'].nunique()}",
                help="Number of unique sectors"
            )
            
            # Compliance status
            breach_count = len(st.session_state.get("breach_alerts", []))
            compliance_rate = ((1 - breach_count / max(len(results_df), 1)) * 100) if breach_count < len(results_df) else 0
            
            kpi_cols[3].metric(
                "Compliance",
                f"{compliance_rate:.0f}%",
                delta=f"{breach_count} breaches" if breach_count > 0 else "Fully compliant",
                delta_color="inverse" if breach_count > 0 else "normal"
            )
            
            # Concentration metrics
            concentration = st.session_state.get("concentration_analysis", {})
            kpi_cols[4].metric(
                "Top 10 Weight",
                f"{concentration.get('top_10_weight', 0):.1f}%",
                help="Combined weight of top 10 holdings"
            )
            kpi_cols[5].metric(
                "HHI Score",
                f"{concentration.get('stock_hhi', 0):.0f}",
                help="Herfindahl-Hirschman Index (lower = more diversified)"
            )
            
            st.markdown("---")
            
            # Visual Dashboard
            dash_col1, dash_col2 = st.columns(2)
            
            with dash_col1:
                # Top 15 holdings
                top_15 = results_df.nlargest(15, 'Weight %')
                others_weight = results_df.nsmallest(len(results_df) - 15, 'Weight %')['Weight %'].sum() if len(results_df) > 15 else 0
                
                plot_data = pd.concat([
                    top_15[['Name', 'Weight %']],
                    pd.DataFrame([{'Name': 'Others', 'Weight %': others_weight}])
                ]) if len(results_df) > 15 else top_15[['Name', 'Weight %']]
                
                fig_pie = px.pie(
                    plot_data,
                    values='Weight %',
                    names='Name',
                    title='Portfolio Composition (Top 15)',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label') # Added text info
                fig_pie.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with dash_col2:
                # Sector allocation
                sector_data = results_df.groupby('Industry')['Weight %'].sum().reset_index().sort_values('Weight %', ascending=False).head(10)
                
                fig_sector = px.bar(
                    sector_data,
                    y='Industry',
                    x='Weight %',
                    orientation='h',
                    title='Top 10 Sector Allocations',
                    color='Weight %',
                    color_continuous_scale='Blues'
                )
                fig_sector.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_sector, use_container_width=True)
            
            # Concentration heatmap
            st.markdown("#### 🗺️ Portfolio Concentration Heatmap")
            
            if 'Market Cap' in results_df.columns:
                pivot_data = results_df.pivot_table(
                    values='Weight %',
                    index='Industry',
                    columns='Market Cap',
                    aggfunc='sum',
                    fill_value=0
                )
                
                fig_heatmap = px.imshow(
                    pivot_data,
                    labels=dict(x="Market Cap", y="Sector", color="Weight %"),
                    title="Allocation Matrix: Sector × Market Cap",
                    color_continuous_scale='RdYlGn_r',
                    aspect="auto"
                )
                fig_heatmap.update_layout(height=max(400, len(pivot_data) * 30))
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # TAB 2: Detailed Analytics
        with analysis_tabs[1]:
            st.markdown("### 🔍 Detailed Portfolio Analytics")
            
            detail_subtabs = st.tabs(["Holdings", "Sectors", "Ratings", "Market Cap", "Geographic"])
            
            with detail_subtabs[0]:
                st.markdown("#### Top 20 Holdings Analysis")
                
                top_20 = results_df.nlargest(20, 'Weight %')
                
                fig = px.bar(
                    top_20,
                    y='Name',
                    x='Weight %',
                    orientation='h',
                    color='Industry',
                    title='Top 20 Holdings by Weight',
                    hover_data=['Symbol', 'Real-time Value (Rs)', 'LTP']
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(
                    top_20[['Name', 'Symbol', 'Industry', 'Weight %', 'Real-time Value (Rs)', 'LTP']].style.format({
                        'Weight %': '{:.2f}%',
                        'Real-time Value (Rs)': '₹{:,.2f}',
                        'LTP': '₹{:,.2f}'
                    }),
                    use_container_width=True
                )
            
            with detail_subtabs[1]:
                st.markdown("#### Sector-wise Detailed Analysis")
                
                sector_analysis = results_df.groupby('Industry').agg({
                    'Weight %': 'sum',
                    'Real-time Value (Rs)': 'sum',
                    'Symbol': 'count'
                }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                
                sector_analysis['Avg Weight per Stock'] = (sector_analysis['Weight %'] / sector_analysis['Count']).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        sector_analysis.reset_index().head(15),
                        y='Industry',
                        x='Weight %',
                        orientation='h',
                        title='Sector Allocation by Weight',
                        color='Weight %',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        sector_analysis.reset_index(),
                        x='Count',
                        y='Weight %',
                        size='Real-time Value (Rs)',
                        color='Industry',
                        title='Sector Diversification Matrix',
                        hover_name='Industry'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(
                    sector_analysis.style.format({
                        'Weight %': '{:.2f}%',
                        'Real-time Value (Rs)': '₹{:,.2f}',
                        'Avg Weight per Stock': '{:.2f}%'
                    }),
                    use_container_width=True
                )
            
            with detail_subtabs[2]:
                if 'Rating' in results_df.columns and not results_df['Rating'].eq('UNKNOWN').all():
                    st.markdown("#### Credit Rating Distribution Analysis")
                    
                    rating_analysis = results_df.groupby('Rating').agg({
                        'Weight %': 'sum',
                        'Symbol': 'count',
                        'Real-time Value (Rs)': 'sum'
                    }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            rating_analysis.reset_index(),
                            values='Weight %',
                            names='Rating',
                            title='Rating Distribution by Weight',
                            hole=0.3
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label') # Added text info
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            rating_analysis.reset_index(),
                            y='Rating',
                            x='Weight %',
                            orientation='h',
                            title='Rating Exposure Detail',
                            color='Weight %',
                            color_continuous_scale='RdYlGn_r'
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(
                        rating_analysis.style.format({
                            'Weight %': '{:.2f}%',
                            'Real-time Value (Rs)': '₹{:,.2f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("Rating information not available or all ratings are 'UNKNOWN' in portfolio data")
            
            with detail_subtabs[3]:
                if 'Market Cap' in results_df.columns and not results_df['Market Cap'].eq('UNKNOWN').all():
                    st.markdown("#### Market Capitalization Distribution Analysis")
                    market_cap_analysis = results_df.groupby('Market Cap').agg({
                        'Weight %': 'sum',
                        'Symbol': 'count',
                        'Real-time Value (Rs)': 'sum'
                    }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.pie(
                            market_cap_analysis.reset_index(),
                            values='Weight %',
                            names='Market Cap',
                            title='Market Cap Distribution by Weight',
                            hole=0.3
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label') # Added text info
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            market_cap_analysis.reset_index(),
                            y='Market Cap',
                            x='Weight %',
                            orientation='h',
                            title='Market Cap Exposure Detail',
                            color='Weight %',
                            color_continuous_scale='Plasma'
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(
                        market_cap_analysis.style.format({
                            'Weight %': '{:.2f}%',
                            'Real-time Value (Rs)': '₹{:,.2f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("Market Cap information not available or all market caps are 'UNKNOWN' in portfolio data")
            
            with detail_subtabs[4]:
                if 'Country' in results_df.columns and results_df['Country'].nunique() > 1 and not results_df['Country'].eq('UNKNOWN').all():
                    st.markdown("#### Geographic Exposure Analysis")
                    geo_analysis = results_df.groupby('Country').agg({
                        'Weight %': 'sum',
                        'Symbol': 'count',
                        'Real-time Value (Rs)': 'sum'
                    }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.pie(
                            geo_analysis.reset_index(),
                            values='Weight %',
                            names='Country',
                            title='Geographic Distribution by Weight',
                            hole=0.3
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label') # Added text info
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            geo_analysis.reset_index(),
                            y='Country',
                            x='Weight %',
                            orientation='h',
                            title='Geographic Exposure Detail',
                            color='Weight %',
                            color_continuous_scale='cividis'
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(
                        geo_analysis.style.format({
                            'Weight %': '{:.2f}%',
                            'Real-time Value (Rs)': '₹{:,.2f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("Geographic information not available, only single country, or all countries are 'UNKNOWN' in portfolio data")
        
        # TAB 3: Risk Metrics
        with analysis_tabs[2]:
            st.markdown("### 📈 Portfolio Risk Metrics")
            
            advanced_metrics = st.session_state.get("advanced_metrics")
            
            if advanced_metrics is None: # Only show button if metrics haven't been calculated or were cleared
                if st.button("Calculate Advanced Risk Metrics", key="calc_adv_risk", type="secondary", use_container_width=True):
                    if kite_client and access_token:
                        advanced_metrics_data = calculate_advanced_metrics(results_df, api_key, access_token)
                        if advanced_metrics_data:
                            st.session_state.advanced_metrics = advanced_metrics_data
                            st.success("✅ Advanced risk metrics calculated!")
                            # Re-fetch advanced_metrics after calculation
                            advanced_metrics = st.session_state.get("advanced_metrics") 
                        else:
                            st.warning("⚠️ Could not calculate advanced risk metrics due to insufficient data.")
                    else:
                        st.warning("🔐 Please connect Kite for live data to calculate advanced risk metrics.")
            
            if advanced_metrics:
                st.markdown("#### VaR & Expected Shortfall (CVaR)")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("VaR (95%)", f"{(advanced_metrics.get('var_95', 0) * 100):.2f}%" if not np.isnan(advanced_metrics.get('var_95', np.nan)) else "N/A")
                col2.metric("VaR (99%)", f"{(advanced_metrics.get('var_99', 0) * 100):.2f}%" if not np.isnan(advanced_metrics.get('var_99', np.nan)) else "N/A")
                col3.metric("VaR (90%)", f"{(advanced_metrics.get('var_90', 0) * 100):.2f}%" if not np.isnan(advanced_metrics.get('var_90', np.nan)) else "N/A")
                col4.metric("CVaR (95%)", f"{(advanced_metrics.get('cvar_95', 0) * 100):.2f}%" if not np.isnan(advanced_metrics.get('cvar_95', np.nan)) else "N/A")
                col5.metric("CVaR (99%)", f"{(advanced_metrics.get('cvar_99', 0) * 100):.2f}%" if not np.isnan(advanced_metrics.get('cvar_99', np.nan)) else "N/A")
                
                st.markdown("#### Modern Portfolio Theory Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Sharpe Ratio", f"{advanced_metrics.get('sharpe_ratio', np.nan):.2f}" if not np.isnan(advanced_metrics.get('sharpe_ratio', np.nan)) else "N/A")
                col2.metric("Sortino Ratio", f"{advanced_metrics.get('sortino_ratio', np.nan):.2f}" if not np.isnan(advanced_metrics.get('sortino_ratio', np.nan)) else "N/A")
                col3.metric("Beta", f"{advanced_metrics.get('beta', np.nan):.2f}" if not np.isnan(advanced_metrics.get('beta', np.nan)) else "N/A")
                col4.metric("Alpha (Annualized)", f"{(advanced_metrics.get('alpha', np.nan) * 100):.2f}%" if not np.isnan(advanced_metrics.get('alpha', np.nan)) else "N/A")
                col5.metric("Tracking Error", f"{(advanced_metrics.get('tracking_error', np.nan) * 100):.2f}%" if not np.isnan(advanced_metrics.get('tracking_error', np.nan)) else "N/A")
                
                col6, col7, col8, col9, col10 = st.columns(5)
                col6.metric("Information Ratio", f"{advanced_metrics.get('information_ratio', np.nan):.2f}" if not np.isnan(advanced_metrics.get('information_ratio', np.nan)) else "N/A")
                col7.metric("Treynor Ratio", f"{(advanced_metrics.get('treynor_ratio', np.nan) * 100):.2f}%" if not np.isnan(advanced_metrics.get('treynor_ratio', np.nan)) else "N/A")
                col8.metric("Calmar Ratio", f"{advanced_metrics.get('calmar_ratio', np.nan):.2f}" if not np.isnan(advanced_metrics.get('calmar_ratio', np.nan)) else "N/A")
                col9.metric("Max Drawdown", f"{(advanced_metrics.get('max_drawdown', np.nan) * 100):.2f}%" if not np.isnan(advanced_metrics.get('max_drawdown', np.nan)) else "N/A")
                col10.metric("Ulcer Index", f"{(advanced_metrics.get('ulcer_index', np.nan) * 100):.2f}%" if not np.isnan(advanced_metrics.get('ulcer_index', np.nan)) else "N/A")

                st.markdown("#### Statistical & Distribution Metrics")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Portfolio Volatility (Annualized)", f"{(advanced_metrics.get('portfolio_volatility', np.nan) * 100):.2f}%" if not np.isnan(advanced_metrics.get('portfolio_volatility', np.nan)) else "N/A")
                col2.metric("Skewness", f"{advanced_metrics.get('skewness', np.nan):.2f}" if not np.isnan(advanced_metrics.get('skewness', np.nan)) else "N/A")
                col3.metric("Kurtosis", f"{advanced_metrics.get('kurtosis', np.nan):.2f}" if not np.isnan(advanced_metrics.get('kurtosis', np.nan)) else "N/A")
                col4.metric("Max Daily Gain", f"{(advanced_metrics.get('max_daily_gain', np.nan) * 100):.2f}%" if not np.isnan(advanced_metrics.get('max_daily_gain', np.nan)) else "N/A")
                col5.metric("Max Daily Loss", f"{(advanced_metrics.get('max_daily_loss', np.nan) * 100):.2f}%" if not np.isnan(advanced_metrics.get('max_daily_loss', np.nan)) else "N/A")

                st.markdown("#### Correlation Analysis")
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Pairwise Correlation", f"{advanced_metrics.get('avg_correlation', np.nan):.2f}" if not np.isnan(advanced_metrics.get('avg_correlation', np.nan)) else "N/A")
                col2.metric("Maximum Pairwise Correlation", f"{advanced_metrics.get('max_correlation', np.nan):.2f}" if not np.isnan(advanced_metrics.get('max_correlation', np.nan)) else "N/A")
                col3.metric("Minimum Pairwise Correlation", f"{advanced_metrics.get('min_correlation', np.nan):.2f}" if not np.isnan(advanced_metrics.get('min_correlation', np.nan)) else "N/A")
                
                # Plot returns distribution only if portfolio_returns is available and not empty
                if advanced_metrics.get("portfolio_returns") is not None and not advanced_metrics["portfolio_returns"].empty:
                    st.markdown("##### Returns Distribution")
                    fig_dist = px.histogram(
                        advanced_metrics['portfolio_returns'] * 100,
                        nbins=50,
                        title="Portfolio Daily Returns Distribution",
                        labels={'value': 'Daily Return (%)'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("#### Stress Testing Scenarios")
                # Ensure portfolio_returns is a Series before passing to stress testing
                portfolio_returns_series = advanced_metrics.get("portfolio_returns")
                total_value = results_df['Real-time Value (Rs)'].sum()
                weights = (results_df['Real-time Value (Rs)'] / total_value).values if total_value > 0 else np.array([])
                
                if portfolio_returns_series is not None and not portfolio_returns_series.empty and total_value > 0:
                    stress_test_results = perform_stress_testing(results_df, portfolio_returns_series, weights)
                    
                    if stress_test_results:
                        st.subheader("Market Crash Scenarios (Impact on Portfolio Value)")
                        scenario_data = {
                            "Scenario": list(stress_test_results['market_crash'].keys()),
                            "Impact (%)": list(stress_test_results['market_crash'].values())
                        }
                        scenario_df = pd.DataFrame(scenario_data)
                        scenario_df['Portfolio Value Impact (Rs)'] = (total_value * scenario_df['Impact (%)'] / 100).round(2)
                        
                        fig_stress = px.bar(
                            scenario_df,
                            x="Scenario",
                            y="Impact (%)",
                            color="Impact (%)",
                            color_continuous_scale=px.colors.sequential.Reds_r,
                            title="Hypothetical Market Crash Impact on Portfolio"
                        )
                        st.plotly_chart(fig_stress, use_container_width=True)
                        
                        st.dataframe(scenario_df.style.format({
                            'Impact (%)': '{:.2f}%',
                            'Portfolio Value Impact (Rs)': '₹{:,.2f}'
                        }), use_container_width=True)
                        
                        st.subheader("Historical Worst Case Scenarios")
                        hist_scenario_df = pd.DataFrame([
                            {"Period": "Worst Day", "Impact (%)": stress_test_results['historical'].get('worst_day', np.nan)},
                            {"Period": "Worst Week", "Impact (%)": stress_test_results['historical'].get('worst_week', np.nan)},
                            {"Period": "Worst Month", "Impact (%)": stress_test_results['historical'].get('worst_month', np.nan)}
                        ])
                        hist_scenario_df['Portfolio Value Impact (Rs)'] = (total_value * hist_scenario_df['Impact (%)'] / 100).round(2)
                        st.dataframe(hist_scenario_df.style.format({
                            'Impact (%)': '{:.2f}%',
                            'Portfolio Value Impact (Rs)': '₹{:,.2f}'
                        }), use_container_width=True)
                        
                        st.subheader("Sector-Specific Shocks")
                        sector_shock_data = []
                        for sector, data in stress_test_results['sector_shocks'].items():
                            sector_shock_data.append({
                                "Sector": sector,
                                "Sector Weight (%)": data['weight'],
                                "10% Sector Shock Impact (%)": data['impact_10'],
                                "20% Sector Shock Impact (%)": data['impact_20']
                            })
                        sector_shock_df = pd.DataFrame(sector_shock_data)
                        
                        st.dataframe(sector_shock_df.style.format({
                            'Sector Weight (%)': '{:.2f}%',
                            '10% Sector Shock Impact (%)': '{:.2f}%',
                            '20% Sector Shock Impact (%)': '{:.2f}%'
                        }), use_container_width=True)
                        
                        st.subheader("Interest Rate Scenarios")
                        for scenario, description in stress_test_results['interest_rate'].items():
                            st.write(f"**{scenario.replace('_', ' ').title()}:** {description}")
                    else:
                        st.info("Stress testing results could not be generated.")
                else:
                    st.info("Insufficient portfolio data or returns history for stress testing.")
            else:
                st.info("Click 'Calculate Advanced Risk Metrics' to view detailed risk analysis.")

        # TAB 4: Rule Validation
        with analysis_tabs[3]:
            st.markdown("### ⚖️ Custom Rule Validation Results")
            
            # Re-run rule validation if needed. This now uses the `rules_text` from the configuration.
            if st.button("Re-run Custom Rule Validation", key="rerun_custom_rules", use_container_width=True):
                custom_rule_results = parse_and_validate_rules_enhanced(st.session_state.get("compliance_rules", ""), results_df)
                st.session_state.custom_rule_results = custom_rule_results
            
            custom_rule_results = st.session_state.get("custom_rule_results", [])
            
            if not custom_rule_results:
                st.info("No custom rules defined or analysis not yet run. Define rules in the configuration section and click 'Analyze Portfolio'.")
            else:
                # Display overall compliance status
                fail_count = sum(1 for r in custom_rule_results if '❌ FAIL' in r['status']) # Check for '❌ FAIL'
                total_rules = len(custom_rule_results)
                
                if fail_count == 0:
                    st.success(f"✅ All {total_rules} custom rules passed!")
                else:
                    st.warning(f"⚠️ {fail_count} out of {total_rules} custom rules failed!")
                
                # Create DataFrame for display
                custom_rules_df = pd.DataFrame(custom_rule_results)
                
                # Style breaches for better visibility
                def color_status_text(val):
                    if 'FAIL' in val:
                        return 'background-color: #ffebee'
                    elif 'Error' in val:
                        return 'background-color: #fff3e0'
                    elif 'Invalid' in val:
                        return 'background-color: #e3f2fd'
                    else:
                        return ''
                
                st.dataframe(
                    custom_rules_df[['rule_type', 'rule', 'status', 'details', 'severity']].style.applymap(color_status_text, subset=['status']),
                    use_container_width=True
                )
        
        # TAB 5: Security Compliance
        with analysis_tabs[4]:
            st.markdown("### 🔐 Security-Level Compliance Overview")
            
            security_compliance_df = st.session_state.get("security_level_compliance")
            
            if security_compliance_df is None or security_compliance_df.empty: # Check if DataFrame is empty or None
                st.info("Security-level compliance data is not available. Please upload a portfolio and run the analysis.")
            else:
                # Overall compliance distribution
                compliance_counts = security_compliance_df['Overall Status'].value_counts()
                
                fig_overall_status = px.pie(
                    names=compliance_counts.index,
                    values=compliance_counts.values,
                    title='Overall Security Compliance Status Distribution',
                    hole=0.4,
                    color='Overall Status', # Map color based on status
                    color_discrete_map={
                        '🟢 Excellent': 'green',
                        '🟡 Good': 'lightgreen',
                        '🟠 Fair': 'orange',
                        '🔴 Poor': 'red'
                    }
                )
                fig_overall_status.update_traces(textposition='inside', textinfo='percent+label') # Added text info
                st.plotly_chart(fig_overall_status, use_container_width=True)
                
                st.markdown("#### Detailed Security Compliance Table")
                
                # Display selected columns from security_compliance_df
                # Ensure only columns that actually exist are included to prevent KeyError
                display_cols = [
                    'Name', 'Symbol', 'Weight %', 'Stock Limit Status', 'Stock Limit Gap (%)',
                    'Stock Limit Utilization (%)', 'Liquidity Status', 'Liquidity Score',
                    'Rating', 'Rating Category', 'Rating Compliance',
                    'Concentration Risk', 'Compliance Score', 'Overall Status'
                ]
                
                display_cols_filtered = [col for col in display_cols if col in security_compliance_df.columns]
                
                st.dataframe(
                    security_compliance_df[display_cols_filtered].style.format({
                        'Weight %': '{:.2f}%',
                        'Stock Limit Gap (%)': '{:.2f}%',
                        'Stock Limit Utilization (%)': '{:.2f}%',
                        'Liquidity Score': '{:.0f}',
                        'Compliance Score': '{:.0f}'
                    }).apply(
                        lambda x: ['background-color: #ffebee' if '❌ Breach' in str(val) else 
                                   'background-color: #fff3e0' if '⚠️ Low' in str(val) or '🟠 High' in str(val) or 'Below Threshold' in str(val) else 
                                   'background-color: #fffde7' if '🟡 Medium' in str(val) else '' for val in x],
                        subset=[col for col in ['Stock Limit Status', 'Liquidity Status', 'Rating Compliance', 'Concentration Risk'] if col in security_compliance_df.columns]
                    ).apply(
                        lambda x: ['background-color: #e8f5e9' if '🟢 Excellent' in str(val) else 
                                   'background-color: #fffde7' if '🟡 Good' in str(val) else 
                                   'background-color: #fff3e0' if '🟠 Fair' in str(val) else 
                                   'background-color: #ffebee' if '🔴 Poor' in str(val) else '' for val in x],
                        subset=[col for col in ['Overall Status'] if col in security_compliance_df.columns]
                    ),
                    use_container_width=True
                )

        # TAB 6: Concentration Analysis
        with analysis_tabs[5]:
            st.markdown("### 📉 Concentration Risk Analysis")
            
            concentration_metrics = st.session_state.get("concentration_analysis")
            
            if concentration_metrics and any(not np.isnan(v) for v in concentration_metrics.values() if isinstance(v, (int, float))): # Check if metrics are not empty/NaN
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Herfindahl-Hirschman Index (HHI)")
                    st.metric(
                        "Stock HHI",
                        f"{concentration_metrics.get('stock_hhi', 0):.0f}",
                        help=concentration_metrics.get('stock_hhi_description', '')
                    )
                    st.markdown(f"**Status:** {concentration_metrics.get('stock_hhi_category')}")
                    
                    st.metric(
                        "Sector HHI",
                        f"{concentration_metrics.get('sector_hhi', 0):.0f}",
                        help=concentration_metrics.get('sector_hhi_description', '')
                    )
                    st.markdown(f"**Status:** {concentration_metrics.get('sector_hhi_category')}")
                
                with col2:
                    st.markdown("#### Diversification Metrics")
                    st.metric("Effective Number of Stocks", f"{concentration_metrics.get('effective_n_stocks', 0):.1f}" if not np.isnan(concentration_metrics.get('effective_n_stocks', np.nan)) else "N/A")
                    st.metric("Effective Number of Sectors", f"{concentration_metrics.get('effective_n_sectors', 0):.1f}" if not np.isnan(concentration_metrics.get('effective_n_sectors', np.nan)) else "N/A")
                    st.metric("Gini Coefficient", f"{concentration_metrics.get('gini_coefficient', 0):.2f}" if not np.isnan(concentration_metrics.get('gini_coefficient', np.nan)) else "N/A")
                
                st.markdown("#### Top Holdings Concentration")
                top_concentration_data = {
                    "Metric": ["Top 1 Holding", "Top 3 Holdings", "Top 5 Holdings", "Top 10 Holdings", "Top 20 Holdings"],
                    "Weight (%)": [
                        concentration_metrics.get('top_1_weight', 0),
                        concentration_metrics.get('top_3_weight', 0),
                        concentration_metrics.get('top_5_weight', 0),
                        concentration_metrics.get('top_10_weight', 0),
                        concentration_metrics.get('top_20_weight', 0)
                    ]
                }
                top_concentration_df = pd.DataFrame(top_concentration_data)
                
                fig_top_concentration = px.bar(
                    top_concentration_df,
                    x="Metric",
                    y="Weight (%)",
                    title="Portfolio Top Holdings Concentration",
                    color="Weight (%)",
                    color_continuous_scale=px.colors.sequential.YlOrRd
                )
                st.plotly_chart(fig_top_concentration, use_container_width=True)
                
                st.dataframe(top_concentration_df.style.format({'Weight (%)': '{:.2f}%'}), use_container_width=True)
                
                st.markdown("#### Top Sector Concentration")
                sector_concentration_data = {
                    "Metric": ["Top Sector", "Top 3 Sectors"],
                    "Weight (%)": [
                        concentration_metrics.get('top_sector_weight', 0),
                        concentration_metrics.get('top_3_sectors_weight', 0)
                    ]
                }
                sector_concentration_df = pd.DataFrame(sector_concentration_data)
                
                fig_top_sector_concentration = px.bar(
                    sector_concentration_df,
                    x="Metric",
                    y="Weight (%)",
                    title="Portfolio Top Sector Concentration",
                    color="Weight (%)",
                    color_continuous_scale=px.colors.sequential.Oranges
                )
                st.plotly_chart(fig_top_sector_concentration, use_container_width=True)
                
                st.dataframe(sector_concentration_df.style.format({'Weight (%)': '{:.2f}%'}), use_container_width=True)
            else:
                st.info("Concentration analysis results not available. Please upload a portfolio and run the analysis.")

        # TAB 7: Liquidity Analysis
        with analysis_tabs[6]:
            st.markdown("### 💧 Liquidity Analysis")
            
            liquidity_metrics = calculate_liquidity_metrics(results_df) # Recalculate or use cached
            
            if 'error' not in liquidity_metrics:
                st.markdown("#### Overall Portfolio Liquidity")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Weighted Avg Volume (90d)", f"{liquidity_metrics.get('weighted_avg_volume', 0):,.0f}")
                col2.metric("Portfolio Liquidity Score", f"{liquidity_metrics.get('portfolio_liquidity_score', 0):.1f}")
                col3.metric("High Liquidity Assets", f"{liquidity_metrics.get('high_liquidity_pct', 0):.1f}%")
                col4.metric("Low Liquidity Assets", f"{liquidity_metrics.get('low_liquidity_pct', 0):.1f}%")
                
                liquidity_data = pd.DataFrame({
                    'Category': ['High Liquidity', 'Medium Liquidity', 'Low Liquidity'],
                    'Weight %': [
                        liquidity_metrics.get('high_liquidity_pct', 0),
                        liquidity_metrics.get('medium_liquidity_pct', 0),
                        liquidity_metrics.get('low_liquidity_pct', 0)
                    ]
                })
                
                fig_liquidity_pie = px.pie(
                    liquidity_data,
                    values='Weight %',
                    names='Category',
                    title='Portfolio Liquidity Distribution',
                    hole=0.3,
                    color='Category',
                    color_discrete_map={
                        'High Liquidity': 'green',
                        'Medium Liquidity': 'orange',
                        'Low Liquidity': 'red'
                    }
                )
                fig_liquidity_pie.update_traces(textposition='inside', textinfo='percent+label') # Added text info
                st.plotly_chart(fig_liquidity_pie, use_container_width=True)
                
                st.markdown("#### Detailed Security Liquidity (Top 20 by Low Liquidity)")
                
                if 'Avg Volume (90d)' in results_df.columns:
                    low_liquidity_stocks = results_df.nsmallest(20, 'Avg Volume (90d)')[
                        ['Symbol', 'Name', 'Weight %', 'Avg Volume (90d)', 'Real-time Value (Rs)']
                    ]
                    
                    if not low_liquidity_stocks.empty:
                        st.dataframe(
                            low_liquidity_stocks.style.format({
                                'Weight %': '{:.2f}%',
                                'Avg Volume (90d)': '{:,.0f}',
                                'Real-time Value (Rs)': '₹{:,.2f}'
                            }),
                            use_container_width=True
                        )
                    else:
                        st.info("No low liquidity stocks identified or sufficient volume data.")
                else:
                    st.info("Average Volume (90d) data not available for detailed liquidity analysis.")
            else:
                st.info("Liquidity analysis data not available. Please ensure 'Avg Volume (90d)' is in your portfolio upload.")


        # TAB 8: Export Report
        with analysis_tabs[7]:
            st.markdown("### 📄 Export Comprehensive Compliance Report")
            st.info("Generate and download a detailed report of your portfolio's compliance status and analytics.")
            
            report_format = st.selectbox("Select Report Format", ["PDF", "CSV (Raw Data)", "Markdown"])
            
            if st.button("Download Report", type="primary", use_container_width=True):
                report_content = ""
                file_name = f"InvsionConnect_Compliance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if report_format == "CSV (Raw Data)":
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Portfolio Data as CSV",
                        data=csv_data,
                        file_name=f"{file_name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    if not st.session_state.get("security_level_compliance", pd.DataFrame()).empty:
                        csv_sec_comp = st.session_state["security_level_compliance"].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Security Compliance Data as CSV",
                            data=csv_sec_comp,
                            file_name=f"{file_name}_SecurityCompliance.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    st.success("CSV files generated!")
                
                elif report_format == "Markdown" or report_format == "PDF": # Prepare markdown for both
                    report_content += f"# Invsion Connect Pro - Portfolio Compliance Report\n\n"
                    report_content += f"**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    report_content += "## 📊 Executive Summary\n\n"
                    
                    total_value = results_df['Real-time Value (Rs)'].sum()
                    breach_count = len(st.session_state.get("breach_alerts", []))
                    compliance_rate = ((1 - breach_count / max(len(results_df), 1)) * 100) if breach_count < len(results_df) else 0

                    report_content += f"- **Total Portfolio Value:** ₹{total_value:,.2f}\n"
                    report_content += f"- **Number of Holdings:** {len(results_df)}\n"
                    report_content += f"- **Compliance Status:** {compliance_rate:.0f}% compliant ({breach_count} breach(es))\n\n"

                    if st.session_state.get("breach_alerts"):
                        report_content += "### 🚨 Compliance Breach Alerts\n\n"
                        for breach in st.session_state["breach_alerts"]:
                            report_content += f"- **{breach['severity']} - {breach['type']}**: {breach['details']}. _Recommendation: {breach.get('recommendation', 'Review portfolio allocation')}_\n"
                        report_content += "\n"

                    report_content += "## 🔍 Detailed Analytics\n\n"
                    report_content += "### Portfolio Holdings (Top 20)\n\n"
                    report_content += results_df.nlargest(20, 'Weight %')[['Name', 'Symbol', 'Industry', 'Weight %', 'Real-time Value (Rs)']].to_markdown(index=False) + "\n\n"

                    report_content += "### Sector-wise Analysis\n\n"
                    sector_analysis = results_df.groupby('Industry').agg({
                        'Weight %': 'sum',
                        'Real-time Value (Rs)': 'sum',
                        'Symbol': 'count'
                    }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                    report_content += sector_analysis.head(15).to_markdown() + "\n\n"
                    
                    if 'Rating' in results_df.columns and not results_df['Rating'].eq('UNKNOWN').all():
                        report_content += "### Credit Rating Distribution\n\n"
                        rating_analysis = results_df.groupby('Rating').agg({
                            'Weight %': 'sum',
                            'Real-time Value (Rs)': 'sum'
                        }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                        report_content += rating_analysis.to_markdown() + "\n\n"

                    report_content += "## 📈 Risk Metrics\n\n"
                    advanced_metrics = st.session_state.get("advanced_metrics")
                    if advanced_metrics:
                        report_content += "### Key Risk Metrics\n\n"
                        report_content += f"- **VaR (95%):** {(advanced_metrics.get('var_95', 0) * 100):.2f}%\n" if not np.isnan(advanced_metrics.get('var_95', np.nan)) else "- **VaR (95%):** N/A\n"
                        report_content += f"- **Sharpe Ratio:** {advanced_metrics.get('sharpe_ratio', 0):.2f}\n" if not np.isnan(advanced_metrics.get('sharpe_ratio', np.nan)) else "- **Sharpe Ratio:** N/A\n"
                        report_content += f"- **Beta:** {advanced_metrics.get('beta', 0):.2f}\n" if not np.isnan(advanced_metrics.get('beta', np.nan)) else "- **Beta:** N/A\n"
                        report_content += f"- **Annualized Volatility:** {(advanced_metrics.get('portfolio_volatility', 0) * 100):.2f}%\n" if not np.isnan(advanced_metrics.get('portfolio_volatility', np.nan)) else "- **Annualized Volatility:** N/A\n"
                        report_content += f"- **Max Drawdown:** {(advanced_metrics.get('max_drawdown', 0) * 100):.2f}%\n\n" if not np.isnan(advanced_metrics.get('max_drawdown', np.nan)) else "- **Max Drawdown:** N/A\n\n"
                    else:
                        report_content += "Risk metrics not calculated or available.\n\n"
                    
                    concentration_metrics = st.session_state.get("concentration_analysis")
                    if concentration_metrics and any(not np.isnan(v) for v in concentration_metrics.values() if isinstance(v, (int, float))):
                        report_content += "### Concentration Analysis\n\n"
                        report_content += f"- **Stock HHI:** {concentration_metrics.get('stock_hhi', 0):.0f} ({concentration_metrics.get('stock_hhi_category')})\n"
                        report_content += f"- **Sector HHI:** {concentration_metrics.get('sector_hhi', 0):.0f} ({concentration_metrics.get('sector_hhi_category')})\n"
                        report_content += f"- **Top 10 Holdings Weight:** {concentration_metrics.get('top_10_weight', 0):.2f}%\n\n"
                    else:
                        report_content += "Concentration analysis not available.\n\n"


                    report_content += "## ⚖️ Custom Rule Validation\n\n"
                    custom_rule_results = st.session_state.get("custom_rule_results", [])
                    if custom_rule_results:
                        custom_rules_df = pd.DataFrame(custom_rule_results)
                        report_content += custom_rules_df[['rule_type', 'rule', 'status', 'details', 'severity']].to_markdown(index=False) + "\n\n"
                    else:
                        report_content += "No custom rule validation results available.\n\n"
                    
                    report_content += "--- Generated by Invsion Connect Pro ---\n"

                    if report_format == "Markdown":
                        st.download_button(
                            label="Download Markdown Report",
                            data=report_content.encode('utf-8'),
                            file_name=f"{file_name}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    elif report_format == "PDF":
                        # For PDF generation, you'd typically use a library like ReportLab or wkhtmltopdf
                        # Streamlit doesn't have a direct PDF generation tool built-in.
                        # As a workaround, you can convert markdown to PDF using external tools or services.
                        # For simplicity, this example will offer a markdown download and mention external PDF conversion.
                        st.warning("Direct PDF generation is complex and often requires external libraries or services.")
                        st.download_button(
                            label="Download Markdown for PDF Conversion",
                            data=report_content.encode('utf-8'),
                            file_name=f"{file_name}.md",
                            mime="text/markdown",
                            help="Download as Markdown and use a tool like Pandoc or an online converter to generate PDF.",
                            use_container_width=True
                        )
                        st.info("💡 Tip: You can copy the Markdown content into an online Markdown to PDF converter (e.g., Dillinger, StackEdit) for a formatted PDF.")
                    
                st.success("Report generation complete!")

# --- Render Tabs (Corrected Structure) ---
# Each tab's content must be within its respective 'with' block.
# This was the primary cause of the merging issue.

with tab_market:
    render_market_historical_tab(k, KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

with tab_compliance:
    render_investment_compliance_tab(k, KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

with tab_ai:
    # AI Analysis tab implementation
    st.header("🤖 AI-Powered Compliance Analysis")
    st.markdown("Advanced AI analysis using Google Gemini for comprehensive compliance insights")
    
    portfolio_df = st.session_state.get("compliance_results_df")
    
    if portfolio_df is None or portfolio_df.empty:
        st.warning("⚠️ Please upload and analyze a portfolio in the 'Investment Compliance Pro' tab first")
    else:
        st.info("💡 Upload scheme documents (SID/KIM) for AI-powered compliance analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "📄 Upload Scheme Documents",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Upload SID, KIM, or investment policy documents"
            )
        
        with col2:
            st.markdown("**Analysis Configuration**")
            analysis_depth = st.select_slider(
                "Depth",
                options=["Quick", "Standard", "Comprehensive"],
                value="Standard"
            )
            include_recommendations = st.checkbox("Recommendations", value=True)
            include_risk = st.checkbox("Risk Assessment", value=True)
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} document(s) uploaded")
            
            if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 AI analyzing portfolio..."):
                    try:
                        # Extract document text
                        docs_text = ""
                        for file in uploaded_files:
                            if file.type == "application/pdf":
                                with fitz.open(stream=file.getvalue(), filetype="pdf") as doc:
                                    for page in doc:
                                        docs_text += page.get_text()
                            else:
                                docs_text += file.getvalue().decode("utf-8")
                        
                        # Prepare portfolio summary
                        total_value = portfolio_df['Real-time Value (Rs)'].sum()
                        top_10 = portfolio_df.nlargest(10, 'Weight %')[['Name', 'Weight %']]
                        sector_weights = portfolio_df.groupby('Industry')['Weight %'].sum().nlargest(10)
                        
                        portfolio_summary = f"""
Portfolio Overview:
- Total Value: ₹{total_value:,.2f}
- Holdings: {len(portfolio_df)}
- Top Stock: {portfolio_df.nlargest(1, 'Weight %')['Name'].values[0]} ({portfolio_df['Weight %'].max():.2f}%)
- Top 10 Weight: {portfolio_df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%

Top 10 Holdings:
{top_10.to_string()}

Sector Allocation:
{sector_weights.to_string()}
"""
                        
                        # Build prompt
                        prompt = f"""You are an expert investment compliance analyst for Indian Asset Management Companies with deep knowledge of SEBI regulations.

**PORTFOLIO DATA:**
{portfolio_summary}

**SCHEME DOCUMENTS:**
{docs_text[:100000]}

**TASK:**
Perform comprehensive compliance analysis covering:

1. **Executive Summary** - Overall compliance status and critical findings
2. **Investment Objective Alignment** - Strategy vs implementation analysis
3. **SEBI Regulatory Compliance** - Single issuer, sector, group limits
4. **Scheme-Specific Restrictions** - Document-based validation
5. **Risk Assessment** - Concentration, liquidity, credit risks
6. **Violations & Concerns** - Severity-classified issues
7. **Recommendations** - Actionable remediation steps

Provide specific, quantitative analysis with document citations. Use markdown formatting with clear sections."""

                        # Call Gemini
                        model = genai.GenerativeModel('gemini-2.0-flash-exp')
                        response = model.generate_content(
                            prompt,
                            generation_config={
                                'temperature': 0.3,
                                'top_p': 0.8,
                                'max_output_tokens': 8192,
                            }
                        )
                        
                        st.session_state.ai_analysis_response = response.text
                        st.success("✅ AI Analysis Complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
        
        # Display results
        if st.session_state.get("ai_analysis_response"):
            st.markdown("---")
            st.markdown("## 📊 AI Compliance Analysis Report")
            st.markdown(st.session_state.ai_analysis_response)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                txt_data = st.session_state.ai_analysis_response.encode('utf-8')
                st.download_button(
                    "📄 Download Text",
                    txt_data,
                    f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain",
                    use_container_width=True
                )
            
            with col2:
                md_data = st.session_state.ai_analysis_response.encode('utf-8')
                st.download_button(
                    "📝 Download Markdown",
                    md_data,
                    f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "text/markdown",
                    use_container_width=True
                )
            
            with col3:
                if st.button("🗑️ Clear Analysis", use_container_width=True):
                    st.session_state.ai_analysis_response = None
                    st.rerun()


# --- Footer ---
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; padding: 30px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; margin-top: 30px;'>
    <h3 style='color: #1e3c72; margin-bottom: 10px;'>Invsion Connect Pro</h3>
    <p style='font-size: 1.1em; margin-bottom: 15px;'><strong>Enterprise Portfolio Compliance & Risk Analytics Platform</strong></p>
    <p style='font-size: 0.9em; color: #555;'>⚠️ This tool provides informational analysis only. Always consult qualified professionals for investment decisions.</p>
    <p style='font-size: 0.85em; color: #777; margin-top: 15px;'>
        Powered by <strong>KiteConnect API</strong> & <strong>Google Gemini AI</strong><br>
        © 2025 Invsion Connect | All Rights Reserved
    </p>
    <div style='margin-top: 20px; padding-top: 15px; border-top: 1px solid #ddd;'>
        <p style='font-size: 0.8em; color: #888;'>
            📊 Real-time Market Data | 🤖 AI-Powered Analysis | 🔒 Secure & Compliant<br>
            Built for Asset Management Companies, Fund Managers & Compliance Teams
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
