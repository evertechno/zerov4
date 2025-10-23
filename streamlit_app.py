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
import fitz  # PyMuPDF for reading PDFs
from supabase import create_client, Client
import hashlib

# --- AI Imports ---
try:
    import google.generativeai as genai
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
    page_title="Invsion Connect - Portfolio Analysis", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Global Constants ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
BENCHMARK_SYMBOL = "NIFTY 50"

# --- Initialize Session State ---
if "supabase_user" not in st.session_state:
    st.session_state["supabase_user"] = None
if "kite_access_token" not in st.session_state:
    st.session_state["kite_access_token"] = None
if "kite_login_response" not in st.session_state:
    st.session_state["kite_login_response"] = None
if "instruments_df" not in st.session_state:
    st.session_state["instruments_df"] = pd.DataFrame()
if "compliance_results_df" not in st.session_state:
    st.session_state["compliance_results_df"] = pd.DataFrame()
if "advanced_metrics" not in st.session_state:
    st.session_state["advanced_metrics"] = None
if "ai_analysis_response" not in st.session_state:
    st.session_state["ai_analysis_response"] = None
if "security_level_compliance" not in st.session_state:
    st.session_state["security_level_compliance"] = pd.DataFrame()
if "breach_alerts" not in st.session_state:
    st.session_state["breach_alerts"] = []
if "current_portfolio_id" not in st.session_state:
    st.session_state["current_portfolio_id"] = None
if "current_analysis_id" not in st.session_state:
    st.session_state["current_analysis_id"] = None

# --- Load Credentials ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    gemini_conf = secrets.get("google_gemini", {})
    supabase_conf = secrets.get("supabase", {})
    
    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")
    if not gemini_conf.get("api_key"):
        errors.append("Google Gemini API key")
    if not supabase_conf.get("url") or not supabase_conf.get("key"):
        errors.append("Supabase credentials (url, key)")
    
    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
        st.info("Ensure your `secrets.toml` includes [kite], [google_gemini], and [supabase] sections.")
        st.stop()
    return kite_conf, gemini_conf, supabase_conf

KITE_CREDENTIALS, GEMINI_CREDENTIALS, SUPABASE_CREDENTIALS = load_secrets()
genai.configure(api_key=GEMINI_CREDENTIALS["api_key"])

# --- Initialize Supabase Client ---
@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["key"])

supabase: Client = init_supabase()

# --- Authentication Functions ---
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def signup_user(email: str, password: str, full_name: str):
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name
                }
            }
        })
        return response
    except Exception as e:
        st.error(f"Signup failed: {e}")
        return None

def login_user(email: str, password: str):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return response
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None

def logout_user():
    try:
        supabase.auth.sign_out()
        st.session_state.clear()
        st.success("Logged out successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Logout failed: {e}")

def get_current_user():
    try:
        user = supabase.auth.get_user()
        return user
    except:
        return None

# --- Database Functions ---
def save_portfolio(user_id: str, portfolio_name: str, portfolio_df: pd.DataFrame, metadata: dict):
    try:
        portfolio_data = {
            "user_id": user_id,
            "portfolio_name": portfolio_name,
            "holdings_data": portfolio_df.to_json(orient='records'),
            "total_value": float(portfolio_df['Real-time Value (Rs)'].sum()),
            "holdings_count": len(portfolio_df),
            "metadata": json.dumps(metadata),
            "created_at": datetime.now().isoformat()
        }
        
        response = supabase.table("portfolios").insert(portfolio_data).execute()
        return response.data[0]['id'] if response.data else None
    except Exception as e:
        st.error(f"Failed to save portfolio: {e}")
        return None

def save_compliance_config(user_id: str, portfolio_id: str, config: dict):
    try:
        config_data = {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "single_stock_limit": config.get("single_stock_limit"),
            "single_sector_limit": config.get("single_sector_limit"),
            "top_10_limit": config.get("top_10_limit"),
            "min_holdings": config.get("min_holdings"),
            "unrated_limit": config.get("unrated_limit"),
            "custom_rules": config.get("custom_rules", ""),
            "created_at": datetime.now().isoformat()
        }
        
        response = supabase.table("compliance_configs").insert(config_data).execute()
        return response.data[0]['id'] if response.data else None
    except Exception as e:
        st.error(f"Failed to save compliance config: {e}")
        return None

def save_analysis_results(user_id: str, portfolio_id: str, config_id: str, 
                          compliance_results: list, security_compliance: pd.DataFrame,
                          breach_alerts: list, advanced_metrics: dict = None):
    try:
        analysis_data = {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "config_id": config_id,
            "compliance_results": json.dumps(compliance_results),
            "security_compliance": security_compliance.to_json(orient='records') if not security_compliance.empty else None,
            "breach_alerts": json.dumps(breach_alerts),
            "advanced_metrics": json.dumps(advanced_metrics) if advanced_metrics else None,
            "analysis_date": datetime.now().isoformat()
        }
        
        response = supabase.table("analysis_results").insert(analysis_data).execute()
        return response.data[0]['id'] if response.data else None
    except Exception as e:
        st.error(f"Failed to save analysis results: {e}")
        return None

def save_ai_analysis(user_id: str, portfolio_id: str, analysis_text: str, 
                    document_names: list, analysis_config: dict):
    try:
        ai_data = {
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "analysis_text": analysis_text,
            "document_names": json.dumps(document_names),
            "analysis_config": json.dumps(analysis_config),
            "created_at": datetime.now().isoformat()
        }
        
        response = supabase.table("ai_analyses").insert(ai_data).execute()
        return response.data[0]['id'] if response.data else None
    except Exception as e:
        st.error(f"Failed to save AI analysis: {e}")
        return None

def get_user_portfolios(user_id: str):
    try:
        response = supabase.table("portfolios")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        st.error(f"Failed to fetch portfolios: {e}")
        return []

def get_portfolio_analyses(portfolio_id: str):
    try:
        response = supabase.table("analysis_results")\
            .select("*, compliance_configs(*)")\
            .eq("portfolio_id", portfolio_id)\
            .order("analysis_date", desc=True)\
            .execute()
        return response.data
    except Exception as e:
        st.error(f"Failed to fetch analyses: {e}")
        return []

def load_portfolio_by_id(portfolio_id: str):
    try:
        response = supabase.table("portfolios")\
            .select("*")\
            .eq("id", portfolio_id)\
            .single()\
            .execute()
        
        if response.data:
            holdings_json = response.data['holdings_data']
            df = pd.read_json(holdings_json, orient='records')
            return df, response.data
        return pd.DataFrame(), None
    except Exception as e:
        st.error(f"Failed to load portfolio: {e}")
        return pd.DataFrame(), None

# --- KiteConnect Functions ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

def get_authenticated_kite_client(api_key: str, access_token: str) -> KiteConnect:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None

@st.cache_data(ttl=86400, show_spinner="Loading instruments...")
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
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

@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, 
                               from_date: datetime.date, to_date: datetime.date, 
                               interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated."]})
    
    instruments_df = load_instruments_cached(api_key, access_token)
    token = find_instrument_token(instruments_df, symbol, exchange)
    
    if not token and symbol in ["NIFTY BANK", "NIFTYBANK", "BANKNIFTY", BENCHMARK_SYMBOL, "SENSEX"]:
        index_exchange = "NSE" if symbol not in ["SENSEX"] else "BSE"
        instruments_secondary = load_instruments_cached(api_key, access_token, index_exchange)
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

def find_instrument_token(df: pd.DataFrame, tradingsymbol: str, exchange: str = DEFAULT_EXCHANGE):
    if df.empty:
        return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None

# --- Compliance Functions ---
def parse_and_validate_rules_enhanced(rules_text: str, portfolio_df: pd.DataFrame):
    results = []
    if not rules_text.strip() or portfolio_df.empty:
        return results
    
    sector_weights = portfolio_df.groupby('Industry')['Weight %'].sum()
    stock_weights = portfolio_df.set_index('Symbol')['Weight %']
    rating_weights = portfolio_df.groupby('Rating')['Weight %'].sum() if 'Rating' in portfolio_df.columns else pd.Series()
    asset_class_weights = portfolio_df.groupby('Asset Class')['Weight %'].sum() if 'Asset Class' in portfolio_df.columns else pd.Series()
    market_cap_weights = portfolio_df.groupby('Market Cap')['Weight %'].sum() if 'Market Cap' in portfolio_df.columns else pd.Series()
    
    def check_pass(actual, op, threshold):
        if op == '>': return actual > threshold
        if op == '<': return actual < threshold
        if op == '>=': return actual >= threshold
        if op == '<=': return actual <= threshold
        if op == '=': return actual == threshold
        return False
    
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
                    'rule': rule, 
                    'status': 'Error', 
                    'details': 'Invalid format.', 
                    'severity': 'N/A'
                })
                continue
            
            op = parts[-2]
            if op not in ['>', '<', '>=', '<=', '=']:
                results.append({
                    'rule': rule, 
                    'status': 'Error', 
                    'details': f"Invalid operator '{op}'.", 
                    'severity': 'N/A'
                })
                continue
            
            threshold = float(parts[-1].replace('%', ''))
            
            # STOCK level rules
            if rule_type == 'STOCK' and len(parts) == 4:
                symbol = parts[1].upper()
                if symbol in stock_weights.index:
                    actual_value = stock_weights.get(symbol, 0.0)
                    details = f"Actual for {symbol}: {actual_value:.2f}%"
                else:
                    results.append({
                        'rule': rule, 
                        'status': '⚠️ Invalid', 
                        'details': f"Symbol '{symbol}' not found.", 
                        'severity': 'N/A'
                    })
                    continue
            
            # SECTOR level rules
            elif rule_type == 'SECTOR':
                sector_name = ' '.join(parts[1:-2]).upper()
                matching_sector = next((s for s in sector_weights.index if s.upper() == sector_name), None)
                if matching_sector:
                    actual_value = sector_weights.get(matching_sector, 0.0)
                    details = f"Actual for {matching_sector}: {actual_value:.2f}%"
                else:
                    results.append({
                        'rule': rule, 
                        'status': '⚠️ Invalid', 
                        'details': f"Sector '{sector_name}' not found.", 
                        'severity': 'N/A'
                    })
                    continue
            
            # TOP_N_STOCKS rules
            elif rule_type == 'TOP_N_STOCKS' and len(parts) == 4:
                n = int(parts[1])
                actual_value = portfolio_df.nlargest(n, 'Weight %')['Weight %'].sum()
                details = f"Actual weight of top {n} stocks: {actual_value:.2f}%"
            
            # TOP_N_SECTORS rules
            elif rule_type == 'TOP_N_SECTORS' and len(parts) == 4:
                n = int(parts[1])
                actual_value = sector_weights.nlargest(n).sum()
                details = f"Actual weight of top {n} sectors: {actual_value:.2f}%"
            
            # COUNT_STOCKS rules
            elif rule_type == 'COUNT_STOCKS' and len(parts) == 3:
                actual_value = len(portfolio_df)
                details = f"Actual count: {actual_value}"
            
            # COUNT_SECTORS rules
            elif rule_type == 'COUNT_SECTORS' and len(parts) == 3:
                actual_value = portfolio_df['Industry'].nunique()
                details = f"Actual count: {actual_value}"
            
            else:
                results.append({
                    'rule': rule, 
                    'status': 'Error', 
                    'details': 'Unrecognized rule format.', 
                    'severity': 'N/A'
                })
                continue
            
            if actual_value is not None:
                passed = check_pass(actual_value, op, threshold)
                status = "✅ PASS" if passed else "❌ FAIL"
                
                if not passed:
                    breach_magnitude = abs(actual_value - threshold)
                    if breach_magnitude > threshold * 0.2:
                        severity = "🔴 Critical"
                    elif breach_magnitude > threshold * 0.1:
                        severity = "🟠 High"
                    else:
                        severity = "🟡 Medium"
                else:
                    severity = "✅ Compliant"
                
                results.append({
                    'rule': rule,
                    'status': status,
                    'details': f"{details} | Rule: {op} {threshold}",
                    'severity': severity,
                    'actual_value': actual_value,
                    'threshold': threshold,
                    'breach_amount': actual_value - threshold if not passed else 0
                })
        
        except (ValueError, IndexError) as e:
            results.append({
                'rule': rule, 
                'status': 'Error', 
                'details': f"Parse error: {e}", 
                'severity': 'N/A'
            })
    
    return results

def calculate_security_level_compliance(portfolio_df: pd.DataFrame, rules_config: dict):
    if portfolio_df.empty:
        return pd.DataFrame()
    
    security_compliance = portfolio_df.copy()
    
    single_stock_limit = rules_config.get('single_stock_limit', 10.0)
    security_compliance['Stock Limit Breach'] = security_compliance['Weight %'].apply(
        lambda x: '❌ Breach' if x > single_stock_limit else '✅ Compliant'
    )
    security_compliance['Stock Limit Gap (%)'] = single_stock_limit - security_compliance['Weight %']
    
    security_compliance['Concentration Risk'] = security_compliance['Weight %'].apply(
        lambda x: '🔴 High' if x > 8 else '🟡 Medium' if x > 5 else '🟢 Low'
    )
    
    return security_compliance

def calculate_advanced_metrics(portfolio_df, api_key, access_token):
    symbols = portfolio_df['Symbol'].tolist()
    weights = (portfolio_df['Real-time Value (Rs)'] / portfolio_df['Real-time Value (Rs)'].sum()).values
    from_date = datetime.now().date() - timedelta(days=366)
    to_date = datetime.now().date()
    
    returns_df = pd.DataFrame()
    failed_symbols = []
    
    progress_bar = st.progress(0, "Fetching historical data for metrics...")
    
    for i, symbol in enumerate(symbols):
        hist_data = get_historical_data_cached(api_key, access_token, symbol, from_date, to_date, 'day')
        if not hist_data.empty and '_error' not in hist_data.columns:
            returns_df[symbol] = hist_data['close'].pct_change()
        else:
            failed_symbols.append(symbol)
        progress_bar.progress((i + 1) / len(symbols), f"Fetching data for {symbol}...")
    
    if failed_symbols:
        st.warning(f"Could not fetch historical data for: {', '.join(failed_symbols)}")
    
    returns_df.dropna(how='all', inplace=True)
    returns_df.fillna(0, inplace=True)
    
    if returns_df.empty:
        st.error("Not enough historical data to calculate advanced metrics.")
        return None
    
    portfolio_returns = returns_df.dot(weights)
    
    var_95 = portfolio_returns.quantile(0.05)
    var_99 = portfolio_returns.quantile(0.01)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    benchmark_data = get_historical_data_cached(api_key, access_token, BENCHMARK_SYMBOL, from_date, to_date, 'day')
    
    if benchmark_data.empty or '_error' in benchmark_data.columns:
        portfolio_beta = None
        alpha = None
        tracking_error = None
        information_ratio = None
    else:
        benchmark_returns = benchmark_data['close'].pct_change()
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
        aligned_returns.columns = ['portfolio', 'benchmark']
        
        covariance = aligned_returns.cov().iloc[0, 1]
        benchmark_variance = aligned_returns['benchmark'].var()
        portfolio_beta = covariance / benchmark_variance if benchmark_variance > 0 else None
        
        portfolio_annual_return = ((1 + aligned_returns['portfolio'].mean()) ** 252 - 1)
        benchmark_annual_return = ((1 + aligned_returns['benchmark'].mean()) ** 252 - 1)
        risk_free_rate = 0.06
        
        if portfolio_beta:
            alpha = portfolio_annual_return - (risk_free_rate + portfolio_beta * (benchmark_annual_return - risk_free_rate))
        else:
            alpha = None
        
        tracking_diff = aligned_returns['portfolio'] - aligned_returns['benchmark']
        tracking_error = tracking_diff.std() * np.sqrt(252)
        
        if tracking_error and tracking_error > 0:
            information_ratio = (portfolio_annual_return - benchmark_annual_return) / tracking_error
        else:
            information_ratio = None
    
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    portfolio_annual_return = ((1 + portfolio_returns.mean()) ** 252 - 1)
    sortino_ratio = (portfolio_annual_return - 0.06) / downside_std if downside_std > 0 else None
    
    correlation_matrix = returns_df.corr()
    avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
    
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    weighted_vol = np.sum(weights * returns_df.std() * np.sqrt(252))
    diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else None
    
    progress_bar.empty()
    
    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "beta": portfolio_beta,
        "alpha": alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "sortino_ratio": sortino_ratio,
        "avg_correlation": avg_correlation,
        "diversification_ratio": diversification_ratio,
        "portfolio_volatility": portfolio_vol
    }

# --- AI Analysis Functions ---
def extract_text_from_files(uploaded_files):
    full_text = ""
    for file in uploaded_files:
        full_text += f"\n\n--- DOCUMENT: {file.name} ---\n\n"
        if file.type == "application/pdf":
            with fitz.open(stream=file.getvalue(), filetype="pdf") as doc:
                for page in doc:
                    full_text += page.get_text()
        else:
            full_text += file.getvalue().decode("utf-8")
    return full_text

def get_portfolio_summary(df):
    if df.empty:
        return "No portfolio data available."
    
    total_value = df['Real-time Value (Rs)'].sum()
    top_10_stocks = df.nlargest(10, 'Weight %')[['Name', 'Weight %']]
    sector_weights = df.groupby('Industry')['Weight %'].sum().nlargest(10)
    
    summary = f"""**Portfolio Snapshot (as of {datetime.now().strftime('%Y-%m-%d')})**
    
- **Total Value:** ₹ {total_value:,.2f}
- **Number of Holdings:** {len(df)}
- **Top Stock Weight:** {df['Weight %'].max():.2f}%
- **Top 10 Combined Weight:** {df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%

**Top 10 Holdings:**
"""
    for _, row in top_10_stocks.iterrows():
        summary += f"- {row['Name']}: {row['Weight %']:.2f}%\n"
    
    summary += "\n**Top 10 Sector Exposures:**\n"
    for sector, weight in sector_weights.items():
        summary += f"- {sector}: {weight:.2f}%\n"
    
    return summary

# --- Authentication UI ---
def render_auth_page():
    st.title("🔐 Invsion Connect - Login")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login to Your Account")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if login_email and login_password:
                with st.spinner("Logging in..."):
                    response = login_user(login_email, login_password)
                    if response and response.user:
                        st.session_state["supabase_user"] = response.user
                        st.success("Login successful!")
                        st.rerun()
            else:
                st.error("Please enter both email and password.")
    
    with tab2:
        st.subheader("Create New Account")
        signup_name = st.text_input("Full Name", key="signup_name")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")
        
        if st.button("Sign Up", type="primary", use_container_width=True):
            if not all([signup_name, signup_email, signup_password, signup_password_confirm]):
                st.error("Please fill in all fields.")
            elif signup_password != signup_password_confirm:
                st.error("Passwords do not match.")
            elif len(signup_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                with st.spinner("Creating account..."):
                    response = signup_user(signup_email, signup_password, signup_name)
                    if response:
                        st.success("Account created! Please check your email to verify your account, then login.")

# --- Main Application ---
def main():
    # Check authentication
    user = get_current_user()
    
    if not user or not user.user:
        render_auth_page()
        return
    
    st.session_state["supabase_user"] = user.user
    
    # Initialize Kite
    kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
    login_url = kite_unauth_client.login_url()
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 Welcome, {user.user.user_metadata.get('full_name', 'User')}!")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
        
        st.markdown("---")
        st.markdown("### 🔗 Kite Connect")
        
        if not st.session_state["kite_access_token"]:
            st.markdown("Connect to Kite for live market data")
            st.link_button("🔗 Login to Kite", login_url, use_container_width=True)
        
        request_token_param = st.query_params.get("request_token")
        if request_token_param and not st.session_state["kite_access_token"]:
            with st.spinner("Authenticating with Kite..."):
                try:
                    data = kite_unauth_client.generate_session(
                        request_token_param, 
                        api_secret=KITE_CREDENTIALS["api_secret"]
                    )
                    st.session_state["kite_access_token"] = data.get("access_token")
                    st.session_state["kite_login_response"] = data
                    st.sidebar.success("Kite authenticated!")
                    st.query_params.clear()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Kite authentication failed: {e}")
        
        if st.session_state["kite_access_token"]:
            st.success("Kite Connected ✅")
            if st.button("Disconnect Kite", use_container_width=True):
                st.session_state["kite_access_token"] = None
                st.session_state["kite_login_response"] = None
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📁 Your Portfolios")
        
        portfolios = get_user_portfolios(user.user.id)
        if portfolios:
            portfolio_names = [f"{p['portfolio_name']} ({p['created_at'][:10]})" for p in portfolios]
            selected_portfolio_idx = st.selectbox(
                "Load Saved Portfolio",
                range(len(portfolios)),
                format_func=lambda x: portfolio_names[x],
                key="portfolio_selector"
            )
            
            if st.button("📂 Load Selected Portfolio", use_container_width=True):
                selected_portfolio = portfolios[selected_portfolio_idx]
                df, metadata = load_portfolio_by_id(selected_portfolio['id'])
                if not df.empty:
                    st.session_state["compliance_results_df"] = df
                    st.session_state["current_portfolio_id"] = selected_portfolio['id']
                    st.success(f"Loaded: {selected_portfolio['portfolio_name']}")
                    st.rerun()
        else:
            st.info("No saved portfolios yet.")
    
    # Main content
    st.title("📊 Invsion Connect - Portfolio Compliance Platform")
    st.markdown("Advanced portfolio analysis and regulatory compliance validation")
    
    # Tabs
    tabs = st.tabs(["💼 Portfolio Compliance", "🤖 AI Analysis", "📊 Historical Analyses"])
    
    with tabs[0]:
        render_compliance_tab(user.user)
    
    with tabs[1]:
        render_ai_tab(user.user)
    
    with tabs[2]:
        render_history_tab(user.user)

def render_compliance_tab(user):
    st.header("💼 Investment Compliance & Portfolio Analysis")
    
    if not st.session_state.get("kite_access_token"):
        st.warning("⚠️ Please connect to Kite Connect in the sidebar to fetch live prices.")
        return
    
    k = get_authenticated_kite_client(
        KITE_CREDENTIALS["api_key"], 
        st.session_state["kite_access_token"]
    )
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("1. Upload Portfolio")
        portfolio_name = st.text_input(
            "Portfolio Name",
            value=f"Portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}",
            help="Enter a name for this portfolio"
        )
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type="csv",
            help="Required columns: Symbol, Industry, Quantity, Market/Fair Value(Rs. in Lacs)"
        )
        
        st.markdown("##### Compliance Thresholds")
        with st.expander("⚙️ Configure Limits"):
            single_stock_limit = st.number_input("Single Stock Limit (%)", 1.0, 25.0, 10.0, 0.5)
            single_sector_limit = st.number_input("Single Sector Limit (%)", 5.0, 50.0, 25.0, 1.0)
            top_10_limit = st.number_input("Top 10 Holdings Limit (%)", 20.0, 80.0, 50.0, 5.0)
            min_holdings = st.number_input("Minimum Holdings Count", 10, 200, 30, 5)
            unrated_limit = st.number_input("Unrated Securities Limit (%)", 0.0, 30.0, 10.0, 1.0)
    
    with col2:
        st.subheader("2. Custom Compliance Rules")
        rules_text = st.text_area(
            "Define Rules (one per line)",
            height=200,
            value="""# Example Rules
STOCK RELIANCE < 10
SECTOR BANKING < 25
TOP_N_STOCKS 10 <= 50
COUNT_STOCKS >= 30""",
            help="Define custom validation rules"
        )
        
        with st.expander("📖 Rule Syntax Guide"):
            st.markdown("""
            **Available Rules:**
            - `STOCK [Symbol] <op> [Value]%` - Single stock limit
            - `SECTOR [Name] <op> [Value]%` - Sector exposure
            - `TOP_N_STOCKS [N] <op> [Value]%` - Top N concentration
            - `COUNT_STOCKS <op> [Value]` - Holdings count
            - `COUNT_SECTORS <op> [Value]` - Sector count
            
            **Operators:** `>`, `<`, `>=`, `<=`, `=`
            """)
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [str(col).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_') for col in df.columns]
            
            header_map = {
                'isin': 'ISIN',
                'name_of_the_instrument': 'Name',
                'symbol': 'Symbol',
                'industry': 'Industry',
                'quantity': 'Quantity',
                'rating': 'Rating',
                'asset_class': 'Asset Class',
                'market_cap': 'Market Cap',
                'issuer_group': 'Issuer Group',
                'country': 'Country',
                'instrument_type': 'Instrument Type',
                'avg_volume_(90d)': 'Avg Volume (90d)',
                'market_fair_value(rs_in_lacs)': 'Uploaded Value (Lacs)'
            }
            df = df.rename(columns=header_map)
            
            for col in ['Rating', 'Asset Class', 'Industry', 'Market Cap', 'Issuer Group', 'Country', 'Instrument Type']:
                if col in df.columns:
                    df[col] = df[col].fillna('UNKNOWN').str.strip().str.upper()
            
            if st.button("🔍 Analyze Portfolio", type="primary", use_container_width=True):
                with st.spinner("Analyzing portfolio..."):
                    # Fetch live prices
                    symbols = df['Symbol'].unique().tolist()
                    ltp_data = k.ltp([f"{DEFAULT_EXCHANGE}:{s}" for s in symbols])
                    prices = {sym: ltp_data.get(f"{DEFAULT_EXCHANGE}:{sym}", {}).get('last_price') for sym in symbols}
                    
                    df_results = df.copy()
                    df_results['LTP'] = df_results['Symbol'].map(prices)
                    df_results['Real-time Value (Rs)'] = (
                        df_results['LTP'] * pd.to_numeric(df_results['Quantity'], errors='coerce')
                    ).fillna(0)
                    total_value = df_results['Real-time Value (Rs)'].sum()
                    df_results['Weight %'] = (df_results['Real-time Value (Rs)'] / total_value * 100) if total_value > 0 else 0
                    
                    # Run compliance checks
                    rules_config = {
                        'single_stock_limit': single_stock_limit,
                        'single_sector_limit': single_sector_limit,
                        'min_liquidity': 100000
                    }
                    
                    security_compliance = calculate_security_level_compliance(df_results, rules_config)
                    validation_results = parse_and_validate_rules_enhanced(rules_text, df_results)
                    
                    # Generate breach alerts
                    breaches = []
                    if (df_results['Weight %'] > single_stock_limit).any():
                        breach_stocks = df_results[df_results['Weight %'] > single_stock_limit]
                        for _, stock in breach_stocks.iterrows():
                            breaches.append({
                                'type': 'Single Stock Limit',
                                'severity': '🔴 Critical',
                                'details': f"{stock['Symbol']} at {stock['Weight %']:.2f}% (Limit: {single_stock_limit}%)"
                            })
                    
                    sector_weights = df_results.groupby('Industry')['Weight %'].sum()
                    if (sector_weights > single_sector_limit).any():
                        breach_sectors = sector_weights[sector_weights > single_sector_limit]
                        for sector, weight in breach_sectors.items():
                            breaches.append({
                                'type': 'Sector Limit',
                                'severity': '🟠 High',
                                'details': f"{sector} at {weight:.2f}% (Limit: {single_sector_limit}%)"
                            })
                    
                    # Save to session state
                    st.session_state.compliance_results_df = df_results
                    st.session_state.security_level_compliance = security_compliance
                    st.session_state.breach_alerts = breaches
                    
                    # Save to database
                    portfolio_id = save_portfolio(
                        user.id,
                        portfolio_name,
                        df_results,
                        {
                            'total_value': total_value,
                            'holdings_count': len(df_results),
                            'upload_timestamp': datetime.now().isoformat()
                        }
                    )
                    st.session_state["current_portfolio_id"] = portfolio_id
                    
                    compliance_config = {
                        'single_stock_limit': single_stock_limit,
                        'single_sector_limit': single_sector_limit,
                        'top_10_limit': top_10_limit,
                        'min_holdings': min_holdings,
                        'unrated_limit': unrated_limit,
                        'custom_rules': rules_text
                    }
                    
                    config_id = save_compliance_config(user.id, portfolio_id, compliance_config)
                    
                    analysis_id = save_analysis_results(
                        user.id,
                        portfolio_id,
                        config_id,
                        validation_results,
                        security_compliance,
                        breaches
                    )
                    st.session_state["current_analysis_id"] = analysis_id
                    
                    st.success("✅ Analysis complete and saved to database!")
                    if breaches:
                        st.warning(f"⚠️ {len(breaches)} compliance breach(es) detected!")
        
        except Exception as e:
            st.error(f"Failed to process file: {e}")
            st.exception(e)
    
    # Display results
    results_df = st.session_state.get("compliance_results_df", pd.DataFrame())
    
    if not results_df.empty and 'Weight %' in results_df.columns:
        st.markdown("---")
        
        if st.session_state.get("breach_alerts"):
            st.error("🚨 **Compliance Breach Alert**")
            breach_df = pd.DataFrame(st.session_state["breach_alerts"])
            st.dataframe(breach_df, use_container_width=True, hide_index=True)
        
        analysis_tabs = st.tabs([
            "📊 Dashboard",
            "⚖️ Rule Validation",
            "🔐 Security Compliance",
            "📈 Advanced Metrics"
        ])
        
        # Dashboard Tab
        with analysis_tabs[0]:
            st.subheader("Portfolio Dashboard")
            total_value = results_df['Real-time Value (Rs)'].sum()
            
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Portfolio Value", f"₹ {total_value:,.2f}")
            kpi_cols[1].metric("Holdings", f"{len(results_df)}")
            kpi_cols[2].metric("Sectors", f"{results_df['Industry'].nunique()}")
            kpi_cols[3].metric("Top Stock", f"{results_df['Weight %'].max():.2f}%")
            kpi_cols[4].metric("Status", "✅ Pass" if not st.session_state.get("breach_alerts") else f"❌ {len(st.session_state['breach_alerts'])}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_10 = results_df.nlargest(10, 'Weight %')
                fig_pie = px.pie(
                    top_10,
                    values='Weight %',
                    names='Name',
                    title='Top 10 Holdings',
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                sector_data = results_df.groupby('Industry')['Weight %'].sum().reset_index().sort_values('Weight %', ascending=False).head(10)
                fig_sector = px.bar(
                    sector_data,
                    x='Weight %',
                    y='Industry',
                    orientation='h',
                    title='Top 10 Sectors'
                )
                fig_sector.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_sector, use_container_width=True)
        
        # Rule Validation Tab
        with analysis_tabs[1]:
            st.subheader("⚖️ Compliance Rule Validation")
            
            validation_results = parse_and_validate_rules_enhanced(rules_text, results_df)
            
            if validation_results:
                total_rules = len(validation_results)
                passed = sum(1 for r in validation_results if r['status'] == "✅ PASS")
                failed = sum(1 for r in validation_results if r['status'] == "❌ FAIL")
                
                summary_cols = st.columns(3)
                summary_cols[0].metric("Total Rules", total_rules)
                summary_cols[1].metric("✅ Passed", passed)
                summary_cols[2].metric("❌ Failed", failed)
                
                st.markdown("---")
                
                for res in validation_results:
                    if res['status'] == "✅ PASS":
                        with st.expander(f"{res['status']} | `{res['rule']}`", expanded=False):
                            st.success(f"**Status:** {res['status']}")
                            st.write(f"**Details:** {res['details']}")
                    elif res['status'] == "❌ FAIL":
                        with st.expander(f"{res['status']} {res.get('severity', '')} | `{res['rule']}`", expanded=True):
                            st.error(f"**Status:** {res['status']} - {res.get('severity', '')}")
                            st.write(f"**Details:** {res['details']}")
                            if 'actual_value' in res:
                                cols = st.columns(3)
                                cols[0].metric("Actual", f"{res['actual_value']:.2f}%")
                                cols[1].metric("Threshold", f"{res['threshold']:.2f}%")
                                cols[2].metric("Breach", f"{res['breach_amount']:.2f}%")
            else:
                st.info("No validation results available.")
        
        # Security Compliance Tab
        with analysis_tabs[2]:
            st.subheader("🔐 Security-Level Compliance")
            
            security_df = st.session_state.get("security_level_compliance", pd.DataFrame())
            
            if not security_df.empty:
                breach_count = (security_df['Stock Limit Breach'] == '❌ Breach').sum()
                compliant_count = (security_df['Stock Limit Breach'] == '✅ Compliant').sum()
                
                summary_cols = st.columns(4)
                summary_cols[0].metric("Total Securities", len(security_df))
                summary_cols[1].metric("✅ Compliant", compliant_count)
                summary_cols[2].metric("❌ Breaches", breach_count)
                summary_cols[3].metric("Breach Rate", f"{(breach_count/len(security_df)*100):.1f}%")
                
                st.dataframe(
                    security_df[['Name', 'Symbol', 'Industry', 'Weight %', 'Stock Limit Breach', 'Concentration Risk']],
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No security compliance data available.")
        
        # Advanced Metrics Tab
        with analysis_tabs[3]:
            st.subheader("📈 Advanced Risk Analytics")
            
            if st.button("Calculate Advanced Metrics", type="primary"):
                with st.spinner("Calculating metrics... This may take 1-2 minutes."):
                    metrics = calculate_advanced_metrics(
                        results_df,
                        KITE_CREDENTIALS["api_key"],
                        st.session_state["kite_access_token"]
                    )
                    st.session_state.advanced_metrics = metrics
                    
                    # Save metrics to database
                    if st.session_state.get("current_analysis_id"):
                        try:
                            supabase.table("analysis_results")\
                                .update({"advanced_metrics": json.dumps(metrics)})\
                                .eq("id", st.session_state["current_analysis_id"])\
                                .execute()
                        except Exception as e:
                            st.warning(f"Could not save metrics: {e}")
            
            if st.session_state.advanced_metrics:
                metrics = st.session_state.advanced_metrics
                
                st.markdown("#### Risk Metrics")
                risk_cols = st.columns(4)
                risk_cols[0].metric("VaR (95%)", f"{metrics['var_95'] * 100:.2f}%")
                risk_cols[1].metric("VaR (99%)", f"{metrics['var_99'] * 100:.2f}%")
                risk_cols[2].metric("CVaR (95%)", f"{metrics['cvar_95'] * 100:.2f}%")
                risk_cols[3].metric("Volatility", f"{metrics['portfolio_volatility']:.2f}%")
                
                st.markdown("#### Performance Metrics")
                perf_cols = st.columns(4)
                perf_cols[0].metric("Beta", f"{metrics['beta']:.3f}" if metrics['beta'] else "N/A")
                perf_cols[1].metric("Alpha", f"{metrics['alpha'] * 100:.2f}%" if metrics['alpha'] else "N/A")
                perf_cols[2].metric("Sortino", f"{metrics['sortino_ratio']:.3f}" if metrics['sortino_ratio'] else "N/A")
                perf_cols[3].metric("Avg Correlation", f"{metrics['avg_correlation']:.3f}")
            else:
                st.info("Click button above to calculate advanced metrics.")

def render_ai_tab(user):
    st.header("🤖 AI-Powered Compliance Analysis")
    
    portfolio_df = st.session_state.get("compliance_results_df", pd.DataFrame())
    
    if portfolio_df.empty:
        st.warning("⚠️ Please upload and analyze a portfolio first.")
        return
    
    st.info("💡 Upload scheme documents for AI analysis against your portfolio.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "📄 Upload Documents (SID, KIM, etc.)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
    
    with col2:
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Comprehensive"],
            value="Standard"
        )
        include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    if uploaded_files and st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
        with st.spinner("Running AI analysis..."):
            try:
                docs_text = extract_text_from_files(uploaded_files)
                portfolio_summary = get_portfolio_summary(portfolio_df)
                breach_summary = "\n".join([
                    f"- {b['type']}: {b['details']}" 
                    for b in st.session_state.get("breach_alerts", [])
                ]) or "No breaches detected."
                
                if analysis_depth == "Quick":
                    depth_instruction = "Provide concise analysis of critical issues only."
                elif analysis_depth == "Standard":
                    depth_instruction = "Provide balanced analysis of key compliance areas."
                else:
                    depth_instruction = "Provide exhaustive detailed analysis of all aspects."
                
                prompt = f"""You are an expert investment compliance analyst for Indian Asset Management.

**TASK:** Analyze the portfolio against scheme documents and SEBI regulations.

{depth_instruction}

**PORTFOLIO:**
```
{portfolio_summary}
```

**DETECTED ISSUES:**
```
{breach_summary}
```

**DOCUMENTS:**
```
{docs_text[:120000]}
```

Provide structured analysis with:
1. Executive Summary
2. Investment Objective Alignment
3. Regulatory Compliance (SEBI + Scheme-specific)
4. Risk Assessment
5. Specific Violations
{("6. Recommendations" if include_recommendations else "")}

Begin analysis:
"""
                
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.3,
                        'max_output_tokens': 8192
                    }
                )
                
                st.session_state.ai_analysis_response = response.text
                
                # Save to database
                if st.session_state.get("current_portfolio_id"):
                    save_ai_analysis(
                        user.id,
                        st.session_state["current_portfolio_id"],
                        response.text,
                        [f.name for f in uploaded_files],
                        {
                            'depth': analysis_depth,
                            'include_recommendations': include_recommendations
                        }
                    )
                
                st.success("✅ AI Analysis complete and saved!")
            
            except Exception as e:
                st.error(f"AI analysis failed: {e}")
                st.exception(e)
    
    if st.session_state.get("ai_analysis_response"):
        st.markdown("---")
        st.markdown("## 📊 AI Analysis Report")
        st.markdown(st.session_state.ai_analysis_response)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📄 Download Report",
                st.session_state.ai_analysis_response.encode('utf-8'),
                f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
                use_container_width=True
            )
        with col2:
            if st.button("🗑️ Clear Report", use_container_width=True):
                st.session_state.ai_analysis_response = None
                st.rerun()

def render_history_tab(user):
    st.header("📊 Historical Analyses")
    
    portfolios = get_user_portfolios(user.id)
    
    if not portfolios:
        st.info("No saved portfolios yet.")
        return
    
    for portfolio in portfolios:
        with st.expander(f"📁 {portfolio['portfolio_name']} - {portfolio['created_at'][:10]}"):
            st.write(f"**Total Value:** ₹{portfolio['total_value']:,.2f}")
            st.write(f"**Holdings:** {portfolio['holdings_count']}")
            st.write(f"**Created:** {portfolio['created_at']}")
            
            analyses = get_portfolio_analyses(portfolio['id'])
            
            if analyses:
                st.markdown(f"**{len(analyses)} Analysis/Analyses:**")
                for analysis in analyses:
                    st.write(f"- {analysis['analysis_date'][:10]}")
                    if analysis.get('compliance_configs'):
                        config = analysis['compliance_configs']
                        st.write(f"  - Single Stock Limit: {config.get('single_stock_limit')}%")
                        st.write(f"  - Sector Limit: {config.get('single_sector_limit')}%")
            else:
                st.write("No analyses yet.")

if __name__ == "__main__":
    main()
