
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
import ta
import fitz
import hashlib
import requests 
import scipy.stats as stats 
import traceback

# --- AI Imports ---
try:
    import google.generativeai as genai
except ImportError:
    st.error("Google Generative AI library not found. Install: pip install google-generativeai")
    st.stop()
    
# --- KiteConnect Imports ---
try:
    from kiteconnect import KiteConnect
except ImportError:
    st.error("KiteConnect library not found. Install: pip install kiteconnect")
    st.stop()

# --- Supabase Import ---
try:
    from supabase import create_client, Client
except ImportError:
    st.error("Supabase library not found. Install: pip install supabase")
    st.stop()


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect", layout="wide", initial_sidebar_state="expanded")

# --- Global Constants ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
BENCHMARK_NIFTY_SYMBOL = "NIFTY 50" # Use a specific constant for the default benchmark
COMPLIANCE_API_BASE_URL = "https://zeroapiv4.onrender.com/api/v1" 

# --- Standard Advanced Metrics Structure ---
DEFAULT_ADVANCED_METRICS = {
    "annualized_return": 0.0,
    "portfolio_volatility": 0.0,
    "sharpe_ratio": 0.0,
    "var_95": 0.0,
    "var_99": 0.0,
    "cvar_95": 0.0,
    "beta": 0.0,
    "alpha": 0.0,
    "tracking_error": 0.0,
    "information_ratio": 0.0,
    "sortino_ratio": 0.0,
    "avg_correlation": 0.0,
    "diversification_ratio": 0.0
}


# Initialize session state with robust defaults
if "user_authenticated" not in st.session_state: st.session_state["user_authenticated"] = False
if "user_id" not in st.session_state: st.session_state["user_id"] = None
if "user_email" not in st.session_state: st.session_state["user_email"] = None
if "kite_access_token" not in st.session_state: st.session_state["kite_access_token"] = None
if "compliance_results_df" not in st.session_state: st.session_state["compliance_results_df"] = pd.DataFrame()
if "compliance_results" not in st.session_state: st.session_state["compliance_results"] = []

# FIX: Initialize advanced_metrics to a safe dict
if "advanced_metrics" not in st.session_state or st.session_state["advanced_metrics"] is None: 
    st.session_state["advanced_metrics"] = DEFAULT_ADVANCED_METRICS.copy()
# Ensure it is a dictionary structure
elif not isinstance(st.session_state["advanced_metrics"], dict):
    st.session_state["advanced_metrics"] = DEFAULT_ADVANCED_METRICS.copy()


if "ai_analysis_response" not in st.session_state: st.session_state["ai_analysis_response"] = None
if "security_level_compliance" not in st.session_state: st.session_state["security_level_compliance"] = pd.DataFrame()
if "breach_alerts" not in st.session_state: st.session_state["breach_alerts"] = []
if "saved_analyses" not in st.session_state: st.session_state["saved_analyses"] = []
if "current_rules_text" not in st.session_state: st.session_state["current_rules_text"] = ""
if "current_portfolio_id" not in st.session_state: st.session_state["current_portfolio_id"] = None
if "current_portfolio_name" not in st.session_state: st.session_state["current_portfolio_name"] = None
if "kim_documents" not in st.session_state: st.session_state["kim_documents"] = {}
if "compliance_stage" not in st.session_state: st.session_state["compliance_stage"] = "upload"
# Stress Test State
if "stress_summary" not in st.session_state: st.session_state["stress_summary"] = None
if "stressed_df" not in st.session_state: st.session_state["stressed_df"] = None
if "stressed_compliance_results" not in st.session_state: st.session_state["stressed_compliance_results"] = None
if "risk_returns_df" not in st.session_state: st.session_state["risk_returns_df"] = pd.DataFrame()
if "benchmark_returns" not in st.session_state: st.session_state["benchmark_returns"] = pd.Series()
if "current_benchmark_symbol" not in st.session_state: st.session_state["current_benchmark_symbol"] = BENCHMARK_NIFTY_SYMBOL


if "threshold_configs" not in st.session_state: 
    st.session_state["threshold_configs"] = {
        'single_stock_limit': 10.0,
        'single_sector_limit': 25.0,
        'group_exposure_limit': 25.0,
        'top_10_holdings_limit': 50.0,
        'cash_equivalent_min': 0.0,
        'cash_equivalent_max': 10.0,
        'foreign_security_limit': 50.0,
        'derivative_limit': 50.0,
        'unlisted_security_limit': 10.0,
        'min_holdings': 20,
        'max_holdings': 100,
        'min_sectors': 5,
        'max_single_holding': 10.0,
        'liquidity_ratio_min': 0.9
    }


# --- Load Credentials ---
def load_secrets():
    """Safely load credentials from Streamlit secrets."""
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    gemini_conf = secrets.get("google_gemini", {})
    supabase_conf = secrets.get("supabase", {})
    
    errors = []
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials")
    if not gemini_conf.get("api_key"):
        errors.append("Gemini API key")
    if not supabase_conf.get("url") or not supabase_conf.get("key"):
        errors.append("Supabase credentials")

    if errors:
        st.error(f"Missing credentials: {', '.join(errors)}")
        st.stop()
    return kite_conf, gemini_conf, supabase_conf

KITE_CREDENTIALS, GEMINI_CREDENTIALS, SUPABASE_CREDENTIALS = load_secrets()
genai.configure(api_key=GEMINI_CREDENTIALS["api_key"])

# --- Initialize Supabase ---
@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["key"])

supabase = init_supabase()


# --- Authentication Functions ---
def register_user(email: str, password: str):
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        if response.user:
            return True, "Registration successful! Please check your email to verify your account."
        return False, "Registration failed."
    except Exception as e:
        error_message = str(e)
        if "already registered" in error_message.lower():
            return False, "User already exists with this email."
        return False, f"Error: {error_message}"

def login_user(email: str, password: str):
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            st.session_state["user_authenticated"] = True
            st.session_state["user_id"] = response.user.id
            st.session_state["user_email"] = response.user.email
            st.session_state["supabase_session"] = response.session
            return True, "Login successful!"
        return False, "Invalid credentials."
    except Exception as e:
        error_message = str(e)
        if "Invalid login credentials" in error_message:
            return False, "Invalid email or password."
        return False, f"Error: {error_message}"

def logout_user():
    try:
        supabase.auth.sign_out()
    except:
        pass
    st.session_state.clear()


# --- Enhanced Database Functions ---
def save_kim_document(user_id: str, portfolio_name: str, document_text: str, file_name: str):
    """Save KIM/SID document for a portfolio"""
    try:
        # Check if document already exists
        existing = supabase.table('kim_documents').select('*').eq('user_id', user_id).eq('portfolio_name', portfolio_name).execute()
        
        doc_hash = hashlib.md5(document_text.encode()).hexdigest()
        
        doc_data = {
            'user_id': user_id,
            'portfolio_name': portfolio_name,
            'document_text': document_text,
            'file_name': file_name,
            'document_hash': doc_hash,
            'extracted_at': datetime.now().isoformat()
        }
        
        if existing.data:
            # Update existing
            result = supabase.table('kim_documents').update(doc_data).eq('id', existing.data[0]['id']).execute()
        else:
            # Insert new
            result = supabase.table('kim_documents').insert(doc_data).execute()
        
        return True, result.data[0]['id'] if result.data else None
    except Exception as e:
        st.error(f"Error saving KIM document: {str(e)}")
        return False, None

def get_kim_document(user_id: str, portfolio_name: str):
    """Retrieve saved KIM document for a portfolio"""
    try:
        result = supabase.table('kim_documents').select('*').eq('user_id', user_id).eq('portfolio_name', portfolio_name).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        st.error(f"Error fetching KIM document: {str(e)}")
        return None

def save_portfolio_with_stages(user_id: str, portfolio_name: str, portfolio_data: dict, compliance_stage: str):
    """Save portfolio at different stages of analysis"""
    try:
        # Check if portfolio already exists
        existing = supabase.table('portfolios').select('*').eq('user_id', user_id).eq('portfolio_name', portfolio_name).execute()
        
        portfolio_record = {
            'user_id': user_id,
            'portfolio_name': portfolio_name,
            'holdings_data': portfolio_data.get('holdings_data'),
            'total_value': float(portfolio_data.get('total_value', 0)),
            'holdings_count': portfolio_data.get('holdings_count', 0),
            'metadata': portfolio_data.get('metadata', {}),
            'analysis_stage': compliance_stage
        }
        
        if existing.data:
            # Update existing portfolio
            result = supabase.table('portfolios').update(portfolio_record).eq('id', existing.data[0]['id']).execute()
            portfolio_id = existing.data[0]['id']
        else:
            # Create new portfolio
            result = supabase.table('portfolios').insert(portfolio_record).execute()
            portfolio_id = result.data[0]['id'] if result.data else None
        
        return True, portfolio_id
    except Exception as e:
        st.error(f"Error saving portfolio: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return False, None

def save_compliance_analysis(user_id: str, portfolio_id: str, compliance_data: dict):
    """Save compliance analysis results - FIXED for existing schema"""
    try:
        # Save or update compliance config
        config_record = {
            'user_id': user_id,
            'portfolio_id': portfolio_id,
            'single_stock_limit': float(compliance_data.get('threshold_configs', {}).get('single_stock_limit', 10.0)),
            'single_sector_limit': float(compliance_data.get('threshold_configs', {}).get('single_sector_limit', 25.0)),
            'custom_rules': compliance_data.get('custom_rules', ''),
            'threshold_configs': compliance_data.get('threshold_configs', {})
        }
        
        # Check if config exists
        existing_config = supabase.table('compliance_configs').select('*').eq('portfolio_id', portfolio_id).execute()
        
        if existing_config.data:
            config_result = supabase.table('compliance_configs').update(config_record).eq('id', existing_config.data[0]['id']).execute()
            config_id = existing_config.data[0]['id']
        else:
            config_result = supabase.table('compliance_configs').insert(config_record).execute()
            config_id = config_result.data[0]['id'] if config_result.data else None
        
        if not config_id:
            return False, None
        
        # Save or update analysis results - ONLY fields that exist in schema
        analysis_record = {
            'user_id': user_id,
            'portfolio_id': portfolio_id,
            'config_id': config_id,
            'compliance_results': compliance_data.get('compliance_results', []),
            'security_compliance': compliance_data.get('security_compliance'),
            'breach_alerts': compliance_data.get('breach_alerts', [])
        }
        
        # Only add ai_analysis if it exists and is not None
        if compliance_data.get('ai_analysis'):
            analysis_record['ai_analysis'] = compliance_data['ai_analysis']
        
        # Check if analysis exists
        existing_analysis = supabase.table('analysis_results').select('*').eq('portfolio_id', portfolio_id).execute()
        
        if existing_analysis.data:
            analysis_result = supabase.table('analysis_results').update(analysis_record).eq('id', existing_analysis.data[0]['id']).execute()
        else:
            analysis_result = supabase.table('analysis_results').insert(analysis_record).execute()
        
        # Store advanced_metrics and risk_returns_df separately in portfolio metadata
        current_metadata = supabase.table('portfolios').select('metadata').eq('id', portfolio_id).single().execute().data.get('metadata', {})
        
        # Ensure current_metadata is a dictionary
        if not isinstance(current_metadata, dict):
            current_metadata = {}

        # Update metadata fields
        current_metadata['last_updated'] = datetime.now().isoformat()
        
        if compliance_data.get('advanced_metrics'):
            current_metadata['advanced_metrics'] = compliance_data['advanced_metrics']
            
        if compliance_data.get('risk_returns_df') is not None:
            # Store DataFrame as JSON string
            current_metadata['risk_returns_df'] = compliance_data['risk_returns_df']
            
        if compliance_data.get('benchmark_symbol'):
            current_metadata['benchmark_symbol'] = compliance_data['benchmark_symbol']
        
        portfolio_metadata_update = {'metadata': current_metadata}
        supabase.table('portfolios').update(portfolio_metadata_update).eq('id', portfolio_id).execute()

        # Update portfolio stage
        supabase.table('portfolios').update({'analysis_stage': 'ai_completed' if compliance_data.get('ai_analysis') else 'compliance_done'}).eq('id', portfolio_id).execute()
        
        return True, portfolio_id
    except Exception as e:
        st.error(f"Error saving compliance analysis: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return False, None

def get_user_portfolios(user_id: str):
    """Get all portfolios for a user"""
    try:
        result = supabase.table('portfolios').select(
            '*, analysis_results(*), compliance_configs(*)'
        ).eq('user_id', user_id).order('created_at', desc=True).execute()
        return result.data if result.data else []
    except Exception as e:
        st.error(f"Error fetching portfolios: {str(e)}")
        return []

def load_portfolio_full(portfolio_id: str):
    """Load complete portfolio with all analysis data - FIXED for existing schema"""
    try:
        # Get portfolio
        portfolio_result = supabase.table('portfolios').select('*').eq('id', portfolio_id).execute()
        if not portfolio_result.data:
            return None
        
        portfolio = portfolio_result.data[0]
        
        # Get compliance config
        config_result = supabase.table('compliance_configs').select('*').eq('portfolio_id', portfolio_id).execute()
        
        # Get analysis results
        analysis_result = supabase.table('analysis_results').select('*').eq('portfolio_id', portfolio_id).execute()
        
        # Get KIM document
        kim_result = supabase.table('kim_documents').select('*').eq('user_id', portfolio['user_id']).eq('portfolio_name', portfolio['portfolio_name']).execute()
        
        # Extract advanced_metrics and risk_returns_df from portfolio metadata if available
        advanced_metrics = DEFAULT_ADVANCED_METRICS.copy()
        risk_returns_df = pd.DataFrame()
        benchmark_symbol = BENCHMARK_NIFTY_SYMBOL
        
        metadata = portfolio.get('metadata', {})
        if isinstance(metadata, str): # Handle case where metadata might be stored as a string
             try:
                 metadata = json.loads(metadata)
             except json.JSONDecodeError:
                 metadata = {}
        
        if metadata and isinstance(metadata, dict):
            if metadata.get('advanced_metrics'):
                # Ensure loaded metrics is a dictionary
                advanced_metrics.update(metadata['advanced_metrics'])
            
            if metadata.get('risk_returns_df'):
                try:
                    # Risk returns is stored as JSON string
                    risk_returns_df = pd.read_json(metadata['risk_returns_df'])
                except Exception as e:
                    pass
            
            if metadata.get('benchmark_symbol'):
                benchmark_symbol = metadata['benchmark_symbol']
        
        # Combine data
        combined = {
            'id': portfolio_id,
            'portfolio_name': portfolio.get('portfolio_name'),
            'analysis_date': portfolio.get('created_at'),
            'analysis_stage': portfolio.get('analysis_stage', 'upload'),
            'portfolio_data': portfolio.get('holdings_data'),
            'compliance_rules': config_result.data[0].get('custom_rules') if config_result.data else None,
            'threshold_configs': config_result.data[0].get('threshold_configs', {}) if config_result.data else {},
            'compliance_results': analysis_result.data[0].get('compliance_results') if analysis_result.data else None,
            'security_compliance': analysis_result.data[0].get('security_compliance') if analysis_result.data else None,
            'breach_alerts': analysis_result.data[0].get('breach_alerts') if analysis_result.data else None,
            'advanced_metrics': advanced_metrics,
            'risk_returns_df': risk_returns_df, # Add loaded returns data
            'ai_analysis': analysis_result.data[0].get('ai_analysis') if analysis_result.data else None,
            'kim_document': kim_result.data[0] if kim_result.data else None,
            'metadata': portfolio.get('metadata', {}),
            'benchmark_symbol': benchmark_symbol # Load benchmark symbol
        }
        
        return combined
    except Exception as e:
        st.error(f"Error loading portfolio: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def delete_portfolio(portfolio_id: str):
    """Delete portfolio and all related data"""
    try:
        # Get portfolio to find portfolio_name
        portfolio = supabase.table('portfolios').select('*').eq('id', portfolio_id).execute()
        if portfolio.data:
            portfolio_name = portfolio.data[0]['portfolio_name']
            user_id = portfolio.data[0]['user_id']
            
            # Delete KIM document
            supabase.table('kim_documents').delete().eq('user_id', user_id).eq('portfolio_name', portfolio_name).execute()
        
        # Delete portfolio (cascade will handle related records)
        supabase.table('portfolios').delete().eq('id', portfolio_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting portfolio: {str(e)}")
        return False


# --- KiteConnect Setup ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
login_url = kite_unauth_client.login_url()


# --- Utility Functions ---
def get_authenticated_kite_client(api_key: str, access_token: str):
    if api_key and access_token:
        k = KiteConnect(api_key=api_key)
        k.set_access_token(access_token)
        return k
    return None

@st.cache_data(ttl=60)
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    k = get_authenticated_kite_client(api_key, access_token)
    if not k: return {"_error": "Not authenticated"}
    try:
        return k.ltp([f"{exchange.upper()}:{symbol.upper()}"]).get(f"{exchange.upper()}:{symbol.upper()}")
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date, to_date, interval: str, exchange: str = DEFAULT_EXCHANGE):
    k = get_authenticated_kite_client(api_key, access_token)
    if not k: return pd.DataFrame({"_error": ["Not authenticated"]})
    
    try:
        instruments = k.instruments(exchange)
        df_inst = pd.DataFrame(instruments)
        
        # Normalize the input symbol for matching
        symbol_normalized = symbol.upper().replace(' ', '').replace('-', '')
        
        # Handle index symbols with improved matching
        token_row = pd.DataFrame()
        
        # Try matching NIFTY 50 (most common benchmark)
        if symbol_normalized in ['NIFTY50', 'NIFTY', BENCHMARK_NIFTY_SYMBOL.replace(' ', '')]:
            # The official Kite trading symbol for Nifty 50 on NSE is typically just 'NIFTY'
            token_row = df_inst[
                (df_inst['exchange'] == 'NSE') & 
                (df_inst['tradingsymbol'] == 'NIFTY')
            ]
        
        # Try matching NIFTY MIDCAP 100
        elif 'MIDCAP100' in symbol_normalized or 'MIDCAP' in symbol_normalized:
            # Search for relevant symbols like NIFTYMIDCAP100
            token_row = df_inst[
                (df_inst['exchange'] == 'NSE') & 
                (df_inst['tradingsymbol'].str.contains('MIDCAP', case=False, na=False))
            ]
            
        
        # If still empty, try exact equity symbol matching
        if token_row.empty:
            token_row = df_inst[
                (df_inst['exchange'] == exchange.upper()) & 
                (df_inst['tradingsymbol'] == symbol.upper())
            ]
        
        if token_row.empty:
            return pd.DataFrame({"_error": [f"Token not found for {symbol} on {exchange}"]})
        
        # Get the first matching token
        token = int(token_row.iloc[0]['instrument_token'])
        
        # Kite API expects datetime objects
        start_datetime = datetime.combine(from_date, datetime.min.time())
        end_datetime = datetime.combine(to_date, datetime.max.time())
        
        data = k.historical_data(token, from_date=start_datetime, 
                                to_date=end_datetime, interval=interval)
        df = pd.DataFrame(data)
        
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [str(e)]})


# --- Compliance API Integration Helper ---
def call_compliance_api(endpoint: str, payload: dict):
    """
    Generic helper function to call a compliance API endpoint.
    Returns JSON response data or None on error.
    """
    try:
        url = f"{COMPLIANCE_API_BASE_URL}{endpoint}"
        # st.info(f"Calling API: {url} with payload (truncated): {str(payload)[:500]}...") # Log payload for debug
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        # st.success(f"API call to {endpoint} successful!")
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API HTTP Error ({endpoint}): {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.ConnectionError as e:
        st.error(f"API Connection Error ({endpoint}): {e}")
        return None
    except requests.exceptions.Timeout as e:
        st.error(f"API Timeout Error ({endpoint}): {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An unexpected API Request Error occurred ({endpoint}): {e}")
        return None
    except json.JSONDecodeError:
        try:
            st.error(f"Failed to decode JSON response from API ({endpoint}): {response.text}")
        except:
             st.error(f"Failed to decode JSON response from API ({endpoint}): No response text available.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API call ({endpoint}): {e}")
        return None


# --- Enhanced Compliance Functions (using API) ---
def call_compliance_api_run_check(portfolio_df: pd.DataFrame, rules_text: str, threshold_configs: dict):
    """
    Calls the API to run compliance checks.
    
    Fix: Ensure that DataFrame columns passed to pd.to_numeric are Series
    and handle potential nested list/array elements.
    """
    
    # Ensure a defensive copy
    df_clean = portfolio_df.copy()
    
    for col in ['Quantity', 'LTP', 'Real-time Value (Rs)', 'Weight %']:
        if col in df_clean.columns:
            try:
                # If a column's dtype is 'object', it might contain mixed types or nested lists.
                # Attempt to 'unwrap' if elements are single-item lists/arrays.
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].apply(
                        lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) == 1 else x
                    )
                # Then, convert to numeric.
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
            except Exception as e:
                st.exception(f"Critical data cleaning failure on column {col} before API call: {e}")
                # If conversion fails dramatically, return empty to prevent API call with bad data
                return [] 
                
    payload = {
        "portfolio": df_clean.to_dict('records'),
        "rules_text": rules_text,
        "threshold_configs": threshold_configs
    }
    api_response = call_compliance_api("/simulate/portfolio", payload)
    return api_response['compliance_results'] if api_response and 'compliance_results' in api_response else []


def calculate_security_level_compliance(portfolio_df: pd.DataFrame, threshold_configs: dict):
    """Enhanced security compliance with more thresholds"""
    if portfolio_df.empty:
        return pd.DataFrame()
    
    security_compliance = portfolio_df.copy()
    single_stock_limit = threshold_configs.get('single_stock_limit', 10.0)
    
    security_compliance['Stock Limit Breach'] = security_compliance['Weight %'].apply(
        lambda x: '❌ Breach' if x > single_stock_limit else '✅ Compliant'
    )
    security_compliance['Stock Limit Gap (%)'] = single_stock_limit - security_compliance['Weight %']
    security_compliance['Concentration Risk'] = security_compliance['Weight %'].apply(
        lambda x: '🔴 High' if x > 8 else '🟡 Medium' if x > 5 else '🟢 Low'
    )
    
    # Add liquidity classification (placeholder - would need actual liquidity data)
    security_compliance['Liquidity'] = '🟢 High' # This is a placeholder, as true liquidity data is not integrated
    
    return security_compliance

def calculate_advanced_metrics(portfolio_df, api_key, access_token, lookback_days, benchmark_symbol):
    """Calculate portfolio risk metrics including Beta, Alpha, and Sharpe"""
    
    # Initialize metrics to default structure in case of failure
    metrics = DEFAULT_ADVANCED_METRICS.copy()
    combined_returns = pd.DataFrame()

    # 1. Prepare data fetching
    symbols = portfolio_df['Symbol'].tolist()
    from_date = datetime.now().date() - timedelta(days=lookback_days)
    to_date = datetime.now().date()
    
    returns_df = pd.DataFrame()
    failed_symbols = []
    
    progress_bar = st.progress(0, text="Fetching historical data...")
    
    # 2. Fetch Stock Historical Data
    for i, symbol in enumerate(symbols):
        hist_data = get_historical_data_cached(api_key, access_token, symbol, from_date, to_date, 'day')
        if not hist_data.empty and '_error' not in hist_data.columns:
            returns_df[symbol] = hist_data['close'].pct_change()
        else:
            failed_symbols.append(symbol)
        progress_bar.progress(int(0.4 * (i + 1) / len(symbols) * 100), text=f"Fetching {symbol}...")
    
    if failed_symbols:
        st.warning(f"Failed to fetch historical data for: {len(failed_symbols)} symbols.")
    
    # 3. Fetch Benchmark Data
    progress_bar.progress(40, text=f"Fetching benchmark data ({benchmark_symbol})...")
    benchmark_data = get_historical_data_cached(api_key, access_token, benchmark_symbol, from_date, to_date, 'day')
    
    if benchmark_data.empty or '_error' in benchmark_data.columns:
        error_msg = benchmark_data.iloc[0]['_error'] if '_error' in benchmark_data.columns else "Unknown error."
        st.error(f"Failed to fetch benchmark data for {benchmark_symbol}. {error_msg}. Using zero returns for benchmark.")
        benchmark_returns = pd.Series(0, index=returns_df.index if not returns_df.empty else pd.DatetimeIndex([]))
        risk_free_rate = 0.04 / TRADING_DAYS_PER_YEAR # Proxy
    else:
        benchmark_returns = benchmark_data['close'].pct_change()
        risk_free_rate = 0.04 / TRADING_DAYS_PER_YEAR # Daily Risk-Free Rate (e.g., 4% Annual)

    # 4. Clean and Align Data
    all_returns = pd.concat([returns_df, benchmark_returns.rename('Benchmark')], axis=1)
    all_returns.dropna(how='all', inplace=True)
    all_returns.fillna(0, inplace=True)
    
    if all_returns.empty:
        st.error("Not enough aligned data for metrics.")
        progress_bar.empty()
        return metrics, combined_returns
    
    successful_symbols = returns_df.columns.tolist()
    
    # Ensure portfolio_df_success uses symbols that actually returned data
    portfolio_df_success = portfolio_df.set_index('Symbol').reindex(successful_symbols).reset_index()
    portfolio_df_success['Real-time Value (Rs)'] = pd.to_numeric(portfolio_df_success['Real-time Value (Rs)'], errors='coerce').fillna(0)

    total_value_success = portfolio_df_success['Real-time Value (Rs)'].sum()
    
    if total_value_success == 0:
        progress_bar.empty()
        st.error("Total portfolio value is zero after filtering for successful price fetches or all holdings have zero value.")
        return metrics, combined_returns
    
    weights = (portfolio_df_success['Real-time Value (Rs)'] / total_value_success).values
    portfolio_returns = all_returns[successful_symbols].dot(weights).rename('Portfolio')
    
    # Align portfolio returns with benchmark returns
    combined_returns = pd.concat([portfolio_returns, all_returns['Benchmark']], axis=1).dropna()
    
    # Final check before calculation
    if combined_returns.empty or 'Portfolio' not in combined_returns.columns or 'Benchmark' not in combined_returns.columns:
        progress_bar.empty()
        st.error("Combined returns DataFrame is empty or missing required columns after alignment.")
        return metrics, combined_returns

    # 5. Calculate Metrics
    
    # A. Value at Risk (VaR) & CVaR (Historical Method)
    var_95 = combined_returns['Portfolio'].quantile(0.05)
    var_99 = combined_returns['Portfolio'].quantile(0.01)
    cvar_95 = combined_returns['Portfolio'][combined_returns['Portfolio'] <= var_95].mean()
    
    # B. Portfolio Volatility (Annualized Standard Deviation)
    portfolio_vol = combined_returns['Portfolio'].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # C. Beta (Covariance / Variance of Benchmark)
    if combined_returns['Benchmark'].var() > 1e-6: # Check for non-zero variance
        beta = combined_returns['Portfolio'].cov(combined_returns['Benchmark']) / combined_returns['Benchmark'].var()
    else:
        beta = 0.0
        
    # D. Alpha (Jensen's Alpha) and Tracking Error
    
    # Annualize returns
    annualized_portfolio_return = combined_returns['Portfolio'].mean() * TRADING_DAYS_PER_YEAR
    annualized_benchmark_return = combined_returns['Benchmark'].mean() * TRADING_DAYS_PER_YEAR

    # Calculate Alpha (Jensen's)
    # Alpha = Rp - [Rf + Beta * (Rm - Rf)]
    annualized_risk_free_rate = risk_free_rate * TRADING_DAYS_PER_YEAR
    alpha = annualized_portfolio_return - (annualized_risk_free_rate + beta * (annualized_benchmark_return - annualized_risk_free_rate))
    
    # E. Sharpe Ratio (Annualized)
    sharpe_ratio = (annualized_portfolio_return - annualized_risk_free_rate) / portfolio_vol if portfolio_vol > 1e-6 else 0
    
    # F. Tracking Error (Annualized Standard Deviation of Active Return)
    active_return = combined_returns['Portfolio'] - combined_returns['Benchmark']
    tracking_error = active_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # G. Information Ratio (Alpha / Tracking Error)
    information_ratio = alpha / tracking_error if tracking_error > 1e-6 else 0
    
    # H. Sortino Ratio (Measures return against downside risk)
    downside_returns = combined_returns['Portfolio'][combined_returns['Portfolio'] < 0]
    downside_volatility = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if not downside_returns.empty else 0
    sortino_ratio = (annualized_portfolio_return - annualized_risk_free_rate) / downside_volatility if downside_volatility > 1e-6 else 0
    
    # I. Stock-level correlation and diversification
    corr_matrix = returns_df.corr()
    # Calculate mean correlation of the upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    avg_correlation = corr_matrix.where(mask).stack().mean() if not corr_matrix.empty else 0.0

    # Diversification Ratio (DR = Sum(W*Vol) / Portfolio Vol)
    stock_vols_annual = returns_df.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Ensure indices match
    stock_vols_annual = stock_vols_annual.reindex(successful_symbols)
    weights_series = pd.Series(weights, index=successful_symbols)
    
    weighted_stock_vols = (stock_vols_annual * weights_series).sum()
    diversification_ratio = weighted_stock_vols / portfolio_vol if portfolio_vol > 1e-6 else 0
    
    progress_bar.progress(100, text="Metrics calculated.")
    progress_bar.empty()
    
    metrics = {
        "annualized_return": annualized_portfolio_return,
        "portfolio_volatility": portfolio_vol,
        "sharpe_ratio": sharpe_ratio,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "beta": beta,
        "alpha": alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "sortino_ratio": sortino_ratio,
        "avg_correlation": avg_correlation,
        "diversification_ratio": diversification_ratio
    }
    
    # Store the daily returns data for visualization later
    return metrics, combined_returns.reset_index()

# --- Stress Testing Functions (using local implementation) ---
def run_stress_test(original_df, scenario_type, params):
    """
    Applies a stress scenario to a portfolio DataFrame.
    """
    stressed_df = original_df.copy()
    
    # Ensure 'Real-time Value (Rs)' is numeric before calculations
    stressed_df['Real-time Value (Rs)'] = pd.to_numeric(stressed_df['Real-time Value (Rs)'], errors='coerce').fillna(0.0)
    
    original_total_value = stressed_df['Real-time Value (Rs)'].sum()
    
    # Ensure 'Stressed Value (Rs)' is initialized based on the original value
    stressed_df['Stressed Value (Rs)'] = stressed_df['Real-time Value (Rs)']

    if scenario_type == "Market Crash":
        shock_pct = params['percentage'] / 100.0
        stressed_df['Stressed Value (Rs)'] = stressed_df['Real-time Value (Rs)'] * (1 - shock_pct)

    elif scenario_type == "Sector Shock":
        shock_pct = params['percentage'] / 100.0
        sector = params['sector']
        stressed_df['Stressed Value (Rs)'] = stressed_df.apply(
            lambda row: row['Real-time Value (Rs)'] * (1 - shock_pct) if row['Industry'] == sector else row['Real-time Value (Rs)'],
            axis=1
        )
    
    elif scenario_type == "Single Stock Failure":
        shock_pct = params['percentage'] / 100.0
        symbol = params['symbol']
        stressed_df['Stressed Value (Rs)'] = stressed_df.apply(
            lambda row: row['Real-time Value (Rs)'] * (1 - shock_pct) if row['Symbol'] == symbol else row['Real-time Value (Rs)'],
            axis=1
        )
            
    stressed_total_value = stressed_df['Stressed Value (Rs)'].sum()
    
    # Recalculate weights based on new stressed values
    stressed_df['Stressed Weight %'] = (stressed_df['Stressed Value (Rs)'] / stressed_total_value * 100) if stressed_total_value > 0 else 0
    
    summary = {
        "original_value": original_total_value,
        "stressed_value": stressed_total_value,
        "loss_value": original_total_value - stressed_total_value,
        "loss_pct": ((original_total_value - stressed_total_value) / original_total_value) * 100 if original_total_value > 0 else 0
    }
    
    return stressed_df, summary

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
        return "No portfolio data."
    
    total_value = df['Real-time Value (Rs)'].sum()
    top_10_stocks = df.nlargest(10, 'Weight %')[['Name', 'Weight %']]
    
    # Ensure 'Industry' column exists before grouping
    if 'Industry' in df.columns:
        sector_weights = df.groupby('Industry')['Weight %'].sum().nlargest(10)
    else:
        sector_weights = pd.Series()
    
    summary = f"""**Portfolio Snapshot**

- **Total Value:** ₹ {total_value:,.2f}
- **Holdings:** {len(df)}
- **Top Stock Weight:** {df['Weight %'].max():.2f}%

**Top 10 Holdings:**
"""
    for _, row in top_10_stocks.iterrows():
        summary += f"- {row['Name']}: {row['Weight %']:.2f}%\n"
    
    summary += "\n**Top Sectors:**\n"
    if not sector_weights.empty:
        for sector, weight in sector_weights.items():
            summary += f"- {sector}: {weight:.2f}%\n"
    else:
        summary += "- Sector data unavailable.\n"
    
    return summary


def render_portfolio_card(portfolio):
    """Helper function to render portfolio card in history"""
    portfolio_name = portfolio.get('portfolio_name', 'Unnamed Portfolio')
    analysis_date = datetime.fromisoformat(portfolio['created_at'])
    
    with st.container():
        col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
        
        with col1:
            st.markdown(f"**📁 {portfolio_name}**")
            st.caption(f"{analysis_date.strftime('%Y-%m-%d %H:%M')}")
            
            metadata = portfolio.get('metadata')
            if metadata:
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                
                info_parts = []
                if metadata.get('total_value'):
                    info_parts.append(f"₹{metadata['total_value']:,.0f}")
                if metadata.get('holdings_count'):
                    info_parts.append(f"{metadata['holdings_count']} holdings")
                
                if info_parts:
                    st.caption(" | ".join(info_parts))
        
        with col2:
            # Show stage and breach count
            stage = portfolio.get('analysis_stage', 'upload')
            stage_text = {
                'upload': '📤 Uploaded',
                'compliance_done': '✅ Compliance',
                'ai_completed': '🤖 AI Complete'
            }.get(stage, 'Unknown')
            
            st.caption(stage_text)
            
            # Show breach count if available
            if portfolio.get('analysis_results'):
                results = portfolio['analysis_results']
                if results and len(results) > 0:
                    breach_alerts = results[0].get('breach_alerts', [])
                    if breach_alerts:
                        st.caption(f"❌ {len(breach_alerts)} breaches")
        
        with col3:
            if st.button("📂", key=f"load_hist_{portfolio['id']}", use_container_width=True, help="Load Portfolio"):
                loaded = load_portfolio_full(portfolio['id'])
                if loaded:
                    # Load all data into session state
                    st.session_state["current_portfolio_id"] = portfolio['id']
                    st.session_state["current_portfolio_name"] = loaded['portfolio_name']
                    st.session_state["compliance_stage"] = loaded['analysis_stage']
                    
                    if loaded.get('portfolio_data'):
                        if isinstance(loaded['portfolio_data'], str):
                            st.session_state["compliance_results_df"] = pd.read_json(loaded['portfolio_data'])
                        else:
                            st.session_state["compliance_results_df"] = pd.DataFrame(loaded['portfolio_data'])
                    
                    if loaded.get('threshold_configs'):
                        st.session_state["threshold_configs"] = loaded['threshold_configs']
                    
                    if loaded.get('compliance_rules'):
                        st.session_state["current_rules_text"] = loaded['compliance_rules']
                    
                    if loaded.get('compliance_results'):
                        st.session_state["compliance_results"] = loaded['compliance_results']
                    
                    if loaded.get('security_compliance'):
                        if isinstance(loaded['security_compliance'], str):
                            st.session_state["security_level_compliance"] = pd.read_json(loaded['security_compliance'])
                            # Ensure the index is reset if loading from JSON string
                            st.session_state["security_level_compliance"].reset_index(drop=True, inplace=True)
                        elif loaded['security_compliance']:
                            st.session_state["security_level_compliance"] = pd.DataFrame(loaded['security_compliance'])
                    
                    if loaded.get('breach_alerts'):
                        st.session_state["breach_alerts"] = loaded['breach_alerts']
                    
                    # Ensure advanced_metrics is loaded or defaults used
                    st.session_state["advanced_metrics"] = loaded.get('advanced_metrics', DEFAULT_ADVANCED_METRICS.copy())
                    if st.session_state["advanced_metrics"] is None:
                        st.session_state["advanced_metrics"] = DEFAULT_ADVANCED_METRICS.copy()
                            
                    if loaded.get('risk_returns_df') is not None and not loaded['risk_returns_df'].empty:
                        st.session_state["risk_returns_df"] = loaded['risk_returns_df']
                    else:
                        st.session_state["risk_returns_df"] = pd.DataFrame()

                    # Load benchmark symbol
                    st.session_state["current_benchmark_symbol"] = loaded.get('benchmark_symbol', BENCHMARK_NIFTY_SYMBOL)
                            
                    if loaded.get('ai_analysis'):
                        st.session_state["ai_analysis_response"] = loaded['ai_analysis']
                    
                    if loaded.get('kim_document'):
                        st.session_state["kim_documents"][loaded['portfolio_name']] = loaded['kim_document']

                    # Clear stress test state
                    st.session_state["stress_summary"] = None
                    st.session_state["stressed_df"] = None
                    st.session_state["stressed_compliance_results"] = None
                    
                    st.success("✅ Portfolio Loaded!")
                    time.sleep(0.5)
                    st.rerun()
        
        with col4:
            if st.button("🗑️", key=f"delete_hist_{portfolio['id']}", use_container_width=True):
                if delete_portfolio(portfolio['id']):
                    st.success("Deleted!")
                    st.session_state["saved_analyses"] = get_user_portfolios(st.session_state["user_id"])
                    time.sleep(0.5)
                    st.rerun()
        
        st.markdown("---")


# --- Authentication UI ---
def render_auth_page():
    st.title("🔐 Invsion Connect")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Professional Portfolio Compliance Platform")
        
        auth_tab1, auth_tab2 = st.tabs(["Login", "Register"])
        
        with auth_tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Login", use_container_width=True, type="primary")
                
                if submit_login:
                    if email and password:
                        success, message = login_user(email, password)
                        if success:
                            st.success(message)
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("Enter email and password.")
        
        with auth_tab2:
            with st.form("register_form"):
                reg_email = st.text_input("Email", key="reg_email")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                reg_password_confirm = st.text_input("Confirm Password", type="password")
                submit_register = st.form_submit_button("Register", use_container_width=True, type="primary")
                
                if submit_register:
                    if reg_email and reg_password and reg_password_confirm:
                        if reg_password != reg_password_confirm:
                            st.error("Passwords don't match!")
                        elif len(reg_password) < 6:
                            st.error("Password must be at least 6 characters.")
                        else:
                            success, message = register_user(reg_email, reg_password)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    else:
                        st.warning("Fill all fields.")


# --- Enhanced Threshold Configuration UI ---
def render_threshold_config():
    """Render comprehensive threshold configuration panel"""
    st.subheader("⚙️ Compliance Thresholds")
    
    # Ensure all keys exist before accessing
    # Use DEFAULT_ADVANCED_METRICS as source for defaults for *safety*, though
    # this function is for `threshold_configs`, so use a more appropriate default source.
    # For now, relying on initial st.session_state["threshold_configs"] default.
            
    with st.expander("📊 Basic Limits", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["threshold_configs"]['single_stock_limit'] = st.number_input(
                "Single Stock Limit (%)", 1.0, 25.0, 
                st.session_state["threshold_configs"].get('single_stock_limit', 10.0), 0.5,
                key="config_single_stock_limit", help="Maximum weight for any single stock"
            )
            st.session_state["threshold_configs"]['single_sector_limit'] = st.number_input(
                "Single Sector Limit (%)", 5.0, 50.0,
                st.session_state["threshold_configs"].get('single_sector_limit', 25.0), 1.0,
                key="config_single_sector_limit", help="Maximum weight for any single sector"
            )
            st.session_state["threshold_configs"]['group_exposure_limit'] = st.number_input(
                "Group Exposure Limit (%)", 5.0, 50.0,
                st.session_state["threshold_configs"].get('group_exposure_limit', 25.0), 1.0,
                key="config_group_exposure_limit", help="Maximum exposure to single business group"
            )
        with col2:
            st.session_state["threshold_configs"]['top_10_holdings_limit'] = st.number_input(
                "Top 10 Holdings Limit (%)", 30.0, 80.0,
                st.session_state["threshold_configs"].get('top_10_holdings_limit', 50.0), 5.0,
                key="config_top_10_holdings_limit", help="Maximum weight of top 10 holdings combined"
            )
            st.session_state["threshold_configs"]['max_single_holding'] = st.number_input(
                "Max Single Holding (%)", 1.0, 25.0,
                st.session_state["threshold_configs"].get('max_single_holding', 10.0), 0.5,
                key="config_max_single_holding", help="Absolute maximum for any single position"
            )
    
    with st.expander("💰 Cash & Liquidity"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["threshold_configs"]['cash_equivalent_min'] = st.number_input(
                "Min Cash Equivalent (%)", 0.0, 20.0,
                st.session_state["threshold_configs"].get('cash_equivalent_min', 0.0), 0.5,
                key="config_cash_equivalent_min", help="Minimum cash and cash equivalents"
            )
        with col2:
            st.session_state["threshold_configs"]['cash_equivalent_max'] = st.number_input(
                "Max Cash Equivalent (%)", 0.0, 50.0,
                st.session_state["threshold_configs"].get('cash_equivalent_max', 10.0), 1.0,
                key="config_cash_equivalent_max", help="Maximum cash and cash equivalents"
            )
        
        st.session_state["threshold_configs"]['liquidity_ratio_min'] = st.number_input(
            "Min Liquidity Ratio", 0.0, 1.0,
            st.session_state["threshold_configs"].get('liquidity_ratio_min', 0.9), 0.05,
            key="config_liquidity_ratio_min", help="Minimum portfolio liquidity ratio"
        )
    
    with st.expander("🌐 Special Instruments"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["threshold_configs"]['foreign_security_limit'] = st.number_input(
                "Foreign Securities Limit (%)", 0.0, 100.0,
                st.session_state["threshold_configs"].get('foreign_security_limit', 50.0), 5.0,
                key="config_foreign_security_limit", help="Maximum exposure to foreign securities"
            )
            st.session_state["threshold_configs"]['derivative_limit'] = st.number_input(
                "Derivatives Limit (%)", 0.0, 100.0,
                st.session_state["threshold_configs"].get('derivative_limit', 50.0), 5.0,
                key="config_derivative_limit", help="Maximum derivatives exposure"
            )
        with col2:
            st.session_state["threshold_configs"]['unlisted_security_limit'] = st.number_input(
                "Unlisted Securities Limit (%)", 0.0, 25.0,
                st.session_state["threshold_configs"].get('unlisted_security_limit', 10.0), 1.0,
                key="config_unlisted_security_limit", help="Maximum exposure to unlisted securities"
            )
    
    with st.expander("📈 Portfolio Structure"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["threshold_configs"]['min_holdings'] = st.number_input(
                "Minimum Holdings", 5, 200,
                st.session_state["threshold_configs"].get('min_holdings', 20), 5,
                key="config_min_holdings", help="Minimum number of holdings"
            )
            st.session_state["threshold_configs"]['min_sectors'] = st.number_input(
                "Minimum Sectors", 1, 20,
                st.session_state["threshold_configs"].get('min_sectors', 5), 1,
                key="config_min_sectors", help="Minimum number of sectors"
            )
        with col2:
            st.session_state["threshold_configs"]['max_holdings'] = st.number_input(
                "Maximum Holdings", 20, 500,
                st.session_state["threshold_configs"].get('max_holdings', 100), 10,
                key="config_max_holdings", help="Maximum number of holdings"
            )


# --- MAIN APP ---
if not st.session_state["user_authenticated"]:
    render_auth_page()
    st.stop()

# User authenticated
st.title("Invsion Connect")
st.markdown(f"Welcome, **{st.session_state['user_email']}** 👋")


# --- Sidebar ---
with st.sidebar:
    st.markdown("### User Account")
    st.info(f"**{st.session_state['user_email']}**")
    
    if st.button("🚪 Logout", use_container_width=True):
        logout_user()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Kite Connect")
    
    if not st.session_state["kite_access_token"]:
        st.link_button("🔗 Login to Kite", login_url, use_container_width=True)
    
    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        with st.spinner("Authenticating..."):
            try:
                data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
                st.session_state["kite_access_token"] = data.get("access_token")
                st.success("Kite connected!")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
    
    if st.session_state["kite_access_token"]:
        st.success("Kite Connected ✅")
        if st.button("Disconnect", use_container_width=True):
            st.session_state["kite_access_token"] = None
            st.rerun()
    
    st.markdown("---")
    st.markdown("### My Portfolios")
    
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state["saved_analyses"] = get_user_portfolios(st.session_state["user_id"])
    
    if not st.session_state.get("saved_analyses"):
        st.session_state["saved_analyses"] = get_user_portfolios(st.session_state["user_id"])
    
    if st.session_state["saved_analyses"]:
        st.markdown(f"**{len(st.session_state['saved_analyses'])} portfolios**")
        
        for portfolio in st.session_state["saved_analyses"][:10]:
            portfolio_name = portfolio.get('portfolio_name', 'Unnamed')
            analysis_stage = portfolio.get('analysis_stage', 'upload')
            
            stage_emoji = {
                'upload': '📤',
                'compliance_done': '✅',
                'ai_completed': '🤖'
            }.get(analysis_stage, '📊')
            
            with st.expander(f"{stage_emoji} {portfolio['portfolio_name']}"):
                st.caption(f"Stage: {analysis_stage}")
                
                if st.button(f"Load", key=f"load_{portfolio['id']}", use_container_width=True):
                    loaded = load_portfolio_full(portfolio['id'])
                    if loaded:
                        # Load all data into session state
                        st.session_state["current_portfolio_id"] = portfolio['id']
                        st.session_state["current_portfolio_name"] = loaded['portfolio_name']
                        st.session_state["compliance_stage"] = loaded['analysis_stage']
                        
                        if loaded.get('portfolio_data'):
                            if isinstance(loaded['portfolio_data'], str):
                                st.session_state["compliance_results_df"] = pd.read_json(loaded['portfolio_data'])
                            else:
                                st.session_state["compliance_results_df"] = pd.DataFrame(loaded['portfolio_data'])
                        
                        if loaded.get('threshold_configs'):
                            st.session_state["threshold_configs"] = loaded['threshold_configs']
                        
                        if loaded.get('compliance_rules'):
                            st.session_state["current_rules_text"] = loaded['compliance_rules']
                        
                        if loaded.get('compliance_results'):
                            st.session_state["compliance_results"] = loaded['compliance_results']
                        
                        if loaded.get('security_compliance'):
                            if isinstance(loaded['security_compliance'], str):
                                st.session_state["security_level_compliance"] = pd.read_json(loaded['security_compliance'])
                                # Ensure the index is reset if loading from JSON string
                                st.session_state["security_level_compliance"].reset_index(drop=True, inplace=True)
                            elif loaded['security_compliance']:
                                st.session_state["security_level_compliance"] = pd.DataFrame(loaded['security_compliance'])
                        
                        if loaded.get('breach_alerts'):
                            st.session_state["breach_alerts"] = loaded['breach_alerts']
                        
                        # Ensure advanced_metrics is loaded or defaults used
                        st.session_state["advanced_metrics"] = loaded.get('advanced_metrics', DEFAULT_ADVANCED_METRICS.copy())
                        if st.session_state["advanced_metrics"] is None:
                            st.session_state["advanced_metrics"] = DEFAULT_ADVANCED_METRICS.copy()
                            
                        if loaded.get('risk_returns_df') is not None and not loaded['risk_returns_df'].empty:
                            st.session_state["risk_returns_df"] = loaded['risk_returns_df']
                        else:
                            st.session_state["risk_returns_df"] = pd.DataFrame()

                        # Load benchmark symbol
                        st.session_state["current_benchmark_symbol"] = loaded.get('benchmark_symbol', BENCHMARK_NIFTY_SYMBOL)
                            
                        if loaded.get('ai_analysis'):
                            st.session_state["ai_analysis_response"] = loaded['ai_analysis']
                        
                        if loaded.get('kim_document'):
                            st.session_state["kim_documents"][loaded['portfolio_name']] = loaded['kim_document']

                        # Clear stress test state
                        st.session_state["stress_summary"] = None
                        st.session_state["stressed_df"] = None
                        st.session_state["stressed_compliance_results"] = None
                        
                        st.success("✅ Portfolio Loaded!")
                        time.sleep(0.5)
                        st.rerun()
                
                if st.button(f"Delete", key=f"del_{portfolio['id']}", use_container_width=True):
                    if delete_portfolio(portfolio['id']):
                        st.success("Deleted!")
                        st.session_state["saved_analyses"] = get_user_portfolios(st.session_state["user_id"])
                        time.sleep(0.5)
                        st.rerun()
    else:
        st.info("No portfolios yet")


# --- Main Tabs ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

# Rename Tab 3 to "Risk & Analysis"
tabs = st.tabs(["💼 Portfolio Analysis", "🤖 AI Analysis", "📈 Risk & Analysis", "🔧 API Interactions", "📚 History"])


# --- TAB 1: Enhanced Compliance Analysis (Mostly Unchanged) ---
with tabs[0]:
    st.header("💼 Portfolio Compliance Analysis")
    
    if not k:
        st.warning("⚠️ Connect to Kite first to fetch real-time prices")
    
    # Portfolio Name Selection/Creation
    col1, col2 = st.columns([3, 1])
    with col1:
        portfolio_name = st.text_input(
            "📁 Portfolio Name",
            value=st.session_state.get("current_portfolio_name", ""),
            placeholder="Enter portfolio name (e.g., 'Large Cap Fund Q4 2024')",
            key="portfolio_name_input",
            help="Give your portfolio a unique name for tracking"
        )
    
    with col2:
        if portfolio_name:
            if st.button("💾 New Portfolio", use_container_width=True):
                st.session_state["current_portfolio_name"] = portfolio_name
                st.session_state["current_portfolio_id"] = None
                st.session_state["compliance_stage"] = "upload"
                st.session_state["compliance_results_df"] = pd.DataFrame()
                st.session_state["compliance_results"] = []
                st.session_state["breach_alerts"] = []
                st.session_state["ai_analysis_response"] = None
                st.session_state["advanced_metrics"] = DEFAULT_ADVANCED_METRICS.copy()
                st.session_state["risk_returns_df"] = pd.DataFrame()
                st.session_state["current_benchmark_symbol"] = BENCHMARK_NIFTY_SYMBOL
                # Clear stress test state
                st.session_state["stress_summary"] = None
                st.session_state["stressed_df"] = None
                st.session_state["stressed_compliance_results"] = None
                st.success(f"New portfolio '{portfolio_name}' created!")
                st.rerun()
    
    if not portfolio_name:
        st.info("👆 Enter a portfolio name to begin")
        st.stop()
    
    # Show current stage
    current_stage = st.session_state.get("compliance_stage", "upload")
    stage_info = {
        'upload': ('📤', 'Upload CSV', 'primary'),
        'compliance_done': ('✅', 'Compliance Complete', 'success'),
        'ai_completed': ('🤖', 'AI Analysis Complete', 'success')
    }
    
    stage_emoji, stage_text, stage_color = stage_info.get(current_stage, ('📊', 'In Progress', 'secondary'))
    st.info(f"{stage_emoji} **Current Stage:** {stage_text}")
    
    st.markdown("---")
    
    # Step 1: Upload Portfolio
    with st.container():
        st.subheader("Step 1: Upload Portfolio CSV")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            uploaded_file = st.file_uploader("CSV file with holdings", type="csv", key="portfolio_csv")
            
            if uploaded_file:
                st.success(f"✅ {uploaded_file.name} uploaded")
        
        with col2:
            render_threshold_config()
    
    st.markdown("---")
    
    # Step 2: Define Custom Rules
    with st.container():
        st.subheader("Step 2: Define Custom Compliance Rules")
        
        st.markdown("""
        **Supported Rule Types:**
        - `STOCK <SYMBOL> <op> <value>` - Single stock weight
        - `SECTOR <NAME> <op> <value>` - Sector weight
        - `TOP_N_STOCKS <N> <op> <value>` - Top N stocks concentration
        - `TOP_N_SECTORS <N> <op> <value>` - Top N sectors concentration
        - `BOTTOM_N_STOCKS <N> <op> <value>` - Bottom N stocks concentration
        - `COUNT_STOCKS <op> <value>` - Total holdings count
        - `COUNT_SECTORS <op> <value>` - Total sectors count
        - `AVG_STOCK_WEIGHT <op> <value>` - Average stock weight
        - `MAX_STOCK_WEIGHT <op> <value>` - Maximum stock weight
        - `MIN_STOCK_WEIGHT <op> <value>` - Minimum stock weight
        - `SECTOR_DIVERSITY <SECTOR> <op> <value>` - Stocks per sector
        - `HHI <op> <value>` - Herfindahl-Hirschman Index
        - `GINI <op> <value>` - Gini coefficient
        
        **Operators:** `<`, `>`, `<=`, `>=`, `=`
        """)
        
        default_rules = st.session_state.get("current_rules_text", """# SEBI Compliance Rules
STOCK RELIANCE < 10
STOCK TCS < 10
SECTOR BANKing < 25
SECTOR IT < 25
TOP_N_STOCKS 10 <= 50
TOP_N_SECTORS 3 <= 60
COUNT_STOCKS >= 20
COUNT_SECTORS >= 5
MAX_STOCK_WEIGHT <= 10
AVG_STOCK_WEIGHT <= 5
HHI < 800""")
        
        rules_text = st.text_area(
            "Custom Rules (one per line, # for comments)",
            height=300,
            value=default_rules,
            key="custom_rules_text_area",
            help="Define your compliance rules here"
        )
    
    st.markdown("---")
    
    # Step 3: Analyze
    if uploaded_file and k:
        if st.button("🔍 Analyze Compliance", type="primary", use_container_width=True, key="analyze_btn"):
            with st.spinner("Analyzing portfolio compliance..."):
                try:
                    # Read CSV
                    df = pd.read_csv(uploaded_file)
                    df.columns = [str(col).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_') for col in df.columns]
                    
                    header_map = {
                        'symbol': 'Symbol',
                        'industry': 'Industry',
                        'quantity': 'Quantity',
                        'name_of_the_instrument': 'Name',
                        'market_fair_value(rs_in_lacs)': 'Uploaded Value (Lacs)',
                        'isin': 'ISIN' 
                    }
                    df = df.rename(columns=header_map)
                    
                    required_cols = ['Symbol', 'Quantity']
                    if not all(col in df.columns for col in required_cols):
                         st.error(f"Missing required columns in CSV: {', '.join(required_cols)}")
                         st.stop()

                    # Data Cleaning and Preparation
                    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
                    df = df[df['Quantity'] > 0].reset_index(drop=True)
                    
                    if 'Industry' in df.columns:
                        df['Industry'] = df['Industry'].fillna('UNKNOWN').astype(str).str.strip().str.upper()
                    else:
                        df['Industry'] = 'UNKNOWN'
                        
                    if 'Name' not in df.columns: 
                        df['Name'] = df['Symbol'] 
                    
                    # Fetch real-time prices
                    symbols = df['Symbol'].unique().tolist()
                    ltp_data = k.ltp([f"{DEFAULT_EXCHANGE}:{s}" for s in symbols])
                    prices = {sym: ltp_data.get(f"{DEFAULT_EXCHANGE}:{sym}", {}).get('last_price') for sym in symbols}
                    
                    df_results = df.copy()
                    df_results['LTP'] = df_results['Symbol'].map(prices)
                    df_results['LTP'] = pd.to_numeric(df_results['LTP'], errors='coerce').fillna(0) # Ensure LTP is numeric
                    
                    # Handle missing prices (e.g., if Kite can't find symbol)
                    missing_price_symbols = df_results[df_results['LTP'] == 0]['Symbol'].unique()
                    if missing_price_symbols.size > 0:
                        st.warning(f"Could not fetch real-time price (LTP=0) for {len(missing_price_symbols)} securities. Calculation accuracy may be compromised.")
                    
                    df_results['Real-time Value (Rs)'] = (df_results['LTP'] * df_results['Quantity']).fillna(0)
                    total_value = df_results['Real-time Value (Rs)'].sum()
                    df_results['Weight %'] = (df_results['Real-time Value (Rs)'] / total_value * 100) if total_value > 0 else 0
                    
                    if total_value == 0:
                        st.error("Total portfolio value is zero. Cannot proceed with weight-based analysis.")
                        st.stop()

                    # Call API for custom rules validation
                    compliance_results = call_compliance_api_run_check(df_results, rules_text, st.session_state["threshold_configs"])
                    
                    # Calculate security-level compliance (local function)
                    security_compliance = calculate_security_level_compliance(df_results, st.session_state["threshold_configs"])
                    
                    # Store in session state
                    st.session_state.compliance_results_df = df_results
                    st.session_state.security_level_compliance = security_compliance
                    st.session_state.compliance_results = compliance_results
                    st.session_state.current_rules_text = rules_text
                    st.session_state.current_portfolio_name = portfolio_name
                    
                    # Detect breaches (local detection for display consistency)
                    breaches = []
                    
                    # --- Local Breach Detection ---
                    single_stock_limit = st.session_state["threshold_configs"]['single_stock_limit']
                    if (df_results['Weight %'] > single_stock_limit).any():
                        breach_stocks = df_results[df_results['Weight %'] > single_stock_limit]
                        for _, stock in breach_stocks.iterrows():
                            breaches.append({
                                'type': 'Single Stock Limit (Local Check)',
                                'severity': '🔴 Critical',
                                'details': f"{stock['Symbol']} at {stock['Weight %']:.2f}% (Limit: {single_stock_limit}%)"
                            })
                    
                    # Sector limit breaches
                    sector_weights = df_results.groupby('Industry')['Weight %'].sum()
                    single_sector_limit = st.session_state["threshold_configs"]['single_sector_limit']
                    if (sector_weights > single_sector_limit).any():
                        breach_sectors = sector_weights[sector_weights > single_sector_limit]
                        for sector, weight in breach_sectors.items():
                            breaches.append({
                                'type': 'Sector Limit (Local Check)',
                                'severity': '🟠 High',
                                'details': f"{sector} at {weight:.2f}% (Limit: {single_sector_limit}%)"
                            })
                    
                    # Custom rule failures (from API)
                    for rule_result in compliance_results:
                        if rule_result['status'] == "FAIL":
                            severity = "🟡 Medium" 
                            if abs(rule_result.get('breach_amount', 0)) > rule_result.get('threshold', 0) * 0.2:
                                severity = "🔴 Critical"
                            elif abs(rule_result.get('breach_amount', 0)) > rule_result.get('threshold', 0) * 0.1:
                                severity = "🟠 High"
                            
                            breaches.append({
                                'type': 'Custom Rule Violation (API Check)',
                                'severity': severity,
                                'details': f"{rule_result['rule']} - {rule_result['details']}"
                            })
                    
                    # Portfolio structure checks (local function as they depend on the updated df_results)
                    if len(df_results) < st.session_state["threshold_configs"]['min_holdings']:
                        breaches.append({
                            'type': 'Min Holdings',
                            'severity': '🟡 Medium',
                            'details': f"Only {len(df_results)} holdings (Min: {st.session_state['threshold_configs']['min_holdings']})"
                        })
                    
                    if len(df_results) > st.session_state["threshold_configs"]['max_holdings']:
                        breaches.append({
                            'type': 'Max Holdings',
                            'severity': '🟡 Medium',
                            'details': f"{len(df_results)} holdings (Max: {st.session_state['threshold_configs']['max_holdings']})"
                        })
                    
                    sector_count = df_results['Industry'].nunique()
                    if sector_count < st.session_state["threshold_configs"]['min_sectors']:
                        breaches.append({
                            'type': 'Min Sectors',
                            'severity': '🟠 High',
                            'details': f"Only {sector_count} sectors (Min: {st.session_state['threshold_configs']['min_sectors']})"
                        })
                    
                    st.session_state.breach_alerts = breaches
                    st.session_state.compliance_stage = "compliance_done"
                    
                    # Save to database
                    portfolio_data = {
                        'holdings_data': df_results.to_json(),
                        'total_value': float(total_value),
                        'holdings_count': len(df_results),
                        'metadata': {
                            'total_value': float(total_value),
                            'holdings_count': len(df_results),
                            'analysis_timestamp': datetime.now().isoformat()
                        }
                    }
                    
                    success, portfolio_id = save_portfolio_with_stages(
                        st.session_state["user_id"],
                        portfolio_name,
                        portfolio_data,
                        "compliance_done"
                    )
                    
                    if success:
                        st.session_state["current_portfolio_id"] = portfolio_id
                        
                        # Save compliance analysis
                        compliance_data = {
                            'threshold_configs': st.session_state["threshold_configs"],
                            'custom_rules': rules_text,
                            'compliance_results': compliance_results,
                            'security_compliance': security_compliance.to_json(date_format='iso'),
                            'breach_alerts': breaches,
                            'advanced_metrics': st.session_state.get("advanced_metrics", DEFAULT_ADVANCED_METRICS.copy()),
                            'risk_returns_df': st.session_state.get("risk_returns_df", pd.DataFrame()).to_json(date_format='iso'),
                            'benchmark_symbol': st.session_state.get("current_benchmark_symbol", BENCHMARK_NIFTY_SYMBOL)
                        }
                        
                        save_compliance_analysis(st.session_state["user_id"], portfolio_id, compliance_data)
                        st.success(f"✅ Compliance Analysis Complete! Portfolio saved.")
                        
                        # Refresh portfolio list
                        st.session_state["saved_analyses"] = get_user_portfolios(st.session_state["user_id"])
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("⚠️ Analysis completed but save failed.")
                
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
    
    # Display results
    results_df = st.session_state.get("compliance_results_df", pd.DataFrame())
    
    if not results_df.empty:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        if st.session_state.get("breach_alerts"):
            st.error(f"🚨 **{len(st.session_state['breach_alerts'])} Compliance Breaches Detected**")
            breach_df = pd.DataFrame(st.session_state["breach_alerts"])
            st.dataframe(breach_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ **No compliance breaches detected!**")
        
        # Removed the 'Metrics' tab content as it's moving to 'Risk & Analysis'
        analysis_tabs = st.tabs([
            "📊 Dashboard",
            "🔍 Details",
            "⚖️ Rules",
            "🔐 Security",
            "📊 Concentration",
            "📄 Report"
        ])
        
        with analysis_tabs[0]:
            st.subheader("Portfolio Dashboard")
            total_value = results_df['Real-time Value (Rs)'].sum()
            
            kpi_cols = st.columns(6)
            kpi_cols[0].metric("Value", f"₹ {total_value:,.0f}")
            kpi_cols[1].metric("Holdings", f"{len(results_df)}")
            kpi_cols[2].metric("Sectors", f"{results_df['Industry'].nunique() if 'Industry' in results_df.columns else 'N/A'}")
            kpi_cols[3].metric("Top Stock", f"{results_df['Weight %'].max():.2f}%")
            kpi_cols[4].metric("Top 10", f"{results_df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%")
            kpi_cols[5].metric("Status", "✅" if not st.session_state.get("breach_alerts") else f"❌ {len(st.session_state['breach_alerts'])}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_15 = results_df.nlargest(15, 'Weight %')
                fig_pie = px.pie(top_15, values='Weight %', names='Name', title='Top 15 Holdings', hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                if 'Industry' in results_df.columns:
                    sector_data = results_df.groupby('Industry')['Weight %'].sum().reset_index().sort_values('Weight %', ascending=False).head(10)
                    fig_sector = px.bar(sector_data, x='Weight %', y='Industry', orientation='h',
                                       title='Top 10 Sectors', color='Weight %', color_continuous_scale='Blues')
                    fig_sector.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_sector, use_container_width=True)
                else:
                    st.info("Industry data not available for sector analysis.")

        with analysis_tabs[1]:
            st.subheader("Holdings Details")
            st.dataframe(results_df[['Name', 'Symbol', 'Industry', 'Weight %', 'Real-time Value (Rs)', 'LTP', 'Quantity']].style.format({
                'Weight %': '{:.2f}%',
                'Real-time Value (Rs)': '₹{:,.2f}',
                'LTP': '₹{:,.2f}',
                'Quantity': '{:,.0f}'
            }), use_container_width=True, height=500)
        
        # Note: analysis_tabs[2] is now 'Rules'
        with analysis_tabs[2]:
            st.subheader("Rule Validation Results")
            
            validation_results = st.session_state.get("compliance_results", [])
            
            if validation_results:
                total_rules = len(validation_results)
                # API returns "PASS" or "FAIL"
                passed = sum(1 for r in validation_results if r['status'] == "PASS")
                failed = sum(1 for r in validation_results if r['status'] == "FAIL")
                errors = sum(1 for r in validation_results if r['status'] == 'Error') # assuming API sends 'Error' for parsing issues
                
                summary_cols = st.columns(4)
                summary_cols[0].metric("Total Rules", total_rules)
                summary_cols[1].metric("✅ Passed", passed)
                summary_cols[2].metric("❌ Failed", failed)
                summary_cols[3].metric("⚠️ Errors", errors)
                
                st.markdown("---")
                
                failed_rules = [r for r in validation_results if r['status'] == "FAIL"]
                passed_rules = [r for r in validation_results if r['status'] == "PASS"]
                error_rules = [r for r in validation_results if r['status'] == 'Error']
                
                if failed_rules:
                    st.markdown("### ❌ Failed Rules")
                    for res in failed_rules:
                        # Re-calculate severity based on our logic for display consistency
                        severity = "🟡 Medium" 
                        if abs(res.get('breach_amount', 0)) > res.get('threshold', 0) * 0.2:
                            severity = "🔴 Critical"
                        elif abs(res.get('breach_amount', 0)) > res.get('threshold', 0) * 0.1:
                            severity = "🟠 High"

                        with st.expander(f"❌ FAIL {severity} | `{res['rule']}`", expanded=True):
                            st.error(f"**Status:** FAIL")
                            st.write(f"**Details:** {res['details']}")
                            if 'breach_amount' in res:
                                st.write(f"**Breach Amount:** {res['breach_amount']:.2f}")
                
                if passed_rules:
                    st.markdown("### ✅ Passed Rules")
                    for res in passed_rules:
                        with st.expander(f"✅ PASS | `{res['rule']}`", expanded=False):
                            st.success(f"**Status:** PASS")
                            st.write(f"**Details:** {res['details']}")
                
                if error_rules:
                    st.markdown("### ⚠️ Rule Errors")
                    for res in error_rules:
                        with st.expander(f"Error | `{res['rule']}`", expanded=False):
                            st.warning(f"**Status:** {res['status']}")
                            st.write(f"**Details:** {res['details']}")
            else:
                st.info("No custom rules validated. Add rules and click Analyze.")
        
        # Note: analysis_tabs[3] is now 'Security'
        with analysis_tabs[3]:
            st.subheader("Security-Level Compliance")
            
            security_df = st.session_state.get("security_level_compliance", pd.DataFrame())
            
            if not security_df.empty:
                breach_count = (security_df['Stock Limit Breach'] == '❌ Breach').sum()
                compliant_count = (security_df['Stock Limit Breach'] == '✅ Compliant').sum()
                
                summary_cols = st.columns(3)
                summary_cols[0].metric("Total Securities", len(security_df))
                summary_cols[1].metric("✅ Compliant", compliant_count)
                summary_cols[2].metric("❌ Breaches", breach_count)
                
                st.dataframe(security_df[['Name', 'Symbol', 'Industry', 'Weight %', 'Stock Limit Breach', 'Concentration Risk']].style.format({
                    'Weight %': '{:.2f}%'
                }), use_container_width=True, height=500)
        
        # Note: analysis_tabs[4] is now 'Concentration'
        with analysis_tabs[4]:
            st.subheader("Concentration Analysis")
            
            results_df['Weight %'] = pd.to_numeric(results_df['Weight %'], errors='coerce').fillna(0) # Ensure numeric
            sorted_df = results_df.sort_values('Weight %', ascending=False).reset_index(drop=True)
            sorted_df['Cumulative Weight %'] = sorted_df['Weight %'].cumsum()
            sorted_df['Rank'] = range(1, len(sorted_df) + 1)
            
            if not sorted_df.empty:
                fig_lorenz = go.Figure()
                fig_lorenz.add_trace(go.Scatter(
                    x=sorted_df['Rank'],
                    y=sorted_df['Cumulative Weight %'],
                    mode='lines+markers',
                    name='Portfolio',
                    line=dict(color='blue', width=2)
                ))
                fig_lorenz.add_trace(go.Scatter(
                    x=[0, len(sorted_df)],
                    y=[0, 100],
                    mode='lines',
                    name='Perfect Equality',
                    line=dict(color='red', dash='dash')
                ))
                fig_lorenz.update_layout(
                    title='Concentration Curve',
                    xaxis_title='Holdings Rank',
                    yaxis_title='Cumulative Weight %',
                    height=400
                )
                st.plotly_chart(fig_lorenz, use_container_width=True)
                
                st.markdown("### Concentration Benchmarks")
                bench_cols = st.columns(5)
                bench_cols[0].metric("Top 1", f"{sorted_df.iloc[0]['Weight %']:.2f}%" if not sorted_df.empty else "N/A")
                bench_cols[1].metric("Top 3", f"{sorted_df.head(3)['Weight %'].sum():.2f}%" if len(sorted_df) >= 3 else "N/A")
                bench_cols[2].metric("Top 5", f"{sorted_df.head(5)['Weight %'].sum():.2f}%" if len(sorted_df) >= 5 else "N/A")
                bench_cols[3].metric("Top 10", f"{sorted_df.head(10)['Weight %'].sum():.2f}%" if len(sorted_df) >= 10 else "N/A")
                bench_cols[4].metric("Top 20", f"{sorted_df.head(20)['Weight %'].sum():.2f}%" if len(sorted_df) >= 20 else "N/A")
                
                # HHI and Gini
                hhi = (results_df['Weight %'] ** 2).sum()
                weights_sorted = results_df['Weight %'].sort_values().values
                n = len(weights_sorted)
                gini = 0 # Initialize to 0 for cases where calculation might fail or not be applicable
                if n > 0: # Avoid division by zero
                    sum_weights = np.sum(weights_sorted)
                    if sum_weights > 0:
                        index = np.arange(1, n + 1)
                        relative_weights = weights_sorted / sum_weights
                        # Gini = (1/n) * Sum((2*i - n - 1) * w_i) / Sum(w_i)
                        gini = (np.sum((2 * index - n - 1) * relative_weights)) / n
                        
                
                st.markdown("### Concentration Indices")
                index_cols = st.columns(2)
                index_cols[0].metric("HHI (Herfindahl-Hirschman)", f"{hhi:.2f}", help="Lower is more diversified. <1000 is good")
                index_cols[1].metric("Gini Coefficient", f"{gini:.4f}", help="0=perfect equality, 1=maximum inequality")
            else:
                 st.info("No holdings data available for concentration analysis.")
        
        # Note: analysis_tabs[5] is now 'Report'
        with analysis_tabs[5]:
            st.subheader("Export Report")
            
            if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
                from io import BytesIO
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Holdings sheet
                    results_df.to_excel(writer, sheet_name='Holdings', index=False)
                    
                    # Sector analysis
                    if 'Industry' in results_df.columns:
                        sector_analysis = results_df.groupby('Industry').agg({
                            'Weight %': 'sum',
                            'Real-time Value (Rs)': 'sum',
                            'Symbol': 'count'
                        }).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                        sector_analysis.to_excel(writer, sheet_name='Sector Analysis')
                    else:
                        st.warning("Industry data not available for sector analysis in report.")
                    
                    # Compliance results
                    if st.session_state.get("compliance_results"):
                        compliance_df = pd.DataFrame(st.session_state["compliance_results"])
                        compliance_df.to_excel(writer, sheet_name='Compliance Rules', index=False)
                    
                    # Breach alerts
                    if st.session_state.get("breach_alerts"):
                        breach_df = pd.DataFrame(st.session_state["breach_alerts"])
                        breach_df.to_excel(writer, sheet_name='Breaches', index=False)
                    
                    # Advanced Metrics (if available)
                    if st.session_state.get("advanced_metrics"):
                        metrics_df = pd.DataFrame([st.session_state["advanced_metrics"]]).T
                        metrics_df.columns = ['Value']
                        metrics_df.to_excel(writer, sheet_name='Advanced Metrics')
                    
                    # Threshold configs
                    config_df = pd.DataFrame([st.session_state["threshold_configs"]]).T
                    config_df.columns = ['Value']
                    config_df.to_excel(writer, sheet_name='Thresholds')
                
                output.seek(0)
                st.download_button(
                    "📥 Download Excel Report",
                    output,
                    f"compliance_report_{portfolio_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )


# --- TAB 2: Enhanced AI Analysis (Unchanged) ---
with tabs[1]:
    st.header("🤖 AI-Powered Compliance Analysis")
    
    portfolio_df = st.session_state.get("compliance_results_df")
    current_portfolio_name = st.session_state.get("current_portfolio_name")
    
    if not current_portfolio_name:
        st.warning("⚠️ Please create/load a portfolio first in the Portfolio Analysis tab")
        st.stop()
    
    if portfolio_df is None or portfolio_df.empty:
        st.warning("⚠️ Please analyze portfolio compliance first in the Portfolio Analysis tab")
        st.stop()
    
    if st.session_state.get("compliance_stage") not in ["compliance_done", "ai_completed"]:
        st.warning("⚠️ Complete compliance analysis first")
        st.stop()
    
    st.info(f"📁 **Portfolio:** {current_portfolio_name}")
    
    st.markdown("---")
    
    # Check if KIM document already exists
    existing_kim = get_kim_document(st.session_state["user_id"], current_portfolio_name)
    
    if existing_kim:
        st.success(f"✅ KIM/SID document already uploaded: **{existing_kim['file_name']}**")
        st.caption(f"Extracted on: {datetime.fromisoformat(existing_kim['extracted_at']).strftime('%Y-%m-%d %H:%M')}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.checkbox("📄 View document excerpt", key="view_kim"):
                st.text_area("Document Text (first 2000 chars)", existing_kim['document_text'][:2000], height=200, disabled=True)
        with col2:
            if st.button("🗑️ Delete & Re-upload", use_container_width=True):
                supabase.table('kim_documents').delete().eq('id', existing_kim['id']).execute()
                st.success("Deleted! Please upload new document.")
                time.sleep(0.5)
                st.rerun()
        
        docs_text = existing_kim['document_text']
        uploaded_docs = None
    else:
        st.subheader("Step 1: Upload KIM/SID Documents")
        uploaded_docs = st.file_uploader(
            "📄 Upload Scheme Documents (PDF/TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload Key Information Memorandum or Scheme Information Document"
        )
        
        docs_text = None
        if uploaded_docs:
            st.success(f"✅ {len(uploaded_docs)} document(s) uploaded")
            
            # Extract and save
            if st.button("💾 Extract & Save Documents", type="secondary", use_container_width=True):
                with st.spinner("Extracting text from documents..."):
                    docs_text_extracted = extract_text_from_files(uploaded_docs)
                    file_names = ", ".join([f.name for f in uploaded_docs])
                    
                    success, doc_id = save_kim_document(
                        st.session_state["user_id"],
                        current_portfolio_name,
                        docs_text_extracted,
                        file_names
                    )
                    
                    if success:
                        st.success("✅ Documents extracted and saved!")
                        st.session_state["kim_documents"][current_portfolio_name] = {
                            'document_text': docs_text_extracted,
                            'file_name': file_names
                        }
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Failed to save documents")
    
    st.markdown("---")
    
    # AI Analysis Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Step 2: Configure AI Analysis")
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Comprehensive"],
            value="Standard",
            key="ai_depth",
            help="Quick: Fast overview | Standard: Detailed analysis | Comprehensive: Deep dive with recommendations"
        )
    
    with col2:
        st.subheader("Analysis Options")
        include_market_context = st.checkbox("Include Market Context", value=True, key="ai_market_context")
        include_recommendations = st.checkbox("Include Recommendations", value=True, key="ai_recommendations")
    
    st.markdown("---")
    
    # Run AI Analysis
    if (docs_text or existing_kim) or st.session_state.get("ai_analysis_response"): # Allow running without docs if there's previous analysis
        if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True, key="ai_analyze_btn"):
            if not docs_text and existing_kim:
                 docs_text = existing_kim['document_text']
                 
            with st.spinner("🤖 AI is analyzing your portfolio..."):
                try:
                    portfolio_summary = get_portfolio_summary(portfolio_df)
                    breach_alerts = st.session_state.get("breach_alerts", [])
                    breach_summary = "\n".join([f"- {b['type']} ({b['severity']}): {b['details']}" for b in breach_alerts]) if breach_alerts else "No breaches detected."
                    
                    # Include compliance results
                    compliance_summary = ""
                    if st.session_state.get("compliance_results"):
                        compliance_summary = "\n**Custom Rule Results:**\n"
                        for rule in st.session_state["compliance_results"]:
                            display_status = "✅ PASS" if rule['status'] == "PASS" else "❌ FAIL" if rule['status'] == "FAIL" else rule['status']
                            
                            severity = "🟡 Medium" 
                            if rule['status'] == "FAIL":
                                if abs(rule.get('breach_amount', 0)) > rule.get('threshold', 0) * 0.2:
                                    severity = "🔴 Critical"
                                elif abs(rule.get('breach_amount', 0)) > rule.get('threshold', 0) * 0.1:
                                    severity = "🟠 High"
                            else:
                                severity = "✅ Compliant"

                            compliance_summary += f"- {display_status} {severity}: {rule['rule']} - {rule['details']}\n"
                    
                    # Include threshold configurations
                    threshold_summary = "\n**Threshold Configurations:**\n"
                    for key, value in st.session_state["threshold_configs"].items():
                        threshold_summary += f"- {key}: {value}\n"
                    
                    # Set prompt parameters based on depth
                    if analysis_depth == "Quick":
                        max_tokens = 8000
                        prompt_template = """You are an expert investment compliance analyst.

**PORTFOLIO:** {portfolio_summary}

**DETECTED ISSUES:** {breach_summary}

**COMPLIANCE RULES:** {compliance_summary}

Provide a concise executive summary covering:
1. Overall compliance status (2-3 sentences)
2. Top 3 critical issues
3. Immediate action items

Keep response under 500 words."""
                    
                    elif analysis_depth == "Standard":
                        max_tokens = 16000
                        prompt_template = """You are an expert investment compliance analyst with SEBI regulations knowledge.

**PORTFOLIO:** {portfolio_summary}

**DETECTED ISSUES:** {breach_summary}

**COMPLIANCE RULES:** {compliance_summary}

**THRESHOLDS:** {threshold_summary}

**SCHEME DOCUMENTS:** {docs_text_snippet}

Provide comprehensive analysis:

## 1. Executive Summary
Overall compliance status and key findings

## 2. Regulatory Compliance
- SEBI regulations (Single Issuer: 10%, Sector: 25%, Group: 25%)
- Scheme-specific requirements from documents

## 3. Portfolio Quality
Risk assessment and diversification analysis

## 4. Issues & Concerns
List all violations with severity and implications

## 5. Recommendations
Specific actionable steps to achieve compliance

Keep response under 2000 words."""
                    
                    else:  # Comprehensive
                        max_tokens = 25000
                        prompt_template = """You are an expert investment compliance analyst with deep knowledge of SEBI regulations and portfolio management.

**TASK:** Comprehensive compliance and risk analysis

**PORTFOLIO SNAPSHOT:**
{portfolio_summary}

**DETECTED BREACHES:**
{breach_summary}

**CUSTOM COMPLIANCE RULES:**
{compliance_summary}

**THRESHOLD CONFIGURATIONS:**
{threshold_summary}

**SCHEME DOCUMENTS (KIM/SID):**
{docs_text_snippet}

**ANALYSIS FRAMEWORK:**

## 1. Executive Summary (300 words)
- Overall compliance status
- Critical findings summary
- Key risk factors
- Priority action items

## 2. Investment Philosophy Alignment (400 words)
- Portfolio vs stated investment objectives
- Style consistency analysis
- Benchmark alignment

## 3. Regulatory Compliance Analysis (500 words)

### 3.1 SEBI Regulations
- Single Issuer Limit (10% max)
- Sectoral Concentration (25% max)
- Group Exposure (25% max)
- Detailed assessment with actual vs limits

### 3.2 Scheme-Specific Requirements
- Extract and verify all limits from uploaded documents
- Cross-reference with actual portfolio
- Highlight any deviations

### 3.3 Custom Rules Validation
- Analyze all custom rule breaches
- Assess severity and implications
- Prioritize remediation

## 4. Portfolio Quality & Risk Assessment (600 words)
- Concentration risk analysis (HHI, Gini coefficient)
- Diversification quality
- Liquidity profile
- Sector allocation efficiency

## 5. Violations & Regulatory Concerns (400 words)
Detailed list with:
- Violation type
- Severity (Critical/High/Medium/Low)
- Current vs Limit
- Potential regulatory implications
- Remediation complexity

## 6. Actionable Recommendations (500 words)

### Immediate Actions (0-30 days)
- Critical breach remediation
- Specific trades to suggest

### Short-term Actions (1-3 months)
- Portfolio rebalancing strategy
- Risk reduction measures

### Long-term Improvements (3-12 months)
- Process improvements
- Systematic risk management

**IMPORTANT:**
- Be specific with numbers and percentages
- Cite exact regulations where applicable
- Provide actionable, practical advice
- Highlight both immediate and strategic concerns
- Use clear severity classifications"""
                    
                    # Truncate docs_text for Gemini input to avoid token limits
                    docs_text_snippet = docs_text[:70000] if docs_text else "No scheme documents provided."

                    prompt = prompt_template.format(
                        portfolio_summary=portfolio_summary,
                        breach_summary=breach_summary,
                        compliance_summary=compliance_summary,
                        threshold_summary=threshold_summary,
                        docs_text_snippet=docs_text_snippet
                    )
                    
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            'temperature': 0.3,
                            'top_p': 0.8,
                            'max_output_tokens': max_tokens,
                        }
                    )
                    
                    st.session_state.ai_analysis_response = response.text
                    st.session_state.compliance_stage = "ai_completed"
                    
                    # Save to database
                    if st.session_state.get("current_portfolio_id"):
                        compliance_data = {
                            'threshold_configs': st.session_state["threshold_configs"],
                            'custom_rules': st.session_state.get("current_rules_text", ""),
                            'compliance_results': st.session_state.get("compliance_results", []),
                            'security_compliance': st.session_state.get("security_level_compliance", pd.DataFrame()).to_json(date_format='iso'),
                            'breach_alerts': st.session_state.get("breach_alerts", []),
                            'advanced_metrics': st.session_state.get("advanced_metrics", DEFAULT_ADVANCED_METRICS.copy()),
                            'risk_returns_df': st.session_state.get("risk_returns_df", pd.DataFrame()).to_json(date_format='iso'),
                            'benchmark_symbol': st.session_state.get("current_benchmark_symbol", BENCHMARK_NIFTY_SYMBOL),
                            'ai_analysis': response.text
                        }
                        
                        save_compliance_analysis(st.session_state["user_id"], st.session_state["current_portfolio_id"], compliance_data)
                        
                        # Update portfolio stage
                        supabase.table('portfolios').update({'analysis_stage': 'ai_completed'}).eq('id', st.session_state["current_portfolio_id"]).execute()
                        
                        st.success("✅ AI Analysis Complete and Saved!")
                        
                        # Refresh portfolio list
                        st.session_state["saved_analyses"] = get_user_portfolios(st.session_state["user_id"])
                        time.sleep(1)
                        st.rerun()
                
                except Exception as e:
                    st.error(f"❌ AI Analysis Error: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
    else:
        st.info("Upload KIM/SID documents or select an existing one to proceed with AI Analysis.")

    # Display AI Analysis Results
    if st.session_state.get("ai_analysis_response"):
        st.markdown("---")
        st.markdown("## 🤖 AI Analysis Report")
        st.markdown("---")
        
        st.markdown(st.session_state.ai_analysis_response)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            txt_data = st.session_state.ai_analysis_response.encode('utf-8')
            st.download_button(
                "📄 Download as TXT",
                txt_data,
                f"ai_analysis_{current_portfolio_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )
        
        with col2:
            md_data = st.session_state.ai_analysis_response.encode('utf-8')
            st.download_button(
                "📝 Download as Markdown",
                md_data,
                f"ai_analysis_{current_portfolio_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                use_container_width=True
            )
        
        with col3:
            if st.button("🗑️ Clear Analysis", use_container_width=True):
                st.session_state.ai_analysis_response = None
                st.rerun()

# --- TAB 3: Risk & Analysis (Enhanced) ---
with tabs[2]:
    st.header("📈 Risk & Analysis")

    portfolio_df = st.session_state.get("compliance_results_df")
    
    if 'compliance_results_df' not in st.session_state or st.session_state.compliance_results_df.empty:
        st.warning("⚠️ Please upload and analyze a portfolio in the 'Portfolio Analysis' tab first.")
        st.stop()
    
    if not k:
        st.warning("⚠️ Connect to Kite for real-time risk data.")
        st.stop()
        
    st.info(f"Analyzing: **{st.session_state.get('current_portfolio_name', 'Unnamed Portfolio')}**")
    
    # Removed risk_tab3
    risk_tab1, risk_tab2 = st.tabs(["Advanced Metrics & Performance", "Stress Testing & Scenarios"])

    # --- Risk Tab 1: Advanced Metrics ---
    with risk_tab1:
        st.subheader("Advanced Portfolio Risk Metrics")
        
        col_bench, col_period, col_calc = st.columns([2, 1, 1])
        
        with col_bench:
            st.session_state["current_benchmark_symbol"] = st.text_input(
                "Benchmark Symbol (e.g., NIFTY 50, MIDCAP 100)",
                value=st.session_state.get("current_benchmark_symbol", BENCHMARK_NIFTY_SYMBOL),
                key="benchmark_symbol_input",
                help="Enter the symbol of the benchmark index you want to compare against (e.g., 'NIFTY 50' or 'NIFTYMIDCAP 100')"
            ).strip().upper()
        
        with col_period:
            lookback = st.slider("Lookback Period (Days)", 90, 730, 366, key="risk_lookback")
            
        with col_calc:
            if st.button("🔄 Calculate/Recalculate Advanced Metrics", type="primary", use_container_width=True):
                if not st.session_state["current_benchmark_symbol"]:
                    st.error("Please enter a benchmark symbol.")
                else:
                    with st.spinner(f"Calculating advanced metrics using {lookback} days of history against {st.session_state['current_benchmark_symbol']}..."):
                        metrics, returns_df = calculate_advanced_metrics(
                            portfolio_df, 
                            api_key, 
                            access_token, 
                            lookback, 
                            st.session_state["current_benchmark_symbol"]
                        )
                        
                        # Update session state with results or defaults if calculation failed
                        if metrics and metrics.get('annualized_return') is not None:
                            st.session_state.advanced_metrics = metrics
                            st.session_state.risk_returns_df = returns_df
                            st.success("✅ Metrics calculated.")
                        else:
                            st.session_state.advanced_metrics = DEFAULT_ADVANCED_METRICS.copy()
                            st.session_state.risk_returns_df = pd.DataFrame()
                            st.error("Metrics calculation failed (see warnings above).")
                            
                        if st.session_state.get("current_portfolio_id"):
                            # Save metrics and returns data (returns as JSON string)
                            compliance_data = {
                                'threshold_configs': st.session_state["threshold_configs"],
                                'custom_rules': st.session_state.get("current_rules_text", ""),
                                'compliance_results': st.session_state.get("compliance_results", []),
                                'security_compliance': st.session_state.get("security_level_compliance", pd.DataFrame()).to_json(date_format='iso'),
                                'breach_alerts': st.session_state.get("breach_alerts", []),
                                'advanced_metrics': st.session_state.advanced_metrics,
                                'risk_returns_df': st.session_state.risk_returns_df.to_json(date_format='iso'),
                                'benchmark_symbol': st.session_state.get("current_benchmark_symbol", BENCHMARK_NIFTY_SYMBOL),
                                'ai_analysis': st.session_state.get("ai_analysis_response")
                            }
                            
                            save_compliance_analysis(st.session_state["user_id"], st.session_state["current_portfolio_id"], compliance_data)
                            st.success("✅ Metrics saved!")
                        st.rerun() # Rerun to refresh the display

        
        metrics = st.session_state.get("advanced_metrics", DEFAULT_ADVANCED_METRICS.copy())
        returns_df = st.session_state.get("risk_returns_df", pd.DataFrame())

        if metrics and metrics.get('annualized_return') is not None:
            st.markdown("---")
            st.markdown(f"### Performance Metrics (vs {st.session_state['current_benchmark_symbol']})")
            
            perf_cols = st.columns(4)
            # Accessing metrics using .get() for safety, and casting to float
            perf_cols[0].metric("Annualized Return", f"{float(metrics.get('annualized_return', 0.0)) * 100:.2f}%")
            perf_cols[1].metric("Portfolio Volatility", f"{float(metrics.get('portfolio_volatility', 0.0)) * 100:.2f}%")
            perf_cols[2].metric("Sharpe Ratio", f"{float(metrics.get('sharpe_ratio', 0.0)):.2f}")
            perf_cols[3].metric("Sortino Ratio", f"{float(metrics.get('sortino_ratio', 0.0)):.2f}") # FIX for Error 1
            
            st.markdown("### Relative Risk Metrics")
            relative_cols = st.columns(4)
            relative_cols[0].metric("Beta", f"{float(metrics.get('beta', 0.0)):.2f}")
            relative_cols[1].metric("Jensen's Alpha", f"{float(metrics.get('alpha', 0.0)) * 100:.2f}%")
            relative_cols[2].metric("Tracking Error", f"{float(metrics.get('tracking_error', 0.0)) * 100:.2f}%")
            relative_cols[3].metric("Information Ratio", f"{float(metrics.get('information_ratio', 0.0)):.2f}")

            st.markdown("### Concentration & Downside Risk")
            risk_cols = st.columns(4)
            risk_cols[0].metric("Diversification Ratio", f"{float(metrics.get('diversification_ratio', 0.0)):.2f}")
            risk_cols[1].metric("Avg Correlation", f"{float(metrics.get('avg_correlation', 0.0)):.2f}")
            risk_cols[2].metric("VaR (95%) Daily", f"{float(metrics.get('var_95', 0.0)) * 100:.2f}%")
            risk_cols[3].metric("CVaR (95%) Daily", f"{float(metrics.get('cvar_95', 0.0)) * 100:.2f}%")
            
            if not returns_df.empty and 'Portfolio' in returns_df.columns and 'Benchmark' in returns_df.columns:
                st.markdown("---")
                st.subheader("Performance Visualization")
                
                # Cumulative Return Plot
                # Ensure date column is the index for plotting
                if 'date' in returns_df.columns:
                    returns_df.set_index('date', inplace=True)
                
                cumulative_returns = (1 + returns_df[['Portfolio', 'Benchmark']]).cumprod() - 1
                cumulative_returns = cumulative_returns * 100 # Display as percentage
                
                # Rename Benchmark column for clearer display
                cumulative_returns = cumulative_returns.rename(columns={'Benchmark': st.session_state['current_benchmark_symbol']})

                fig_cum = px.line(cumulative_returns, y=['Portfolio', st.session_state['current_benchmark_symbol']],
                                  title="Cumulative Returns (%) vs Benchmark",
                                  labels={'value': 'Cumulative Return (%)', 'index': 'Date', 'variable': 'Index'})
                fig_cum.update_layout(legend_title_text="Index")
                st.plotly_chart(fig_cum, use_container_width=True)
                
                # Active Risk Plot (Tracking Error visualization)
                returns_df['Active Return'] = returns_df['Portfolio'] - returns_df['Benchmark']
                
                fig_active = px.area(returns_df.reset_index(), x='date', y='Active Return',
                                     title='Daily Active Return (Tracking Error)',
                                     labels={'Active Return': 'Daily Active Return', 'date': 'Date'})
                st.plotly_chart(fig_active, use_container_width=True)
            elif not returns_df.empty:
                st.warning("Returns data is incomplete for visualization (missing 'Portfolio' or 'Benchmark' columns).")
        else:
            st.info("Click 'Calculate/Recalculate Advanced Metrics' to generate portfolio risk data.")
            

    # --- Risk Tab 2: Stress Testing ---
    with risk_tab2:
        st.subheader("Stress Testing and Scenario Modeling")

        col1, col2 = st.columns([1, 2])
        with col1:
            scenario_type = st.selectbox(
                "Select a Stress Scenario",
                ["Market Crash", "Sector Shock", "Single Stock Failure"],
                key="stress_scenario_type_risk"
            )
        
        params = {}
        with col2:
            if scenario_type == "Market Crash":
                params['percentage'] = st.slider("Market-wide Drop (%)", 5, 50, 20, key="scen_pct_market")
            elif scenario_type == "Sector Shock":
                all_sectors = sorted(portfolio_df['Industry'].unique().tolist())
                if not all_sectors:
                    st.warning("No sectors found in portfolio data.")
                    st.stop()
                params['sector'] = st.selectbox("Select Sector to Shock", all_sectors, key="scen_sector")
                params['percentage'] = st.slider(f"Drop in {params['sector']} Sector (%)", 5, 75, 25, key="scen_pct_sector")
            elif scenario_type == "Single Stock Failure":
                all_stocks = sorted(portfolio_df['Symbol'].unique().tolist())
                if not all_stocks:
                    st.warning("No stocks found in portfolio data.")
                    st.stop()
                params['symbol'] = st.selectbox("Select Stock to Shock", all_stocks, key="scen_symbol")
                params['percentage'] = st.slider(f"Drop in {params['symbol']} (%)", 10, 90, 50, key="scen_pct_stock")

        if st.button("🔬 Run Stress Test", use_container_width=True, type="primary", key="run_stress_risk"):
            with st.spinner("Simulating scenario and auditing compliance..."):
                stressed_df, summary = run_stress_test(portfolio_df, scenario_type, params)
                st.session_state['stressed_df'] = stressed_df
                st.session_state['stress_summary'] = summary
                
                # --- FIX: Ensure columns are clean before calling the API checker ---
                stressed_df_for_api = stressed_df.rename(columns={'Stressed Weight %': 'Weight %'}).copy()
                
                stressed_compliance_results = [] # Initialize here to ensure it always exists
                
                # Validate and clean the DataFrame elements before calling the API
                for col in ['Quantity', 'LTP', 'Real-time Value (Rs)', 'Weight %']:
                    if col in stressed_df_for_api.columns:
                        # Ensure the data in the column is a scalar or simple list/array
                        # This extra step tries to flatten any nested structures within cells
                        # that might prevent pd.to_numeric from operating directly on the Series.
                        # For example, if a cell accidentally became `[50.0]`, this turns it to `50.0`.
                        try:
                            # Apply a lambda function to each element to flatten if it's a list/array
                            # This is a defensive step for unexpected data structures in cells.
                            stressed_df_for_api[col] = stressed_df_for_api[col].apply(
                                lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) == 1 else x
                            )
                            # Then, convert the Series to numeric
                            stressed_df_for_api[col] = pd.to_numeric(stressed_df_for_api[col], errors='coerce').fillna(0.0)
                        except Exception as e:
                            st.error(f"Error cleaning column '{col}' for API call in stress test: {e}. Skipping API call.")
                            stressed_compliance_results = [] # Indicate failure to get results
                            break # Exit the loop if cleaning fails for a column
                else: # This else block executes if the for loop completes without a 'break'
                    stressed_compliance_results = call_compliance_api_run_check(
                        stressed_df_for_api,
                        st.session_state.current_rules_text,
                        st.session_state.threshold_configs
                    )
                # --- END FIX ---
                
                st.session_state['stressed_compliance_results'] = stressed_compliance_results
                st.success("Stress test simulation complete.")
        
        
        if 'stress_summary' in st.session_state and st.session_state['stress_summary'] is not None:
            st.markdown("---")
            st.subheader("Stress Test Results: Impact Summary")
            
            summary = st.session_state['stress_summary']
            stressed_df = st.session_state['stressed_df']
            
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Original Value", f"₹ {summary['original_value']:,.0f}")
            kpi_cols[1].metric("Stressed Value", f"₹ {summary['stressed_value']:,.0f}")
            kpi_cols[2].metric(
                label="Loss (Value)", 
                value=f"₹ {summary['loss_value']:,.0f}", 
                delta=f"-₹ {summary['loss_value']:,.0f}",
                delta_color="inverse"
            )
            kpi_cols[3].metric(
                label="Loss (%)", 
                value=f"{summary['loss_pct']:.2f}%", 
                delta=f"-{summary['loss_pct']:.2f}%",
                delta_color="inverse"
            )

            st.markdown("#### Post-Stress Compliance Audit")
            stressed_results = st.session_state['stressed_compliance_results']
            new_breaches = [r for r in stressed_results if r['status'] == "FAIL"]
            
            if not new_breaches:
                st.success("✅ **Portfolio remains compliant under this stress scenario.**")
            else:
                st.error(f"🚨 **{len(new_breaches)} Compliance Breaches Triggered Under Stress!**")
                breach_data = []
                for breach in new_breaches:
                    severity = "🟡 Medium" 
                    if abs(breach.get('breach_amount', 0)) > breach.get('threshold', 0) * 0.2:
                        severity = "🔴 Critical"
                    elif abs(breach.get('breach_amount', 0)) > breach.get('threshold', 0) * 0.1:
                        severity = "🟠 High"

                    breach_data.append({
                        "Rule": breach['rule'],
                        "Severity": severity,
                        "Details": breach['details']
                    })
                st.dataframe(pd.DataFrame(breach_data), use_container_width=True, hide_index=True)
            
            st.markdown("#### Detailed Portfolio Impact")
            
            # Show top 5 losers
            display_df = stressed_df.copy()
            display_df['Value Change (Rs)'] = display_df['Stressed Value (Rs)'] - display_df['Real-time Value (Rs)']
            top_5_losers = display_df.sort_values('Value Change (Rs)').head(5)[['Symbol', 'Value Change (Rs)', 'Weight %', 'Stressed Weight %']]
            
            st.dataframe(top_5_losers.style.format({
                'Value Change (Rs)': '₹{:,.0f}',
                'Weight %': '{:.2f}%',
                'Stressed Weight %': '{:.2f}%'
            }), use_container_width=True)


# --- TAB 4: API Interactions (Unchanged) ---
with tabs[3]:
    st.header("🔧 Compliance API Interactions")
    st.markdown("Interact directly with the compliance backend API for various simulation and suggestion tasks.")

    current_portfolio_df = st.session_state.get("compliance_results_df")
    current_rules_text = st.session_state.get("current_rules_text")
    current_threshold_configs = st.session_state.get("threshold_configs")

    if current_portfolio_df is None or current_portfolio_df.empty:
        st.warning("⚠️ Please load or analyze a portfolio in 'Portfolio Analysis' tab to use API functions.")
        st.stop()
    if not current_rules_text:
        st.warning("⚠️ Please define compliance rules in 'Portfolio Analysis' tab to use API functions.")
        st.stop()
    
    st.info(f"**Using Portfolio:** `{st.session_state.get('current_portfolio_name', 'Unnamed Portfolio')}`")
    st.caption("The portfolio data, rules, and thresholds from the 'Portfolio Analysis' tab are automatically used.")

    # Convert current_portfolio_df to a JSON-serializable list of dicts for the API calls
    portfolio_for_api = current_portfolio_df[['Symbol', 'Name', 'Quantity', 'LTP', 'Industry']].copy()
    portfolio_for_api.fillna({'Industry': 'UNKNOWN'}, inplace=True) 
    portfolio_for_api['Quantity'] = pd.to_numeric(portfolio_for_api['Quantity'], errors='coerce').fillna(0)
    portfolio_for_api['LTP'] = pd.to_numeric(portfolio_for_api['LTP'], errors='coerce').fillna(0)
    portfolio_for_api = portfolio_for_api[portfolio_for_api['Quantity'] > 0] # Filter out zero quantity holdings


    api_call_tab1, api_call_tab2, api_call_tab3, api_call_tab4 = st.tabs([
        "Pre-Trade Simulation", 
        "Optimal Trade Suggester", 
        "Cash Flow Simulation", 
        "Block Trade Allocation"
    ])

    with api_call_tab1:
        st.subheader("Simulate Proposed Trades")
        st.write("Test a single buy/sell trade against your current portfolio and rules.")
        
        trade_col1, trade_col2 = st.columns(2)
        
        default_symbol = portfolio_for_api['Symbol'].iloc[0] if not portfolio_for_api.empty else "RELIANCE"
        
        with trade_col1:
            trade_symbol = st.text_input("Trade Symbol", key="trade_symbol", value=default_symbol)
            trade_action = st.selectbox("Action", ["BUY", "SELL"], key="trade_action")
        with trade_col2:
            trade_quantity = st.number_input("Quantity", min_value=1, value=10, key="trade_quantity")
            
        current_ltp_for_trade = portfolio_for_api[portfolio_for_api['Symbol'] == trade_symbol]['LTP'].iloc[0] if trade_symbol in portfolio_for_api['Symbol'].values else 0.0
        trade_ltp = st.number_input(f"LTP for {trade_symbol}", value=float(current_ltp_for_trade), min_value=0.01)
        
        trade_industry = portfolio_for_api[portfolio_for_api['Symbol'] == trade_symbol]['Industry'].iloc[0] if trade_symbol in portfolio_for_api['Symbol'].values else "UNKNOWN"
        trade_industry = st.text_input(f"Industry for {trade_symbol}", value=str(trade_industry))

        if st.button("Simulate Trade", type="primary"):
            trade_payload = {
                "portfolio": portfolio_for_api.to_dict('records'),
                "rules_text": current_rules_text,
                "threshold_configs": current_threshold_configs,
                "trade": {
                    "symbol": trade_symbol.upper(),
                    "action": trade_action.upper(),
                    "quantity": int(trade_quantity),
                    "ltp": float(trade_ltp),
                    "industry": trade_industry.upper(),
                    "name": trade_symbol.upper() # Add 'Name' to trade for API compatibility
                }
            }
            with st.spinner(f"Simulating {trade_action} {trade_quantity} {trade_symbol}..."):
                response_data = call_compliance_api("/simulate/trade", trade_payload)
                if response_data:
                    st.success("Pre-trade simulation results:")
                    simulated_df = pd.DataFrame(response_data['simulated_portfolio'])
                    st.markdown("##### Simulated Portfolio")
                    st.dataframe(simulated_df.style.format({'Real-time Value (Rs)': '₹{:,.2f}', 'Weight %': '{:.2f}%', 'LTP': '₹{:,.2f}'}), use_container_width=True)
                    st.markdown("##### Compliance Results After Trade")
                    st.dataframe(pd.DataFrame(response_data['compliance_results']), use_container_width=True, hide_index=True)
                else:
                    st.error("Failed to get pre-trade simulation results.")

    with api_call_tab2:
        st.subheader("Optimal Trade Suggester")
        st.write("Get suggestions for trades to resolve any detected compliance breaches.")

        if st.button("Get Trade Suggestions", type="primary"):
            trade_suggestion_payload = {
                "portfolio": portfolio_for_api.to_dict('records'),
                "rules_text": current_rules_text,
                "threshold_configs": current_threshold_configs
            }
            with st.spinner("Requesting trade suggestions..."):
                response_data = call_compliance_api("/suggest/trades", trade_suggestion_payload)
                if response_data:
                    if response_data['suggestions']:
                        st.success("Optimal trade suggestions:")
                        st.dataframe(pd.DataFrame(response_data['suggestions']), use_container_width=True, hide_index=True)
                    else:
                        st.info(response_data.get('message', "No suggestions available, portfolio might be compliant."))
                    
                    st.markdown("---")
                    st.markdown("##### Current Compliance Before Suggestions (from API)")
                    st.dataframe(pd.DataFrame(response_data['current_compliance']), use_container_width=True, hide_index=True)
                else:
                    st.error("Failed to get trade suggestions.")

    with api_call_tab3:
        st.subheader("Cash Flow Simulation")
        st.write("Simulate the impact of adding or withdrawing cash from the portfolio.")

        cash_amount = st.number_input("Cash Amount (Rs)", value=100000.0, step=10000.0, help="Positive for inflow, negative for outflow.")
        
        if st.button("Simulate Cash Flow", type="primary"):
            cash_flow_payload = {
                "portfolio": portfolio_for_api.to_dict('records'),
                "rules_text": current_rules_text,
                "threshold_configs": current_threshold_configs,
                "cash_flow": {
                    "amount": float(cash_amount)
                }
            }
            with st.spinner("Simulating cash flow..."):
                response_data = call_compliance_api("/simulate/cashflow", cash_flow_payload)
                if response_data:
                    st.success("Cash flow simulation results:")
                    simulated_df = pd.DataFrame(response_data['simulated_portfolio'])
                    st.markdown("##### Simulated Portfolio with Cash Flow")
                    st.dataframe(simulated_df.style.format({'Real-time Value (Rs)': '₹{:,.2f}', 'Weight %': '{:.2f}%', 'LTP': '₹{:,.2f}'}), use_container_width=True)
                    st.markdown("##### Compliance Results After Cash Flow")
                    st.dataframe(pd.DataFrame(response_data['compliance_results']), use_container_width=True, hide_index=True)
                else:
                    st.error("Failed to simulate cash flow.")

    with api_call_tab4:
        st.subheader("Block Trade Allocation Check")
        st.write("Check how a hypothetical block trade (split and allocated to your current portfolio) affects compliance. This assumes the entire block trade is allocated to this one portfolio for demonstration.")

        bt_symbol = st.text_input("Block Trade Symbol", key="bt_symbol", value="RELIANCE")
        bt_ltp = st.number_input("Block Trade LTP (Rs)", value=2500.0, key="bt_ltp")
        bt_quantity = st.number_input("Block Trade Total Quantity", min_value=1, value=500, key="bt_quantity")
        bt_action = st.selectbox("Block Trade Action", ["BUY", "SELL"], key="bt_action_block")
        
        # Determine industry for the block trade symbol
        bt_industry = portfolio_for_api[portfolio_for_api['Symbol'].str.upper() == bt_symbol.upper()]['Industry'].iloc[0] \
                        if bt_symbol.upper() in portfolio_for_api['Symbol'].str.upper().values else "UNKNOWN"
        bt_industry_input = st.text_input(f"Industry for {bt_symbol}", value=str(bt_industry), key="bt_industry")

        if st.button("Check Block Trade Allocation", type="primary"):
            if not st.session_state.get('current_portfolio_id'):
                st.error("No current portfolio loaded to check allocation against. Please load a portfolio first.")
            else:
                # For demonstration, we'll simulate the entire block trade being allocated to the *current* portfolio
                allocation_payload = {
                    "portfolios": [
                        {
                            "id": st.session_state.current_portfolio_id,
                            "holdings": portfolio_for_api.to_dict('records'), # Using the current portfolio
                            "rules_text": current_rules_text,
                            "threshold_configs": current_threshold_configs,
                            "allocation_quantity": int(bt_quantity) # Allocating total quantity to this single portfolio
                        }
                    ],
                    "block_trade": {
                        "symbol": bt_symbol.upper(),
                        "action": bt_action.upper(),
                        "ltp": float(bt_ltp),
                        "industry": bt_industry_input.upper(),
                        "name": bt_symbol.upper() # Add 'Name' to block_trade for API compatibility
                    }
                }
                
                with st.spinner(f"Checking block trade allocation for {bt_symbol}..."):
                    response_data = call_compliance_api("/simulate/block_allocation", allocation_payload)
                    if response_data and response_data['allocation_results']:
                        st.success("Block trade allocation results:")
                        # Display results for each portfolio (in this case, just the one current portfolio)
                        for res in response_data['allocation_results']:
                            st.markdown(f"##### Portfolio ID: {res['portfolio_id']}")
                            if res['breach_count'] > 0:
                                st.error(f"❌ {res['breach_count']} breaches detected after allocation.")
                            else:
                                st.success("✅ Compliant after allocation.")
                            st.dataframe(pd.DataFrame(res['compliance_results']), use_container_width=True, hide_index=True)
                    else:
                        st.error("Failed to check block trade allocation.")


# --- TAB 5: History (Unchanged) ---
with tabs[4]:
    st.header("📚 Portfolio History")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        portfolios = st.session_state.get("saved_analyses", [])
        st.info(f"**{len(portfolios)}** saved portfolios")
    
    with col2:
        if st.button("🔄 Refresh List", use_container_width=True):
            st.session_state["saved_analyses"] = get_user_portfolios(st.session_state["user_id"])
            st.rerun()
    
    if not portfolios:
        st.info("📭 No portfolios yet! Create one in the Portfolio Analysis tab.")
    else:
        st.markdown("---")
        
        # Group by stage
        stage_groups = {
            'ai_completed': [],
            'compliance_done': [],
            'upload': []
        }
        
        for p in portfolios:
            stage = p.get('analysis_stage', 'upload')
            if stage in stage_groups:
                stage_groups[stage].append(p)
        
        # Display by completion status
        if stage_groups['ai_completed']:
            st.markdown("### 🤖 AI Analysis Complete")
            for portfolio in stage_groups['ai_completed'][:10]:
                render_portfolio_card(portfolio)
        
        if stage_groups['compliance_done']:
            st.markdown("### ✅ Compliance Analyzed")
            for portfolio in stage_groups['compliance_done'][:10]:
                render_portfolio_card(portfolio)
        
        if stage_groups['upload']:
            st.markdown("### 📤 Uploaded Only")
            for portfolio in stage_groups['upload'][:10]:
                render_portfolio_card(portfolio)


# --- Footer ---
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Invsion Connect</strong> - Professional Portfolio Compliance Platform</p>
    <p style='font-size: 0.9em;'>⚠️ For informational purposes only. Consult professionals for investment decisions.</p>
    <p style='font-size: 0.8em;'>Powered by KiteConnect, Google Gemini AI & Supabase</p>
    <p style='font-size: 0.8em;'>User: {st.session_state["user_email"]} | Session Active</p>
</div>
""", unsafe_allow_html=True)
