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
import fitz  # PyMuPDF
import hashlib
import requests # Import requests for API calls

# --- AI Imports ---
try:
    import google.generativeai as genai
except ImportError:
    # Running this in a local environment where the library might not be installed
    st.error("Google Generative AI library not found. Install: pip install google-generativeai")
    # Commenting out st.stop() to allow development without immediate AI functionality if needed
    # st.stop()
    pass # Continue execution if AI is not strictly necessary for UI

# --- KiteConnect Imports ---
try:
    from kiteconnect import KiteConnect
except ImportError:
    st.error("KiteConnect library not found. Install: pip install kiteconnect")
    # st.stop()
    pass # Continue execution if Kite is not strictly necessary for UI

# --- Supabase Import ---
try:
    from supabase import create_client, Client
except ImportError:
    st.error("Supabase library not found. Install: pip install supabase")
    # st.stop()
    pass # Continue execution if Supabase is not strictly necessary for UI


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect Pro", layout="wide", initial_sidebar_state="expanded")

# --- Global Constants ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
BENCHMARK_SYMBOL = "NIFTY 50"
# WARNING: Setting the API base URL to match the structure of the provided Flask code.
# Assuming the Flask app is running locally on port 5001
COMPLIANCE_API_BASE_URL = "https://zeroapiv4.onrender.com/" 

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "user_authenticated": False,
        "user_id": None,
        "user_email": None,
        "kite_access_token": None,
        "compliance_results_df": pd.DataFrame(),
        "compliance_results": [],
        "advanced_metrics": None,
        "ai_analysis_response": None,
        "security_level_compliance": pd.DataFrame(),
        "breach_alerts": [],
        "saved_analyses": [],
        "current_rules_text": "",
        "current_portfolio_id": None,
        "current_portfolio_name": None,
        "kim_documents": {},
        "compliance_stage": "upload",
        "stress_summary": None,
        "stressed_df": None,
        "stressed_compliance_results": None,
        # New API States (Updated to reflect Flask endpoints)
        "comprehensive_analytics_data": None, # Holds data from /analytics/comprehensive
        "rebalance_suggestions": None, # Holds data from /simulation/rebalance
        "optimization_results": None, # Holds data from /simulation/optimize
        "correlation_matrix_data": None, # Holds data from /analytics/correlation
        "volatility_cone_data": None, # Holds data from /simulation/volatility_cone
        "stress_test_api_results": None, # Holds data from /simulation/stress_test
        "threshold_configs": {
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# --- Load Credentials ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    gemini_conf = secrets.get("google_gemini", {})
    supabase_conf = secrets.get("supabase", {})
    
    errors = []
    # Check if necessary secrets exist, even if the libraries are mocked out
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        errors.append("Kite credentials")
    if not gemini_conf.get("api_key"):
        errors.append("Gemini API key")
    if not supabase_conf.get("url") or not supabase_conf.get("key"):
        errors.append("Supabase credentials")

    if errors:
        st.error(f"Missing credentials (using dummy values if libraries failed to import): {', '.join(errors)}")
        # Provide dummy credentials if loading failed to allow UI to render
        kite_conf = {"api_key": "dummy", "api_secret": "dummy", "redirect_uri": "http://localhost:8501"}
        gemini_conf = {"api_key": "dummy"}
        supabase_conf = {"url": "dummy", "key": "dummy"}

    return kite_conf, gemini_conf, supabase_conf

KITE_CREDENTIALS, GEMINI_CREDENTIALS, SUPABASE_CREDENTIALS = load_secrets()

# Initialize AI and DB only if modules were imported successfully
try:
    if GEMINI_CREDENTIALS["api_key"] != "dummy":
        genai.configure(api_key=GEMINI_CREDENTIALS["api_key"])
except NameError:
    pass # Skip if genai was not imported

# --- Initialize Supabase (Cached to prevent reconnects) ---
@st.cache_resource
def init_supabase() -> Client:
    try:
        if SUPABASE_CREDENTIALS["url"] != "dummy":
            return create_client(SUPABASE_CREDENTIALS["url"], SUPABASE_CREDENTIALS["key"])
    except NameError:
        pass # Skip if supabase was not imported
    
    # Fallback dummy client if Supabase is unavailable
    class DummySupabaseClient:
        def auth(self):
            return self
        def sign_up(self, credentials):
            return type('Response', (object,), {'user': None})
        def sign_in_with_password(self, credentials):
            return type('Response', (object,), {'user': type('User', (object,), {'id': 'dummy_id', 'email': credentials['email']}), 'session': 'dummy_session'})
        def sign_out(self):
            pass
        def table(self, table_name):
            return self
        def select(self, *args, **kwargs):
            return self
        def eq(self, *args, **kwargs):
            return self
        def execute(self):
            return type('ExecutionResult', (object,), {'data': []})
        def insert(self, data):
            return self
        def update(self, data):
            return self
        def delete(self):
            return self
        def order(self, *args, **kwargs):
            return self

    return DummySupabaseClient()

supabase = init_supabase()


# --- Authentication Functions (Using Dummy if Supabase failed to import) ---
def register_user(email: str, password: str):
    if type(supabase).__name__ == 'DummySupabaseClient':
        return False, "Database not connected. Cannot register."
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
    if type(supabase).__name__ == 'DummySupabaseClient':
        # Dummy login for local testing if DB failed
        if email == "test@test.com" and password == "password":
            st.session_state["user_authenticated"] = True
            st.session_state["user_id"] = "dummy_id"
            st.session_state["user_email"] = email
            st.session_state["supabase_session"] = "dummy_session"
            return True, "Login successful (Dummy Mode)!"
        return False, "Invalid credentials."
    
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
    if type(supabase).__name__ != 'DummySupabaseClient':
        try:
            supabase.auth.sign_out()
        except:
            pass
    st.session_state.clear()
    init_session_state()

# --- Database Functions (Skipped if DummySupabaseClient is active) ---
# ... (All database functions remain unchanged, assuming successful import/initialization) ...
def save_kim_document(user_id: str, portfolio_name: str, document_text: str, file_name: str):
    """Save KIM/SID document for a portfolio"""
    if type(supabase).__name__ == 'DummySupabaseClient': return True, 1
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
    if type(supabase).__name__ == 'DummySupabaseClient': return None
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
    if type(supabase).__name__ == 'DummySupabaseClient': return True, 'dummy_pid'
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
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False, None

def save_compliance_analysis(user_id: str, portfolio_id: str, compliance_data: dict):
    """Save compliance analysis results - FIXED for existing schema"""
    if type(supabase).__name__ == 'DummySupabaseClient': return True, portfolio_id
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
        
        # Store advanced_metrics separately in portfolio metadata if provided
        if compliance_data.get('advanced_metrics'):
            portfolio_metadata_update = {
                'metadata': {
                    'advanced_metrics': compliance_data['advanced_metrics'],
                    'last_updated': datetime.now().isoformat()
                }
            }
            supabase.table('portfolios').update(portfolio_metadata_update).eq('id', portfolio_id).execute()
        
        # Update portfolio stage
        supabase.table('portfolios').update({'analysis_stage': 'ai_completed' if compliance_data.get('ai_analysis') else 'compliance_done'}).eq('id', portfolio_id).execute()
        
        return True, portfolio_id
    except Exception as e:
        st.error(f"Error saving compliance analysis: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return False, None

def get_user_portfolios(user_id: str):
    """Get all portfolios for a user"""
    if type(supabase).__name__ == 'DummySupabaseClient': return []
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
    if type(supabase).__name__ == 'DummySupabaseClient': return None
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
        
        # Extract advanced_metrics from portfolio metadata if available
        advanced_metrics = None
        if portfolio.get('metadata') and isinstance(portfolio['metadata'], dict):
            advanced_metrics = portfolio['metadata'].get('advanced_metrics')
        
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
            'ai_analysis': analysis_result.data[0].get('ai_analysis') if analysis_result.data else None,
            'kim_document': kim_result.data[0] if kim_result.data else None,
            'metadata': portfolio.get('metadata', {})
        }
        
        return combined
    except Exception as e:
        st.error(f"Error loading portfolio: {str(e)}")
        return None

def delete_portfolio(portfolio_id: str):
    """Delete portfolio and all related data"""
    if type(supabase).__name__ == 'DummySupabaseClient': return True
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
# Initialize Kite client only if the library was imported
try:
    @st.cache_resource(ttl=3600)
    def init_kite_unauth_client(api_key: str) -> KiteConnect:
        return KiteConnect(api_key=api_key)

    kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
    login_url = kite_unauth_client.login_url()
except NameError:
    # Dummy Kite client if import failed
    class DummyKiteClient:
        def login_url(self): return "https://kite.trade/connect/login"
        def generate_session(self, token, api_secret): return {"access_token": "DUMMY_TOKEN"}
        def ltp(self, instruments): return {instruments[0]: {'last_price': 100.0}}
        def instruments(self, exchange): return []
        def historical_data(self, token, from_date, to_date, interval): return []
        def set_access_token(self, token): pass
    
    kite_unauth_client = DummyKiteClient()
    login_url = kite_unauth_client.login_url()

# --- Utility Functions ---
def get_authenticated_kite_client(api_key: str, access_token: str):
    try:
        if api_key and access_token and 'KiteConnect' in globals():
            k = KiteConnect(api_key=api_key)
            k.set_access_token(access_token)
            return k
    except NameError:
        pass # KiteConnect not available
    return None

# NO CACHE FOR LTP - Fetches fresh data every time
def get_ltp_price_fresh(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    k = get_authenticated_kite_client(api_key, access_token)
    if not k: 
        # Dummy price if Kite not connected
        return 100.0 + (hash(symbol) % 100) # Pseudo-random dummy price
    try:
        # Force fresh fetch by not using any cache decorator
        quote = k.ltp([f"{exchange.upper()}:{symbol.upper()}"]).get(f"{exchange.upper()}:{symbol.upper()}")
        if quote and 'last_price' in quote:
            return quote['last_price']
        return 0.0 
    except Exception as e:
        st.error(f"Error fetching LTP for {symbol}: {e}")
        return 0.0

# CACHE ENABLED for Historical Data (Heavy payload, changes infrequently for daily metrics)
@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date, to_date, interval: str, exchange: str = DEFAULT_EXCHANGE):
    k = get_authenticated_kite_client(api_key, access_token)
    if not k: 
        # Return dummy data if Kite not connected
        dates = pd.to_datetime(pd.date_range(end=to_date, periods=TRADING_DAYS_PER_YEAR, freq='B'))
        df = pd.DataFrame({
            "date": dates,
            "close": 100 + np.cumsum(np.random.normal(0, 1, TRADING_DAYS_PER_YEAR)),
            "_error": ["Dummy Data"]
        })
        df.set_index("date", inplace=True)
        return df
    
    try:
        instruments = k.instruments(exchange)
        df_inst = pd.DataFrame(instruments)
        token_row = df_inst[(df_inst['exchange'] == exchange.upper()) & (df_inst['tradingsymbol'] == symbol.upper())]
        
        if token_row.empty:
            return pd.DataFrame({"_error": [f"Token not found for {symbol}"]})
        
        token = int(token_row.iloc[0]['instrument_token'])
        data = k.historical_data(token, from_date=datetime.combine(from_date, datetime.min.time()), 
                                to_date=datetime.combine(to_date, datetime.max.time()), interval=interval)
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
        # st.info(f"Calling API: {url}...")
        response = requests.post(url, json=payload, timeout=90) # Increased timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        # st.success(f"API call to {endpoint} successful!")
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_details = e.response.text
        st.error(f"API HTTP Error ({endpoint}): {e.response.status_code} - Details: {error_details}")
        return None
    except requests.exceptions.ConnectionError as e:
        st.error(f"API Connection Error ({endpoint}): Ensure Flask server is running at {COMPLIANCE_API_BASE_URL.split('/api/v2')[0]}")
        return None
    except requests.exceptions.Timeout as e:
        st.error(f"API Timeout Error ({endpoint}): {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An unexpected API Request Error occurred ({endpoint}): {e}")
        return None
    except json.JSONDecodeError:
        st.error(f"Failed to decode JSON response from API ({endpoint}): {response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API call ({endpoint}): {e}")
        return None


# --- Compliance Functions (using API) ---

def call_compliance_api_run_check(portfolio_df: pd.DataFrame, rules_text: str, threshold_configs: dict):
    """
    Calls the *hypothetical* /simulate/trade endpoint (used in previous logic) 
    or its functional equivalent (since the new Flask code doesn't have the full rule engine).
    
    We will mock the rule engine result structure since the provided Flask code only runs:
    1. VaR/Liquidity/Tax (via /comprehensive)
    2. Optimization (via /optimize)
    3. Stress Test (via /stress_test)
    4. Rebalance (via /rebalance)

    For basic compliance checking, the Flask code provided does NOT have a dedicated endpoint
    like the old Streamlit code assumed. We will use the health check and mock compliance results
    to allow the main analysis tab to proceed, assuming the rules engine API is separate.
    """
    
    # Mocking the compliance API call result structure for rules validation:
    # In a real system, this would call an external rules engine.
    # Since the Flask code provided does *not* contain the complex rule validation logic,
    # we return a mocked, failure-prone structure based on the current rules text.

    mocked_results = []
    
    # 1. Check basic concentration limits (always include this mock)
    single_stock_limit = threshold_configs.get('single_stock_limit', 10.0)
    
    # Find the largest stock weight to mock a breach if it exceeds the limit
    max_weight = portfolio_df['Weight %'].max() if not portfolio_df.empty else 0
    breach_amount = max_weight - single_stock_limit
    
    if breach_amount > 0:
        mocked_results.append({
            'rule': f'MAX_STOCK_WEIGHT <= {single_stock_limit}',
            'status': 'FAIL',
            'details': f'Portfolio Max Weight is {max_weight:.2f}%',
            'threshold': single_stock_limit,
            'current_value': max_weight,
            'breach_amount': breach_amount
        })
    else:
        mocked_results.append({
            'rule': f'MAX_STOCK_WEIGHT <= {single_stock_limit}',
            'status': 'PASS',
            'details': f'Portfolio Max Weight is {max_weight:.2f}%',
            'threshold': single_stock_limit,
            'current_value': max_weight,
            'breach_amount': 0.0
        })

    # 2. Check holdings count (mock based on threshold)
    holdings_count = len(portfolio_df)
    min_holdings = threshold_configs.get('min_holdings', 20)
    if holdings_count < min_holdings:
        mocked_results.append({
            'rule': f'COUNT_STOCKS >= {min_holdings}',
            'status': 'FAIL',
            'details': f'Only {holdings_count} holdings.',
            'threshold': min_holdings,
            'current_value': holdings_count,
            'breach_amount': float(min_holdings - holdings_count)
        })
    else:
        mocked_results.append({
            'rule': f'COUNT_STOCKS >= {min_holdings}',
            'status': 'PASS',
            'details': f'{holdings_count} holdings found.',
            'threshold': min_holdings,
            'current_value': holdings_count,
            'breach_amount': 0.0
        })

    # For all other rules, assume PASS unless manually triggered to FAIL
    if "HHI < 800" in rules_text:
        # Calculate HHI locally for a slightly better mock
        hhi = (portfolio_df['Weight %'] ** 2).sum() * 100
        if hhi > 800:
            mocked_results.append({
                'rule': 'HHI < 800',
                'status': 'FAIL',
                'details': f'HHI is {hhi:.2f}',
                'threshold': 800,
                'current_value': hhi,
                'breach_amount': float(hhi - 800)
            })
        else:
            mocked_results.append({
                'rule': 'HHI < 800',
                'status': 'PASS',
                'details': f'HHI is {hhi:.2f}',
                'threshold': 800,
                'current_value': hhi,
                'breach_amount': 0.0
            })
            
    return mocked_results


def call_comprehensive_analytics_api(portfolio_df: pd.DataFrame):
    """Calls the new comprehensive risk analytics endpoint (/api/v2/analytics/comprehensive)."""
    # Note: Flask API requires LTP, Quantity, Avg_Buy_Price, Beta, Avg_Volume, Volatility
    # We must ensure the DataFrame sent has these columns, using mock defaults if necessary.
    df_send = portfolio_df.rename(columns={'Real-time Value (Rs)': 'Market_Value'}).copy()
    
    # Add mock columns if missing, as the Flask prep function expects them
    if 'Avg_Buy_Price' not in df_send.columns: df_send['Avg_Buy_Price'] = df_send['LTP'] * 0.95 # Mock 5% PnL
    if 'Beta' not in df_send.columns: df_send['Beta'] = 1.0 # Mock Beta
    if 'Avg_Volume' not in df_send.columns: df_send['Avg_Volume'] = 1000000 # Mock Volume
    if 'Volatility' not in df_send.columns: df_send['Volatility'] = 0.20 # Mock Volatility
    if 'Industry' not in df_send.columns: df_send['Industry'] = 'UNKNOWN'

    df_send['Quantity'] = pd.to_numeric(df_send['Quantity'], errors='coerce').fillna(0).astype(int)
    
    payload = {"portfolio": df_send.to_dict('records')}
    # The Flask endpoint is /analytics/comprehensive
    return call_compliance_api("/analytics/comprehensive", payload)

def call_optimization_api(portfolio_df: pd.DataFrame):
    """Calls the new optimization endpoint (/api/v2/simulation/optimize)."""
    # Ensure portfolio sent has LTP, Volatility
    df_send = portfolio_df.copy()
    if 'Volatility' not in df_send.columns: df_send['Volatility'] = 0.20 
    
    payload = {"portfolio": df_send.to_dict('records')}
    return call_compliance_api("/simulation/optimize", payload)

def call_correlation_api(portfolio_df: pd.DataFrame):
    """Calls the new correlation endpoint (/api/v2/analytics/correlation)."""
    # Only need Symbol and Industry for the Flask implementation
    df_send = portfolio_df.copy()
    if 'Industry' not in df_send.columns: df_send['Industry'] = 'UNKNOWN'
    
    payload = {"portfolio": df_send.to_dict('records')}
    return call_compliance_api("/analytics/correlation", payload)

def call_volatility_cone_api(portfolio_df: pd.DataFrame):
    """Calls the new volatility cone endpoint (/api/v2/simulation/volatility_cone)."""
    # Need LTP, Quantity, Volatility
    df_send = portfolio_df.copy()
    if 'Volatility' not in df_send.columns: df_send['Volatility'] = 0.20 
    
    payload = {"portfolio": df_send.to_dict('records')}
    return call_compliance_api("/simulation/volatility_cone", payload)

def call_stress_test_api(portfolio_df: pd.DataFrame, scenarios: dict):
    """Calls the new stress test endpoint (/api/v2/simulation/stress_test)."""
    # Need LTP, Quantity, Industry
    df_send = portfolio_df.copy()
    if 'Industry' not in df_send.columns: df_send['Industry'] = 'UNKNOWN'
    
    payload = {
        "portfolio": df_send.to_dict('records'),
        "scenarios": scenarios
    }
    return call_compliance_api("/simulation/stress_test", payload)

def call_rebalance_api_full(portfolio_df: pd.DataFrame, target_model: list):
    """Calls the new rebalance endpoint (/api/v2/simulation/rebalance)."""
    # Need Symbol, LTP, Quantity, Weight %
    df_send = portfolio_df.copy()
    
    payload = {
        "portfolio": df_send.to_dict('records'),
        "target_model": target_model
    }
    return call_compliance_api("/simulation/rebalance", payload)

# Note: The Monte Carlo API function is now REMOVED as it was not present in the provided Flask code.
# The compliance rules check logic from the old Streamlit code is now replaced with a mock, 
# relying on the /analytics/comprehensive endpoint for *actual* risk metrics.


def calculate_security_level_compliance(portfolio_df: pd.DataFrame, threshold_configs: dict):
    """Enhanced security compliance with more thresholds"""
    if portfolio_df.empty:
        return pd.DataFrame()
    
    security_compliance = portfolio_df.copy()
    single_stock_limit = threshold_configs.get('single_stock_limit', 10.0)
    max_single_holding = threshold_configs.get('max_single_holding', 10.0)
    
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

# Remove the old local calculate_advanced_metrics function as it is now replaced by the API call

# --- Stress Testing Functions (UPDATED to use API) ---
def run_stress_test_api_wrapper(original_df, scenario_type, params):
    """
    Translates Streamlit parameters into the API's scenarios structure
    and calls the /simulation/stress_test endpoint.
    """
    scenarios = {}
    scenario_name = f"{scenario_type}_{datetime.now().strftime('%H%M%S')}"

    if scenario_type == "Market Crash":
        shock_pct = -params['percentage'] / 100.0
        scenarios[scenario_name] = shock_pct

    elif scenario_type == "Sector Shock":
        shock_pct = -params['percentage'] / 100.0
        sector = params['sector'].upper()
        # API requires a dictionary mapping sectors to shocks
        scenarios[scenario_name] = {sector: shock_pct}
    
    elif scenario_type == "Single Stock Failure":
        shock_pct = -params['percentage'] / 100.0
        # The Flask API implementation of stress test only supports global or sector shocks.
        # We must mock the single stock failure or treat it as a tiny sector shock if possible.
        # For simplicity and sticking to the provided Flask API's structure, we'll
        # treat this as a generic market crash if the symbol's industry isn't available.
        
        symbol = params['symbol']
        sector = original_df[original_df['Symbol'] == symbol]['Industry'].iloc[0].upper()
        
        # We create a specific sector shock for the sector of the failing stock
        scenarios[scenario_name] = {sector: shock_pct}
        scenario_name = f"Stock_Failure_{symbol}" # Rename for clarity
        st.warning(f"Note: Single stock failure approximated as a {shock_pct*100:.0f}% shock to the {sector} sector.")
        
    else:
        return None, {"error": "Invalid scenario type."}
            
    api_response = call_stress_test_api(original_df, scenarios)
    
    if api_response and 'stress_test_results' in api_response and scenario_name in api_response['stress_test_results']:
        summary = api_response['stress_test_results'][scenario_name]
        
        # We need to calculate the stressed DF client-side since the API only returns summary
        stressed_total_value = summary['post_stress_value']
        original_total_value = original_df['Real-time Value (Rs)'].sum()

        stressed_df = original_df.copy()
        
        if scenario_type == "Market Crash":
            shock_factor = 1 + scenarios[scenario_name]
            stressed_df['Stressed Value (Rs)'] = stressed_df['Real-time Value (Rs)'] * shock_factor
        elif scenario_type == "Sector Shock":
            sector_shock_map = scenarios[scenario_name]
            shock_sector = list(sector_shock_map.keys())[0]
            shock_factor = 1 + sector_shock_map[shock_sector]
            stressed_df['Stressed Value (Rs)'] = stressed_df.apply(
                lambda row: row['Real-time Value (Rs)'] * shock_factor if row['Industry'] == shock_sector else row['Real-time Value (Rs)'],
                axis=1
            )
        elif scenario_type == "Single Stock Failure":
             # Use the approximation logic (sector shock) applied above
            sector_shock_map = scenarios[scenario_name]
            shock_sector = list(sector_shock_map.keys())[0]
            shock_factor = 1 + shock_sector # shock_sector is the sector shock factor applied above
            stressed_df['Stressed Value (Rs)'] = stressed_df.apply(
                lambda row: row['Real-time Value (Rs)'] * shock_factor if row['Industry'] == shock_sector else row['Real-time Value (Rs)'],
                axis=1
            )

        # Recalculate weights based on new stressed values
        stressed_df['Stressed Weight %'] = (stressed_df['Stressed Value (Rs)'] / stressed_total_value * 100) if stressed_total_value > 0 else 0

        # Enhance summary structure
        enhanced_summary = {
            "original_value": original_total_value,
            "stressed_value": stressed_total_value,
            "loss_value": summary['change_amount'] * -1, # API returns change amount (negative for loss)
            "loss_pct": summary['change_pct'] * -1,
        }

        return stressed_df, enhanced_summary

    return None, {"error": "API response incomplete or failed."}


# --- AI Analysis Functions ---
def extract_text_from_files(uploaded_files):
    # ... (No change)
    full_text = ""
    for file in uploaded_files:
        full_text += f"\n\n--- DOCUMENT: {file.name} ---\n\n"
        if file.type == "application/pdf":
            try:
                with fitz.open(stream=file.getvalue(), filetype="pdf") as doc:
                    for page in doc:
                        full_text += page.get_text()
            except NameError:
                full_text += "(PyMuPDF not imported - cannot extract PDF text)"
        else:
            full_text += file.getvalue().decode("utf-8")
    return full_text

def get_portfolio_summary(df):
    # ... (No change)
    if df.empty:
        return "No portfolio data."
    
    total_value = df['Real-time Value (Rs)'].sum()
    top_10_stocks = df.nlargest(10, 'Weight %')[['Name', 'Weight %']]
    sector_weights = df.groupby('Industry')['Weight %'].sum().nlargest(10)
    
    summary = f"""**Portfolio Snapshot**

- **Total Value:** ₹ {total_value:,.2f}
- **Holdings:** {len(df)}
- **Top Stock:** {df['Weight %'].max():.2f}%

**Top 10 Holdings:**
"""
    for _, row in top_10_stocks.iterrows():
        summary += f"- {row['Name']}: {row['Weight %']:.2f}%\n"
    
    summary += "\n**Top Sectors:**\n"
    for sector, weight in sector_weights.items():
        summary += f"- {sector}: {weight:.2f}%\n"
    
    return summary


def render_portfolio_card(portfolio):
    # ... (No change)
    """Helper function to render portfolio card in history"""
    portfolio_name = portfolio.get('portfolio_name', 'Unnamed Portfolio')
    analysis_date = datetime.fromisoformat(portfolio['created_at'])
    
    with st.container():
        col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
        
        with col1:
            st.markdown(f"**📁 {portfolio_name}**")
            st.caption(f"{analysis_date.strftime('%Y-%m-%d %H:%M')}")
            
            if portfolio.get('metadata'):
                metadata = portfolio['metadata']
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
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
                        elif loaded['security_compliance']:
                            st.session_state["security_level_compliance"] = pd.DataFrame(loaded['security_compliance'])
                    
                    if loaded.get('breach_alerts'):
                        st.session_state["breach_alerts"] = loaded['breach_alerts']
                    
                    if loaded.get('advanced_metrics'):
                        # Load all API risk data into comprehensive_analytics_data state
                        st.session_state["comprehensive_analytics_data"] = loaded['advanced_metrics']
                    
                    if loaded.get('ai_analysis'):
                        st.session_state["ai_analysis_response"] = loaded['ai_analysis']
                    
                    if loaded.get('kim_document'):
                        st.session_state["kim_documents"][loaded['portfolio_name']] = loaded['kim_document']

                    # Clear simulation states
                    st.session_state["stress_summary"] = None
                    st.session_state["stressed_df"] = None
                    st.session_state["stressed_compliance_results"] = None
                    st.session_state["optimization_results"] = None
                    st.session_state["rebalance_suggestions"] = None
                    st.session_state["correlation_matrix_data"] = None
                    st.session_state["volatility_cone_data"] = None
                    st.session_state["stress_test_api_results"] = None
                    
                    st.success("✅ Portfolio Loaded!")
                    time.sleep(0.5)
                    st.rerun()
        
        with col4:
            if st.button("🗑️", key=f"delete_hist_{portfolio['id']}", use_container_width=True, help="Delete Portfolio"):
                if delete_portfolio(portfolio['id']):
                    st.success("Deleted!")
                    st.session_state["saved_analyses"] = get_user_portfolios(st.session_state["user_id"])
                    time.sleep(0.5)
                    st.rerun()
        
        st.markdown("---")


# --- Authentication UI ---
def render_auth_page():
    # ... (No change)
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
    # ... (No change)
    """Render comprehensive threshold configuration panel"""
    st.subheader("⚙️ Compliance Thresholds")
    
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
st.title("Invsion Connect Pro")
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
            
            with st.expander(f"{stage_emoji} {portfolio_name}"):
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
                            elif loaded['security_compliance']:
                                st.session_state["security_level_compliance"] = pd.DataFrame(loaded['security_compliance'])
                        
                        if loaded.get('breach_alerts'):
                            st.session_state["breach_alerts"] = loaded['breach_alerts']
                        
                        if loaded.get('advanced_metrics'):
                            # Load all API risk data into comprehensive_analytics_data state
                            st.session_state["comprehensive_analytics_data"] = loaded['advanced_metrics']
                        
                        if loaded.get('ai_analysis'):
                            st.session_state["ai_analysis_response"] = loaded['ai_analysis']
                        
                        if loaded.get('kim_document'):
                            st.session_state["kim_documents"][loaded['portfolio_name']] = loaded['kim_document']

                        # Clear simulation states
                        st.session_state["stress_summary"] = None
                        st.session_state["stressed_df"] = None
                        st.session_state["stressed_compliance_results"] = None
                        st.session_state["optimization_results"] = None
                        st.session_state["rebalance_suggestions"] = None
                        st.session_state["correlation_matrix_data"] = None
                        st.session_state["volatility_cone_data"] = None
                        st.session_state["stress_test_api_results"] = None
                        
                        st.success("Loaded!")
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

# Added Tabs for Advanced API Features
tabs = st.tabs(["💼 Portfolio Analysis", "🤖 AI Analysis", "⚡ Stress Testing", "🔧 API Interactions", "📊 Advanced Analytics", "📚 History"])


# --- TAB 1: Enhanced Compliance Analysis ---
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
                # Clear all simulation/API states
                st.session_state["stress_summary"] = None
                st.session_state["stressed_df"] = None
                st.session_state["stressed_compliance_results"] = None
                st.session_state["comprehensive_analytics_data"] = None
                st.session_state["optimization_results"] = None
                st.session_state["rebalance_suggestions"] = None
                st.session_state["correlation_matrix_data"] = None
                st.session_state["volatility_cone_data"] = None
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
        **Supported Rule Types (Simulated):**
        - `STOCK <SYMBOL> <op> <value>` - Single stock weight
        - `SECTOR <NAME> <op> <value>` - Sector weight
        - `TOP_N_STOCKS <N> <op> <value>` - Top N stocks concentration
        - `HHI <op> <value>` - Herfindahl-Hirschman Index (Mocked/Calculated client-side)
        
        **Operators:** `<`, `>`, `<=`, `>=`, `=`
        """)
        
        default_rules = st.session_state.get("current_rules_text", """# SEBI Compliance Rules
STOCK RELIANCE < 10
STOCK TCS < 10
SECTOR BANKING < 25
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
            help="Define your compliance rules here"
        )
    
    st.markdown("---")
    
    # Step 3: Analyze / Refresh LTP
    col_analyze, col_refresh = st.columns(2)

    with col_analyze:
        if uploaded_file and k:
            if st.button("🔍 Analyze Compliance & Risk", type="primary", use_container_width=True, key="analyze_btn"):
                with st.spinner("Analyzing portfolio compliance and fetching risk data..."):
                    try:
                        # 1. Read CSV and prepare DF
                        df = pd.read_csv(uploaded_file)
                        df.columns = [str(col).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_') for col in df.columns]
                        
                        header_map = {
                            'symbol': 'Symbol',
                            'industry': 'Industry',
                            'quantity': 'Quantity',
                            'name_of_the_instrument': 'Name',
                            'avg_buy_price': 'Avg_Buy_Price' # Added for API
                        }
                        df = df.rename(columns=header_map)
                        
                        # Add required columns if missing
                        if 'Industry' not in df.columns: df['Industry'] = 'UNKNOWN'
                        if 'Avg_Buy_Price' not in df.columns: df['Avg_Buy_Price'] = 0.0 # Mock Buy Price

                        df['Industry'] = df['Industry'].fillna('UNKNOWN').str.strip().str.upper()
                        if 'Name' not in df.columns: df['Name'] = df['Symbol'] 
                        if 'LTP' not in df.columns: df['LTP'] = 0.0
                        if 'Beta' not in df.columns: df['Beta'] = 1.0 # Mock Beta
                        if 'Volatility' not in df.columns: df['Volatility'] = 0.20 # Mock Volatility
                        if 'Avg_Volume' not in df.columns: df['Avg_Volume'] = 1000000 # Mock Volume

                        # 2. Fetch real-time prices
                        symbols = df['Symbol'].unique().tolist()
                        prices = {sym: get_ltp_price_fresh(api_key, access_token, sym) for sym in symbols}
                        
                        df_results = df.copy()
                        df_results['LTP'] = df_results['Symbol'].map(prices)
                        df_results['LTP'] = pd.to_numeric(df_results['LTP'], errors='coerce').fillna(0)
                        df_results['Real-time Value (Rs)'] = (df_results['LTP'] * pd.to_numeric(df_results['Quantity'], errors='coerce')).fillna(0)
                        total_value = df_results['Real-time Value (Rs)'].sum()
                        df_results['Weight %'] = (df_results['Real-time Value (Rs)'] / total_value * 100) if total_value > 0 else 0
                        
                        # 3. Call API for Comprehensive Analytics (Risk, Liquidity, Tax)
                        comprehensive_data = call_comprehensive_analytics_api(df_results)
                        st.session_state["comprehensive_analytics_data"] = comprehensive_data

                        # 4. Call API Mock for Custom Rules validation (Rules Engine)
                        compliance_results = call_compliance_api_run_check(df_results, rules_text, st.session_state["threshold_configs"])
                        
                        # 5. Calculate security-level compliance (local function)
                        security_compliance = calculate_security_level_compliance(df_results, st.session_state["threshold_configs"])
                        
                        # 6. Store in session state
                        st.session_state.compliance_results_df = df_results
                        st.session_state.security_level_compliance = security_compliance
                        st.session_state.compliance_results = compliance_results
                        st.session_state.current_rules_text = rules_text
                        st.session_state.current_portfolio_name = portfolio_name
                        
                        # 7. Detect breaches (combining local limits and custom rule failures)
                        breaches = []
                        
                        # Stock limit breaches
                        single_stock_limit = st.session_state["threshold_configs"]['single_stock_limit']
                        if (df_results['Weight %'] > single_stock_limit).any():
                            breach_stocks = df_results[df_results['Weight %'] > single_stock_limit]
                            for _, stock in breach_stocks.iterrows():
                                breaches.append({
                                    'type': 'Single Stock Limit',
                                    'severity': '🔴 Critical',
                                    'details': f"{stock['Symbol']} at {stock['Weight %']:.2f}% (Limit: {single_stock_limit}%)"
                                })
                        
                        # Sector limit breaches
                        if 'Industry' in df_results.columns:
                            sector_weights = df_results.groupby('Industry')['Weight %'].sum()
                            single_sector_limit = st.session_state["threshold_configs"]['single_sector_limit']
                            if (sector_weights > single_sector_limit).any():
                                breach_sectors = sector_weights[sector_weights > single_sector_limit]
                                for sector, weight in breach_sectors.items():
                                    breaches.append({
                                        'type': 'Sector Limit',
                                        'severity': '🟠 High',
                                        'details': f"{sector} at {weight:.2f}% (Limit: {single_sector_limit}%)"
                                    })
                        
                        # Custom rule failures (from API Mock)
                        for rule_result in compliance_results:
                            if rule_result['status'] == "FAIL":
                                severity = "🟡 Medium" 
                                if abs(rule_result.get('breach_amount', 0)) > rule_result.get('threshold', 0) * 0.2:
                                    severity = "🔴 Critical"
                                elif abs(rule_result.get('breach_amount', 0)) > rule_result.get('threshold', 0) * 0.1:
                                    severity = "🟠 High"
                                
                                breaches.append({
                                    'type': 'Custom Rule Violation',
                                    'severity': severity,
                                    'details': f"{rule_result['rule']} - {rule_result['details']}"
                                })
                        
                        # Portfolio structure checks
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
                        
                        if 'Industry' in df_results.columns:
                            sector_count = df_results['Industry'].nunique()
                            if sector_count < st.session_state["threshold_configs"]['min_sectors']:
                                breaches.append({
                                    'type': 'Min Sectors',
                                    'severity': '🟠 High',
                                    'details': f"Only {sector_count} sectors (Min: {st.session_state['threshold_configs']['min_sectors']})"
                                })
                        
                        st.session_state.breach_alerts = breaches
                        st.session_state.compliance_stage = "compliance_done"
                        
                        # 8. Save to database
                        portfolio_data = {
                            'holdings_data': df_results.to_json(),
                            'total_value': float(total_value),
                            'holdings_count': len(df_results),
                            'metadata': {
                                'total_value': float(total_value),
                                'holdings_count': len(df_results),
                                'analysis_timestamp': datetime.now().isoformat(),
                                'last_ltp_refresh': datetime.now().isoformat()
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
                                'security_compliance': security_compliance.to_json(),
                                'breach_alerts': breaches,
                                'advanced_metrics': comprehensive_data, # Save comprehensive API data
                                'ai_analysis': None
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
                        st.error(f"Error: {e}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
        
    with col_refresh:
        # Add a refresh LTP button
        if k and not st.session_state.get("compliance_results_df", pd.DataFrame()).empty:
            if st.button("🔄 Refresh LTP Data", type="secondary", use_container_width=True, key="refresh_ltp_btn"):
                # --- LTP Refresh logic (identical to Step 3 in Analyze) ---
                with st.spinner("Fetching latest LTPs and re-calculating portfolio..."):
                    df_to_refresh = st.session_state.compliance_results_df.copy()
                    symbols = df_to_refresh['Symbol'].unique().tolist()
                    
                    prices = {}
                    for symbol in symbols:
                        ltp = get_ltp_price_fresh(api_key, access_token, symbol)
                        prices[symbol] = ltp
                        
                    df_to_refresh['LTP'] = df_to_refresh['Symbol'].map(prices)
                    df_to_refresh['LTP'] = pd.to_numeric(df_to_refresh['LTP'], errors='coerce').fillna(0) 
                    
                    df_to_refresh['Real-time Value (Rs)'] = (df_to_refresh['LTP'] * pd.to_numeric(df_to_refresh['Quantity'], errors='coerce')).fillna(0)
                    total_value = df_to_refresh['Real-time Value (Rs)'].sum()
                    df_to_refresh['Weight %'] = (df_to_refresh['Real-time Value (Rs)'] / total_value * 100) if total_value > 0 else 0
                    
                    # Re-run compliance and risk calls
                    comprehensive_data = call_comprehensive_analytics_api(df_to_refresh)
                    compliance_results = call_compliance_api_run_check(df_to_refresh, rules_text, st.session_state["threshold_configs"])
                    security_compliance = calculate_security_level_compliance(df_to_refresh, st.session_state["threshold_configs"])
                    
                    # Update session state
                    st.session_state.compliance_results_df = df_to_refresh
                    st.session_state.security_level_compliance = security_compliance
                    st.session_state.compliance_results = compliance_results
                    st.session_state["comprehensive_analytics_data"] = comprehensive_data
                    
                    # Detect breaches (logic copied from Analyze for consistency)
                    breaches = []
                    single_stock_limit = st.session_state["threshold_configs"]['single_stock_limit']
                    if (df_to_refresh['Weight %'] > single_stock_limit).any():
                        breach_stocks = df_to_refresh[df_to_refresh['Weight %'] > single_stock_limit]
                        for _, stock in breach_stocks.iterrows():
                            breaches.append({'type': 'Single Stock Limit', 'severity': '🔴 Critical', 'details': f"{stock['Symbol']} at {stock['Weight %']:.2f}% (Limit: {single_stock_limit}%)"})
                    if 'Industry' in df_to_refresh.columns:
                        sector_weights = df_to_refresh.groupby('Industry')['Weight %'].sum()
                        single_sector_limit = st.session_state["threshold_configs"]['single_sector_limit']
                        if (sector_weights > single_sector_limit).any():
                            breach_sectors = sector_weights[sector_weights > single_sector_limit]
                            for sector, weight in breach_sectors.items():
                                breaches.append({'type': 'Sector Limit', 'severity': '🟠 High', 'details': f"{sector} at {weight:.2f}% (Limit: {single_sector_limit}%)"})
                    for rule_result in compliance_results:
                        if rule_result['status'] == "FAIL":
                            severity = "🟡 Medium" 
                            if abs(rule_result.get('breach_amount', 0)) > rule_result.get('threshold', 0) * 0.2:
                                severity = "🔴 Critical"
                            elif abs(rule_result.get('breach_amount', 0)) > rule_result.get('threshold', 0) * 0.1:
                                severity = "🟠 High"
                            breaches.append({'type': 'Custom Rule Violation', 'severity': severity, 'details': f"{rule_result['rule']} - {rule_result['details']}"})
                    if len(df_to_refresh) < st.session_state["threshold_configs"]['min_holdings']:
                        breaches.append({'type': 'Min Holdings', 'severity': '🟡 Medium', 'details': f"Only {len(df_to_refresh)} holdings (Min: {st.session_state['threshold_configs']['min_holdings']})"})
                    if len(df_to_refresh) > st.session_state["threshold_configs"]['max_holdings']:
                        breaches.append({'type': 'Max Holdings', 'severity': '🟡 Medium', 'details': f"{len(df_to_refresh)} holdings (Max: {st.session_state['threshold_configs']['max_holdings']})"})
                    if 'Industry' in df_to_refresh.columns:
                        sector_count = df_to_refresh['Industry'].nunique()
                        if sector_count < st.session_state["threshold_configs"]['min_sectors']:
                            breaches.append({'type': 'Min Sectors', 'severity': '🟠 High', 'details': f"Only {sector_count} sectors (Min: {st.session_state['threshold_configs']['min_sectors']})"})
                    st.session_state.breach_alerts = breaches

                    # Update saved portfolio in DB
                    if st.session_state.get("current_portfolio_id"):
                        total_value = df_to_refresh['Real-time Value (Rs)'].sum()
                        portfolio_data = {
                            'holdings_data': df_to_refresh.to_json(),
                            'total_value': float(total_value),
                            'holdings_count': len(df_to_refresh),
                            'metadata': {
                                'total_value': float(total_value),
                                'holdings_count': len(df_to_refresh),
                                'analysis_timestamp': datetime.now().isoformat(),
                                'last_ltp_refresh': datetime.now().isoformat()
                            }
                        }
                        success_update, _ = save_portfolio_with_stages(st.session_state["user_id"], portfolio_name, portfolio_data, "compliance_done")
                        if success_update:
                            compliance_data = {
                                'threshold_configs': st.session_state["threshold_configs"],
                                'custom_rules': rules_text,
                                'compliance_results': compliance_results,
                                'security_compliance': security_compliance.to_json(),
                                'breach_alerts': breaches,
                                'advanced_metrics': comprehensive_data, # Save new comprehensive API data
                                'ai_analysis': st.session_state.get("ai_analysis_response")
                            }
                            save_compliance_analysis(st.session_state["user_id"], st.session_state["current_portfolio_id"], compliance_data)
                            st.success("✅ LTP data refreshed and portfolio updated in database!")
                            st.session_state["saved_analyses"] = get_user_portfolios(st.session_state["user_id"])
                        else:
                            st.warning("⚠️ LTP refresh successful but failed to update portfolio in database.")
                    else:
                        st.info("Portfolio not saved in database. Refresh applied to current view.")
                    
                    time.sleep(1)
                    st.rerun()
        elif not k:
            st.info("Connect to Kite to enable LTP refresh.")
        elif st.session_state.get("compliance_results_df", pd.DataFrame()).empty:
            st.info("Upload and analyze a portfolio first to enable LTP refresh.")
    
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
        
        analysis_tabs = st.tabs([
            "📊 Dashboard",
            "🔍 Details",
            "📈 Risk & Liquidity", # Renamed tab
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
        
        with analysis_tabs[2]:
            st.subheader("Risk, Liquidity, and Tax Metrics (via API)")
            
            risk_data = st.session_state.get("comprehensive_analytics_data")
            
            if risk_data:
                # 1. Risk Metrics (VaR, Beta)
                st.markdown("### Risk Metrics")
                risk_metrics = risk_data.get('risk_metrics', {})
                risk_cols = st.columns(3)
                risk_cols[0].metric("VaR (95%, 1 Day)", f"₹ {risk_metrics['VaR_95_1Day']:,.0f}")
                risk_cols[1].metric("VaR (99%, 10 Day)", f"₹ {risk_metrics['VaR_99_10Day']:,.0f}")
                risk_cols[2].metric("Portfolio Beta", f"{risk_metrics['Portfolio_Beta']:.2f}")

                # 2. Liquidity Analysis
                st.markdown("### Liquidity Analysis")
                liquidity = risk_data.get('liquidity_analysis', {})
                liquidity_cols = st.columns(2)
                liquidity_cols[0].metric("Max Days to Liquidate", f"{liquidity['max_days_to_liquidate']:.1f} days")
                if liquidity['illiquid_assets']:
                    liquidity_cols[1].error(f"{len(liquidity['illiquid_assets'])} Illiquid Assets")
                    st.expander("Illiquid Assets Details").dataframe(pd.DataFrame(liquidity['illiquid_assets']), use_container_width=True)
                else:
                    liquidity_cols[1].success("No Illiquid Assets detected (Max 3 days)")

                # 3. Tax/PnL Simulation
                st.markdown("### Tax Impact / P&L Simulation")
                tax_sim = risk_data.get('tax_simulation', {})
                tax_cols = st.columns(3)
                tax_cols[0].metric("Total Invested", f"₹ {tax_sim['total_invested']:,.0f}")
                tax_cols[1].metric("Unrealized PnL", f"₹ {tax_sim['total_unrealized_pnl']:,.0f}")
                tax_cols[2].metric("ROI (%)", f"{tax_sim['return_on_investment_pct']:.2f}%")
            else:
                st.info("Risk data not yet available. Click 'Analyze Compliance & Risk' to fetch.")

        with analysis_tabs[3]:
            # ... (Rules Tab - No change, uses mocked compliance_results) ...
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
        
        with analysis_tabs[4]:
            # ... (Security Tab - No change) ...
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
        
        with analysis_tabs[5]:
            # ... (Concentration Tab - No change, uses local calculation) ...
            st.subheader("Concentration Analysis")
            
            sorted_df = results_df.sort_values('Weight %', ascending=False).reset_index(drop=True)
            sorted_df['Cumulative Weight %'] = sorted_df['Weight %'].cumsum()
            sorted_df['Rank'] = range(1, len(sorted_df) + 1)
            
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
            bench_cols[0].metric("Top 1", f"{sorted_df.iloc[0]['Weight %']:.2f}%")
            bench_cols[1].metric("Top 3", f"{sorted_df.head(3)['Weight %'].sum():.2f}%")
            bench_cols[2].metric("Top 5", f"{sorted_df.head(5)['Weight %'].sum():.2f}%")
            bench_cols[3].metric("Top 10", f"{sorted_df.head(10)['Weight %'].sum():.2f}%")
            bench_cols[4].metric("Top 20", f"{sorted_df.head(20)['Weight %'].sum():.2f}%" if len(sorted_df) >= 20 else "N/A")
            
            # HHI and Gini
            hhi = (results_df['Weight %'] ** 2).sum()
            weights_sorted = results_df['Weight %'].sort_values().values
            n = len(weights_sorted)
            gini = 0 
            if n > 0:
                sum_weights = np.sum(weights_sorted)
                if sum_weights > 0:
                    # Gini calculation corrected slightly for stability
                    gini = (2 * np.sum((np.arange(1, n+1)) * weights_sorted)) / (n * sum_weights) - (n + 1) / n
                    if gini < 0: gini = 0 # Ensure Gini is non-negative
            
            st.markdown("### Concentration Indices")
            index_cols = st.columns(2)
            index_cols[0].metric("HHI (Herfindahl-Hirschman)", f"{hhi:.2f}", help="Lower is more diversified. <1000 is good")
            index_cols[1].metric("Gini Coefficient", f"{gini:.4f}", help="0=perfect equality, 1=maximum inequality")
        
        with analysis_tabs[6]:
            # ... (Report Tab - No change) ...
            st.subheader("Export Report")
            
            if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
                from io import BytesIO
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='openyxl') as writer:
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


# --- TAB 2: Enhanced AI Analysis ---
with tabs[1]:
    st.header("🤖 AI-Powered Compliance Analysis")
    
    portfolio_df = st.session_state.get("compliance_results_df")
    current_portfolio_name = st.session_state.get("current_portfolio_name")
    
    if not current_portfolio_name:
        st.warning("⚠️ Please create/load a portfolio first in the Portfolio Analysis tab")
        # st.stop() # Removed stop to prevent breaking the flow if running locally without full setup
    elif portfolio_df is None or portfolio_df.empty:
        st.warning("⚠️ Please analyze portfolio compliance first in the Portfolio Analysis tab")
        # st.stop()
    elif st.session_state.get("compliance_stage") != "compliance_done" and st.session_state.get("compliance_stage") != "ai_completed":
        st.warning("⚠️ Complete compliance analysis first")
        # st.stop()
    else: # Only proceed if data is ready
        st.info(f"📁 **Portfolio:** {current_portfolio_name}")
        
        # Check if KIM document already exists
        existing_kim = get_kim_document(st.session_state["user_id"], current_portfolio_name)
        
        docs_text = None
        if existing_kim:
            # Display existing document info (No Change)
            st.success(f"✅ KIM/SID document already uploaded: **{existing_kim['file_name']}**")
            st.caption(f"Extracted on: {datetime.fromisoformat(existing_kim['extracted_at']).strftime('%Y-%m-%d %H:%M')}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.checkbox("📄 View document excerpt", key="view_kim"):
                    st.text_area("Document Text (first 2000 chars)", existing_kim['document_text'][:2000], height=200, disabled=True)
            with col2:
                if st.button("🗑️ Delete & Re-upload", use_container_width=True):
                    if type(supabase).__name__ != 'DummySupabaseClient':
                        supabase.table('kim_documents').delete().eq('id', existing_kim['id']).execute()
                    st.success("Deleted! Please upload new document.")
                    time.sleep(0.5)
                    st.rerun()
            
            docs_text = existing_kim['document_text']
        else:
            # Document upload logic (No Change)
            st.subheader("Step 1: Upload KIM/SID Documents")
            uploaded_docs = st.file_uploader(
                "📄 Upload Scheme Documents (PDF/TXT)",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Upload Key Information Memorandum or Scheme Information Document"
            )
            
            if uploaded_docs:
                st.success(f"✅ {len(uploaded_docs)} document(s) uploaded")
                
                if st.button("💾 Extract & Save Documents", type="secondary", use_container_width=True):
                    with st.spinner("Extracting text from documents..."):
                        docs_text = extract_text_from_files(uploaded_docs)
                        file_names = ", ".join([f.name for f in uploaded_docs])
                        
                        success, doc_id = save_kim_document(
                            st.session_state["user_id"],
                            current_portfolio_name,
                            docs_text,
                            file_names
                        )
                        
                        if success:
                            st.success("✅ Documents extracted and saved!")
                            st.session_state["kim_documents"][current_portfolio_name] = {
                                'document_text': docs_text,
                                'file_name': file_names
                            }
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Failed to save documents")
                
                # If extraction was run and successful, docs_text will be available next rerun
            
        st.markdown("---")
        
        # AI Analysis Configuration and Run (No major change, relying on genai import)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Step 2: Configure AI Analysis")
            analysis_depth = st.select_slider(
                "Analysis Depth",
                options=["Quick", "Standard", "Comprehensive"],
                value="Standard",
                help="Quick: Fast overview | Standard: Detailed analysis | Comprehensive: Deep dive with recommendations"
            )
        
        with col2:
            st.subheader("Analysis Options")
            include_market_context = st.checkbox("Include Market Context", value=True)
            include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        st.markdown("---")

        if (docs_text or existing_kim) or st.session_state.get("ai_analysis_response"):
            if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True, key="ai_analyze_btn"):
                if 'genai' not in globals():
                    st.error("AI engine is unavailable (Module not imported).")
                else:
                    with st.spinner("🤖 AI is analyzing your portfolio..."):
                        try:
                            if existing_kim: docs_text = existing_kim['document_text']
                            
                            portfolio_summary = get_portfolio_summary(portfolio_df)
                            breach_alerts = st.session_state.get("breach_alerts", [])
                            breach_summary = "\n".join([f"- {b['type']}: {b['details']}" for b in breach_alerts]) if breach_alerts else "No breaches detected."
                            
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
                            
                            threshold_summary = "\n**Threshold Configurations:**\n"
                            for key, value in st.session_state["threshold_configs"].items():
                                threshold_summary += f"- {key}: {value}\n"
                            
                            # --- Prompt Template Selection (Same as before) ---
                            if analysis_depth == "Quick":
                                max_tokens = 8000
                                prompt_template = """... (Quick Template) ..."""
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
                                prompt_template = """... (Standard Template) ..."""
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
                                prompt_template = """... (Comprehensive Template) ..."""
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
- Portfolio construction quality

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
- Stock selection quality
- Hidden risks (correlated positions, cyclical exposure)

## 5. Violations & Regulatory Concerns (400 words)
Detailed list with:
- Violation type
- Severity (Critical/High/Medium/Low)
- Current vs Limit
- Potential regulatory implications
- Remediation complexity

## 6. Industry Best Practices Comparison (300 words)
- Peer fund comparison
- Industry benchmarks
- Best-in-class examples

## 7. Actionable Recommendations (500 words)

### Immediate Actions (0-30 days)
- Critical breach remediation
- Specific trades to suggest

### Short-term Actions (1-3 months)
- Portfolio rebalancing strategy
- Risk reduction measures

### Long-term Improvements (3-12 months)
- Process improvements
- Systematic risk management

## 8. Compliance Roadmap (200 words)
Step-by-step plan with timelines

## 9. Monitoring & Controls (200 words)
Suggested ongoing compliance framework

## 10. Disclaimers & Assumptions (100 words)
Data limitations and assumptions made

**IMPORTANT:**
- Be specific with numbers and percentages
- Cite exact regulations where applicable
- Provide actionable, practical advice
- Highlight both immediate and strategic concerns
- Use clear severity classifications"""

                            
                            # Truncate docs_text for Gemini input
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
                            
                            # Save to database (No Change)
                            if st.session_state.get("current_portfolio_id"):
                                compliance_data = {
                                    'threshold_configs': st.session_state["threshold_configs"],
                                    'custom_rules': st.session_state.get("current_rules_text", ""),
                                    'compliance_results': st.session_state.get("compliance_results", []),
                                    'security_compliance': st.session_state.get("security_level_compliance", pd.DataFrame()).to_json(),
                                    'breach_alerts': st.session_state.get("breach_alerts", []),
                                    'advanced_metrics': st.session_state.get("comprehensive_analytics_data"),
                                    'ai_analysis': response.text
                                }
                                
                                save_compliance_analysis(st.session_state["user_id"], st.session_state["current_portfolio_id"], compliance_data)
                                
                                supabase.table('portfolios').update({'analysis_stage': 'ai_completed'}).eq('id', st.session_state["current_portfolio_id"]).execute()
                                
                                st.success("✅ AI Analysis Complete and Saved!")
                                
                                st.session_state["saved_analyses"] = get_user_portfolios(st.session_state["user_id"])
                                time.sleep(1)
                                st.rerun()
                        
                        except Exception as e:
                            st.error(f"❌ AI Analysis Error: {e}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
            else:
                st.info("Upload KIM/SID documents or select an existing one to proceed with AI Analysis.")

        # Display AI Analysis Results (No Change)
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

# --- TAB 3: Stress Testing & Audit (UPDATED to use API) ---
with tabs[2]:
    st.header("⚡ Portfolio Stress Testing & Audit")

    if 'compliance_results_df' not in st.session_state or st.session_state.compliance_results_df.empty:
        st.warning("⚠️ Please upload and analyze a portfolio in the 'Portfolio Analysis' tab first.")
    else:
        df = st.session_state.compliance_results_df
        st.info(f"Running tests on: **{st.session_state.get('current_portfolio_name', 'Unnamed Portfolio')}**")
        st.markdown("---")

        st.subheader("1. Define Stress Scenario")
        col1, col2 = st.columns([1, 2])
        with col1:
            scenario_type = st.selectbox(
                "Select a Stress Scenario",
                ["Market Crash", "Sector Shock", "Single Stock Failure"],
                key="stress_scenario_type"
            )
        
        params = {}
        with col2:
            if scenario_type == "Market Crash":
                params['percentage'] = st.slider("Market-wide Drop (%)", 5, 50, 20, help="Simulates a uniform drop across all portfolio holdings.")
            elif scenario_type == "Sector Shock":
                all_sectors = sorted(df['Industry'].unique().tolist())
                params['sector'] = st.selectbox("Select Sector to Shock", all_sectors)
                params['percentage'] = st.slider(f"Drop in {params['sector']} Sector (%)", 5, 75, 25)
            elif scenario_type == "Single Stock Failure":
                all_stocks = sorted(df['Symbol'].unique().tolist())
                params['symbol'] = st.selectbox("Select Stock to Shock", all_stocks, help="Simulate an adverse event for a single company.")
                params['percentage'] = st.slider(f"Drop in {params['symbol']} (%)", 10, 90, 50)

        if st.button("🔬 Run Stress Test", use_container_width=True, type="primary"):
            with st.spinner("Calling Stress Test API and auditing compliance..."):
                
                stressed_df, summary = run_stress_test_api_wrapper(df, scenario_type, params)
                
                if 'error' in summary:
                    st.error(f"Stress Test Failed: {summary['error']}")
                else:
                    st.session_state['stressed_df'] = stressed_df
                    st.session_state['stress_summary'] = summary
                    
                    # Re-run compliance audit on the stressed data using the MOCKED compliance API
                    stressed_df_for_api = stressed_df.rename(columns={'Stressed Weight %': 'Weight %', 'Stressed Value (Rs)': 'Real-time Value (Rs)'}).copy()
                    
                    # Ensure columns are present for the mock check
                    if 'LTP' not in stressed_df_for_api.columns: stressed_df_for_api['LTP'] = df['LTP'] 
                    if 'Quantity' not in stressed_df_for_api.columns: stressed_df_for_api['Quantity'] = df['Quantity'] 

                    stressed_compliance_results = call_compliance_api_run_check(
                        stressed_df_for_api,
                        st.session_state.current_rules_text,
                        st.session_state.threshold_configs
                    )
                    
                    st.session_state['stressed_compliance_results'] = stressed_compliance_results
                    st.success("Stress test simulation complete.")

        # --- Display Stress Test Results ---
        if 'stress_summary' in st.session_state and st.session_state['stress_summary'] is not None:
            st.markdown("---")
            st.subheader("2. Stress Test Results")
            
            summary = st.session_state['stress_summary']
            stressed_df = st.session_state['stressed_df']
            
            st.markdown("#### Impact Summary")
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
                    # Re-calculate severity based on our logic for display consistency
                    severity = "🟡 Medium" 
                    if abs(breach.get('breach_amount', 0)) > breach.get('threshold', 0) * 0.2:
                        severity = "🔴 Critical"
                    elif abs(breach.get('breach_amount', 0)) > breach.get('threshold', 0) * 0.1:
                        severity = "🟠 High"

                    breach_data.append({
                        "Rule": breach['rule'],
                        "Severity": severity, # Use calculated severity for display
                        "Details": breach['details']
                    })
                st.dataframe(pd.DataFrame(breach_data), use_container_width=True, hide_index=True)
            
            st.markdown("#### Detailed Portfolio Impact")
            display_df = stressed_df[[
                'Symbol', 'Name', 'Industry', 'Weight %', 'Stressed Weight %', 
                'Real-time Value (Rs)', 'Stressed Value (Rs)'
            ]].copy()
            display_df['Value Change (Rs)'] = display_df['Stressed Value (Rs)'] - display_df['Real-time Value (Rs)']
            display_df['Weight Change (%)'] = display_df['Stressed Weight %'] - display_df['Weight %']
            
            st.dataframe(display_df[[
                'Symbol', 'Name', 'Weight %', 'Stressed Weight %', 'Weight Change (%)',
                'Real-time Value (Rs)', 'Stressed Value (Rs)', 'Value Change (Rs)'
            ]].style.format({
                'Weight %': '{:.2f}%',
                'Stressed Weight %': '{:.2f}%',
                'Weight Change (%)': '{:+.2f}%',
                'Real-time Value (Rs)': '₹{:,.0f}',
                'Stressed Value (Rs)': '₹{:,.0f}',
                'Value Change (Rs)': '₹{:,.0f}',
            }), use_container_width=True)

            st.markdown("#### Visual Impact Analysis")
            top_15_losers = display_df.sort_values('Value Change (Rs)').head(15)
            fig = px.bar(top_15_losers, x='Symbol', y='Value Change (Rs)', 
                         title='Top 15 Holdings by Value Lost',
                         labels={'Value Change (Rs)': 'Loss in Value (Rs)', 'Symbol': 'Stock Symbol'},
                         hover_name='Name')
            fig.update_layout(yaxis_title="Loss in Value (Rs)", xaxis_title="Stock Symbol")
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: API Interactions (UPDATED to integrate new Flask endpoints) ---
with tabs[3]:
    st.header("🔧 Compliance API Interactions")
    st.markdown("Interact directly with the compliance backend API for simulation, optimization, and drift/rebalancing.")

    current_portfolio_df = st.session_state.get("compliance_results_df")
    current_rules_text = st.session_state.get("current_rules_text")
    current_threshold_configs = st.session_state.get("threshold_configs")

    if current_portfolio_df is None or current_portfolio_df.empty:
        st.warning("⚠️ Please load or analyze a portfolio in 'Portfolio Analysis' tab to use API functions.")
        st.stop()
    
    st.info(f"**Using Portfolio:** `{st.session_state.get('current_portfolio_name', 'Unnamed Portfolio')}`")
    
    # --- Prepare Data for API Calls ---
    # Ensure necessary columns are present for the Flask backend to process
    portfolio_for_api = current_portfolio_df.copy()
    
    if 'Industry' not in portfolio_for_api.columns: portfolio_for_api['Industry'] = 'UNKNOWN'
    if 'Name' not in portfolio_for_api.columns: portfolio_for_api['Name'] = portfolio_for_api['Symbol']
    if 'Avg_Buy_Price' not in portfolio_for_api.columns: portfolio_for_api['Avg_Buy_Price'] = portfolio_for_api['LTP'] * 0.95
    if 'Beta' not in portfolio_for_api.columns: portfolio_for_api['Beta'] = 1.0
    if 'Volatility' not in portfolio_for_api.columns: portfolio_for_api['Volatility'] = 0.20
    if 'Avg_Volume' not in portfolio_for_api.columns: portfolio_for_api['Avg_Volume'] = 1000000

    # Rename Streamlit columns to Flask expected columns
    portfolio_for_api = portfolio_for_api.rename(columns={
        'Real-time Value (Rs)': 'Market_Value', 
        'Weight %': 'Weight' # Flask internal prep will re-calculate, but helpful to pass the weight context
    })
    
    # Clean data for API
    for col in ['Quantity', 'LTP', 'Avg_Buy_Price', 'Beta', 'Avg_Volume', 'Volatility']:
        if col in portfolio_for_api.columns:
            portfolio_for_api[col] = pd.to_numeric(portfolio_for_api[col], errors='coerce').fillna(0).astype(float)
    
    # Extract only the essential columns the Flask endpoint needs (LTP, Quantity, Avg_Buy_Price, Beta, Avg_Volume, Volatility, Symbol, Industry)
    flask_required_cols = ['Symbol', 'Name', 'LTP', 'Quantity', 'Industry', 'Avg_Buy_Price', 'Beta', 'Avg_Volume', 'Volatility']
    portfolio_for_api = portfolio_for_api[[col for col in flask_required_cols if col in portfolio_for_api.columns]]

    # Sub-tabs for specific API features
    api_call_tab1, api_call_tab2, api_call_tab3 = st.tabs([
        "Max Sharpe Optimization", 
        "Model Drift & Rebalance",
        "Volatility Cone"
    ])

    # 1. Max Sharpe Optimization (Feature 4)
    with api_call_tab1:
        st.subheader("⚖️ Portfolio Optimization (Max Sharpe)")
        st.write("Calculates optimal portfolio weights for maximum Sharpe Ratio (based on simplified returns/volatility assumptions).")
        
        risk_free_rate = st.number_input("Risk-Free Rate (Annual, decimal)", 0.01, 0.10, 0.04, 0.005)
        
        if st.button("Calculate Optimal Weights", type="primary"):
            with st.spinner("Running mean-variance optimization..."):
                # Call optimization API (Feature 4)
                opt_res = call_optimization_api(portfolio_for_api)
                if opt_res and "optimal_structure" in opt_res:
                    st.session_state["optimization_results"] = opt_res["optimal_structure"]
                else:
                    st.error("Failed to run optimization.")
        
        if st.session_state.get("optimization_results"):
            opt_df = pd.DataFrame(st.session_state["optimization_results"])
            
            st.success("Optimization Complete")
            st.markdown("##### Current vs Optimal Weights")
            
            # Merge back the name/industry for better display
            display_df = current_portfolio_df[['Symbol', 'Name', 'Industry', 'Weight %']].merge(
                opt_df, on='Symbol', how='left'
            ).fillna(0)

            # Calculate reallocation amount
            current_total_value = current_portfolio_df['Real-time Value (Rs)'].sum()
            display_df['Weight Diff (%)'] = display_df['Optimal_Weight_%'] - display_df['Weight %']
            display_df['Reallocation Value (Rs)'] = display_df['Weight Diff (%)'] / 100 * current_total_value
            
            st.dataframe(
                display_df[['Symbol', 'Name', 'Weight %', 'Optimal_Weight_%', 'Weight Diff (%)', 'Reallocation Value (Rs)']]
                .sort_values('Weight Diff (%)', ascending=False)
                .style.format({
                    'Weight %': '{:.2f}%',
                    'Optimal_Weight_%': '{:.2f}%',
                    'Weight Diff (%)': '{:+.2f}%',
                    'Reallocation Value (Rs)': '₹{:,.0f}'
                }),
                use_container_width=True
            )
            
            fig = px.bar(display_df, x='Symbol', y=['Weight %', 'Optimal_Weight_%'], 
                         barmode='group', title='Current vs Optimal Weight Comparison')
            st.plotly_chart(fig, use_container_width=True)

    # 2. Model Drift & Rebalance (Features 6 & 7)
    with api_call_tab2:
        st.subheader("📉 Model Drift Analysis & Rebalancing Orders")
        st.write("Compare current portfolio against a target model and generate trades to fix drift.")
        
        st.markdown("##### Define Target Model Weights (Total must equal 100%)")
        
        # Create a dynamic list for target model input
        target_model_input = []
        for index, row in current_portfolio_df.head(5).iterrows(): # Show top 5 by default
            target_model_input.append({
                "symbol": row['Symbol'],
                "target_weight": st.number_input(f"{row['Symbol']} Target Weight (%)", 0.0, 100.0, value=100.0/len(current_portfolio_df) if len(current_portfolio_df) <= 10 else 0.0, step=0.5, key=f"target_{row['Symbol']}")
            })
        
        if st.button("Check Drift & Generate Orders", type="primary"):
            # Prepare target model list
            target_model_list = [
                {'symbol': item['symbol'], 'target_weight': item['target_weight']}
                for item in target_model_input if item['target_weight'] > 0
            ]
            
            if sum(item['target_weight'] for item in target_model_list) > 105 or sum(item['target_weight'] for item in target_model_list) < 95:
                 st.warning("Warning: Target weights sum significantly deviates from 100%. Results might be skewed.")
            
            with st.spinner("Running drift analysis and generating rebalance orders..."):
                # Call rebalance API (Features 6 & 7)
                rebalance_res = call_rebalance_api_full(portfolio_for_api.rename(columns={'Weight': 'Weight %'}), target_model_list)
                
                if rebalance_res:
                    st.session_state["rebalance_suggestions"] = rebalance_res
                    st.success("Rebalance analysis complete.")
                else:
                    st.error("Failed to run rebalance analysis.")

        if st.session_state.get("rebalance_suggestions"):
            res = st.session_state["rebalance_suggestions"]
            
            st.markdown("##### 1. Drift Analysis")
            st.metric("Total Absolute Drift", f"{res['drift_analysis']['total_absolute_drift']:.2f}%")
            
            drift_df = pd.DataFrame(res['drift_analysis']['drift_summary'])
            if not drift_df.empty:
                st.warning(f"🚨 {len(drift_df)} Assets with >2% Drift")
                st.dataframe(drift_df.style.format({'Weight_%': '{:.2f}%', 'Target_Weight_%': '{:.2f}%', 'Drift_%': '{:+.2f}%'}), use_container_width=True)
            else:
                st.info("No significant drift detected (>2%).")
            
            st.markdown("##### 2. Suggested Rebalance Orders")
            orders_df = pd.DataFrame(res['suggested_orders'])
            if not orders_df.empty:
                st.dataframe(orders_df.style.format({'quantity': '{:,.0f}', 'approx_value': '₹{:,.0f}'}), use_container_width=True)
                
                # Summary
                buy_val = orders_df[orders_df['action'] == 'BUY']['approx_value'].sum()
                sell_val = orders_df[orders_df['action'] == 'SELL']['approx_value'].sum()
                
                col_buy, col_sell, col_net = st.columns(3)
                col_buy.metric("Total Buy Value", f"₹{buy_val:,.0f}")
                col_sell.metric("Total Sell Value", f"₹{sell_val:,.0f}")
                col_net.metric("Net Cash Required", f"₹{buy_val - sell_val:,.0f}")
            else:
                st.info("No trades required to align with the target model.")

    # 3. Volatility Cone (Feature 10)
    with api_call_tab3:
        st.subheader("🎯 Volatility Cone Projection")
        st.write("Projects the expected price range of the portfolio over various time horizons (1-sigma and 2-sigma).")
        
        if st.button("Generate Volatility Cone", type="primary"):
            with st.spinner("Calculating volatility projections..."):
                # Call Volatility Cone API (Feature 10)
                vol_res = call_volatility_cone_api(portfolio_for_api)
                if vol_res and "volatility_cone" in vol_res:
                    st.session_state["volatility_cone_data"] = vol_res["volatility_cone"]
                else:
                    st.error("Failed to calculate volatility cone.")
        
        if st.session_state.get("volatility_cone_data"):
            cone_data = st.session_state["volatility_cone_data"]
            current_value = portfolio_for_api['Market_Value'].sum()

            st.success("Projection Complete")
            
            # Data preparation for chart
            projection_df = pd.DataFrame.from_dict(cone_data, orient='index')
            projection_df.index.name = 'Time Horizon'
            projection_df = projection_df.reset_index()

            # Create the cone chart
            fig = go.Figure()
            
            # 2-Sigma Cone (Outer bounds)
            fig.add_trace(go.Scatter(
                x=projection_df['Time Horizon'], y=projection_df['upper_2std'],
                mode='lines', fill=None, line=dict(width=0),
                name='95% Upper Bound', showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=projection_df['Time Horizon'], y=projection_df['lower_2std'],
                mode='lines', fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', line=dict(width=0),
                name='95% Range'
            ))

            # 1-Sigma Cone (Inner bounds)
            fig.add_trace(go.Scatter(
                x=projection_df['Time Horizon'], y=projection_df['upper_1std'],
                mode='lines', fill=None, line=dict(width=0),
                name='68% Upper Bound', showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=projection_df['Time Horizon'], y=projection_df['lower_1std'],
                mode='lines', fill='tonexty', fillcolor='rgba(0, 100, 200, 0.3)', line=dict(width=0),
                name='68% Range'
            ))

            # Current Value Line
            fig.add_hline(y=current_value, line_dash="dash", line_color="black", annotation_text="Current Value", annotation_position="top left")
            
            fig.update_layout(
                title="Portfolio Volatility Cone Projection",
                xaxis_title="Time Horizon",
                yaxis_title="Projected Portfolio Value (Rs)",
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### Projection Data Table")
            st.dataframe(projection_df.style.format({col: '₹{:,.0f}' for col in projection_df.columns if col != 'Time Horizon'}), use_container_width=True)


# --- TAB 5: Advanced Analytics (Updated to use new Flask endpoints) ---
with tabs[4]:
    st.header("📊 Advanced Risk Analytics")
    
    if st.session_state["compliance_results_df"].empty:
        st.warning("Please analyze a portfolio first.")
        st.stop()
    
    # Use the prepared API data (portfolio_for_api)
    portfolio_for_api = st.session_state.get("compliance_results_df")

    # Sub-tabs for specific API features
    adv_tab1, adv_tab2, adv_tab3 = st.tabs([
        "Correlation Matrix", 
        "HHI & Risk Score",
        "Bulk Compliance Simulation"
    ])

    # 1. Correlation Matrix (Feature 9)
    with adv_tab1:
        st.subheader("🤝 Inter-Asset Correlation Matrix")
        st.write("Simulates correlation matrix based on sector/industry mapping, assuming higher intra-sector correlation.")
        
        if st.button("Generate Correlation Matrix", key="btn_corr_matrix"):
            with st.spinner("Generating simulated correlation matrix..."):
                corr_res = call_correlation_api(portfolio_for_api)
                if corr_res:
                    st.session_state["correlation_matrix_data"] = corr_res
                    st.success("Correlation matrix generated.")
                else:
                    st.error("Failed to generate correlation matrix.")
        
        if st.session_state.get("correlation_matrix_data"):
            corr_data = st.session_state["correlation_matrix_data"]
            symbols = corr_data['symbols']
            matrix = np.array(corr_data['correlation_matrix'])

            st.markdown("##### Matrix Visualization")
            fig = px.imshow(matrix, 
                            x=symbols, y=symbols, 
                            color_continuous_scale='RdBu_r', 
                            aspect="auto", 
                            title="Simulated Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("##### Raw Data")
            corr_df = pd.DataFrame(matrix, index=symbols, columns=symbols)
            st.dataframe(corr_df.style.format('{:.2f}'), use_container_width=True)

    # 2. HHI & Risk Score (Using data fetched by /analytics/comprehensive)
    with adv_tab2:
        st.subheader("🚨 Portfolio Risk & Concentration Summary")
        
        comprehensive_data = st.session_state.get("comprehensive_analytics_data")
        
        if not comprehensive_data:
            st.warning("Comprehensive risk data is missing. Please re-run analysis in Tab 1.")
        else:
            # We must calculate HHI/Gini client-side as the Flask comprehensive endpoint does not return HHI.
            # However, the flask code contains HHI logic in its utility functions, let's mock it from the concentration tab.
            
            # Using local HHI calculation from Tab 1 (Concentration)
            hhi = (portfolio_for_api['Weight %'] ** 2).sum() 
            
            st.markdown("### Concentration Risk")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("HHI Score (Holdings)", f"{hhi:.2f}", help="Lower is more diversified.")
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = hhi,
                    title = {'text': "HHI Score"},
                    gauge = {
                        'axis': {'range': [0, 1000]}, # Adjusted range for HHI 
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [0, 150], 'color': "lightgreen"},
                            {'range': [150, 300], 'color': "yellow"},
                            {'range': [300, 1000], 'color': "red"}],
                    }))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Liquidity Deep Dive
            st.markdown("### Liquidity Deep Dive")
            liquidity = comprehensive_data.get('liquidity_analysis', {})
            
            if liquidity and liquidity['illiquid_assets']:
                illiquid_df = pd.DataFrame(liquidity['illiquid_assets'])
                illiquid_df = illiquid_df.merge(portfolio_for_api[['Symbol', 'Industry', 'Weight %']], on='Symbol', how='left')
                
                st.metric("Total Illiquid Exposure", f"{illiquid_df['Weight %'].sum():.2f}%")
                st.dataframe(illiquid_df, use_container_width=True)
            else:
                st.info("No assets classified as illiquid (Days to Liquidate > 3).")

    # 3. Bulk Compliance Simulation (Feature 8)
    with adv_tab3:
        st.subheader("📦 Bulk Portfolio Compliance Simulation")
        st.write("Simulate compliance and valuation checks across multiple portfolios simultaneously.")
        
        st.warning("NOTE: This requires the target Flask endpoint to have the full compliance rules engine implemented, which is currently NOT included in the provided Flask code. Results will only include basic valuation/status.")

        bulk_uploaded_file = st.file_uploader("Upload CSV of Multiple Portfolios", type="csv", key="bulk_portfolio_csv", help="Format: portfolio_id, symbol, quantity, ltp, industry...")

        if bulk_uploaded_file:
            try:
                bulk_df = pd.read_csv(bulk_uploaded_file)
                st.info(f"Loaded {len(bulk_df)} holdings across {bulk_df['portfolio_id'].nunique()} portfolios.")
                
                # Group data into the required API format
                portfolios_list = []
                for p_id, group in bulk_df.groupby('portfolio_id'):
                    # Ensure minimal required columns for Flask's _vectorized_prep
                    holdings = group.rename(columns={'symbol': 'Symbol', 'quantity': 'Quantity', 'ltp': 'LTP', 'industry': 'Industry'}).to_dict('records')
                    portfolios_list.append({
                        "id": str(p_id),
                        "holdings": holdings
                    })
                
                if st.button("Run Bulk Compliance Check (API)", type="primary"):
                    with st.spinner(f"Processing {len(portfolios_list)} portfolios via API..."):
                        # Rules text is ignored by the current Flask implementation of /compliance/bulk
                        payload = {
                            "portfolios": portfolios_list,
                            "rules_text": st.session_state.get("current_rules_text", "MAX_STOCK_WEIGHT <= 10")
                        }
                        
                        bulk_res = call_compliance_api("/compliance/bulk", payload)
                        
                        if bulk_res and "bulk_results" in bulk_res:
                            bulk_results_df = pd.DataFrame(bulk_res["bulk_results"])
                            st.success("Bulk simulation complete.")
                            st.dataframe(bulk_results_df, use_container_width=True)
                        else:
                            st.error("Bulk API failed to return results.")
            except Exception as e:
                st.error(f"Error processing bulk file: {e}")


# --- TAB 6: History (No change) ---
with tabs[5]:
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
    <p><strong>Invsion Connect Pro</strong> - Advanced Portfolio Analytics</p>
    <p style='font-size: 0.9em;'>⚠️ For informational purposes only. Consult professionals for investment decisions.</p>
    <p style='font-size: 0.8em;'>Powered by KiteConnect, Google Gemini AI & Flask Engine</p>
    <p style='font-size: 0.8em;'>User: {st.session_state["user_email"]} | Session Active</p>
    <p style='font-size: 0.8em;'>API Base URL: {COMPLIANCE_API_BASE_URL}</p>
</div>
""", unsafe_allow_html=True)
