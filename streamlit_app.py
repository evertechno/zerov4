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
BENCHMARK_SYMBOL = "NIFTY 50"

# Initialize session state
if "user_authenticated" not in st.session_state: st.session_state["user_authenticated"] = False
if "user_id" not in st.session_state: st.session_state["user_id"] = None
if "user_email" not in st.session_state: st.session_state["user_email"] = None
if "kite_access_token" not in st.session_state: st.session_state["kite_access_token"] = None
if "compliance_results_df" not in st.session_state: st.session_state["compliance_results_df"] = pd.DataFrame()
if "advanced_metrics" not in st.session_state: st.session_state["advanced_metrics"] = None
if "ai_analysis_response" not in st.session_state: st.session_state["ai_analysis_response"] = None
if "security_level_compliance" not in st.session_state: st.session_state["security_level_compliance"] = pd.DataFrame()
if "breach_alerts" not in st.session_state: st.session_state["breach_alerts"] = []
if "saved_analyses" not in st.session_state: st.session_state["saved_analyses"] = []


# --- Load Credentials ---
def load_secrets():
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
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(email: str, password: str, full_name: str = None):
    try:
        existing = supabase.table('users').select('*').eq('email', email).execute()
        if existing.data:
            return False, "User already exists."
        
        user_data = {
            'email': email,
            'password_hash': hash_password(password),
            'full_name': full_name,
            'created_at': datetime.now().isoformat()
        }
        
        result = supabase.table('users').insert(user_data).execute()
        if result.data:
            return True, "Registration successful!"
        return False, "Registration failed."
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_user(email: str, password: str):
    try:
        result = supabase.table('users').select('*').eq('email', email).eq('password_hash', hash_password(password)).execute()
        
        if result.data and len(result.data) > 0:
            user = result.data[0]
            st.session_state["user_authenticated"] = True
            st.session_state["user_id"] = user['id']
            st.session_state["user_email"] = user['email']
            supabase.table('users').update({'last_login': datetime.now().isoformat()}).eq('id', user['id']).execute()
            return True, "Login successful!"
        return False, "Invalid credentials."
    except Exception as e:
        return False, f"Error: {str(e)}"

def logout_user():
    st.session_state.clear()


# --- Data Persistence Functions ---
def save_analysis_to_supabase(analysis_data: dict):
    try:
        analysis_record = {
            'user_id': st.session_state["user_id"],
            'analysis_date': datetime.now().isoformat(),
            'portfolio_data': analysis_data.get('portfolio_data'),
            'compliance_results': analysis_data.get('compliance_results'),
            'compliance_rules': analysis_data.get('compliance_rules'),
            'breach_alerts': analysis_data.get('breach_alerts'),
            'security_compliance': analysis_data.get('security_compliance'),
            'advanced_metrics': analysis_data.get('advanced_metrics'),
            'ai_analysis': analysis_data.get('ai_analysis'),
            'metadata': analysis_data.get('metadata', {})
        }
        
        result = supabase.table('portfolio_analyses').insert(analysis_record).execute()
        if result.data:
            return True, result.data[0]['id']
        return False, None
    except Exception as e:
        st.error(f"Error saving: {str(e)}")
        return False, None

def get_user_analyses(user_id: str, limit: int = 10):
    try:
        result = supabase.table('portfolio_analyses').select('*').eq('user_id', user_id).order('analysis_date', desc=True).limit(limit).execute()
        return result.data if result.data else []
    except Exception as e:
        st.error(f"Error fetching: {str(e)}")
        return []

def load_analysis_from_supabase(analysis_id: str):
    try:
        result = supabase.table('portfolio_analyses').select('*').eq('id', analysis_id).execute()
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        st.error(f"Error loading: {str(e)}")
        return None

def delete_analysis(analysis_id: str):
    try:
        supabase.table('portfolio_analyses').delete().eq('id', analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting: {str(e)}")
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


# --- Compliance Functions ---
def parse_and_validate_rules_enhanced(rules_text: str, portfolio_df: pd.DataFrame):
    results = []
    if not rules_text.strip() or portfolio_df.empty: 
        return results
    
    sector_weights = portfolio_df.groupby('Industry')['Weight %'].sum()
    stock_weights = portfolio_df.set_index('Symbol')['Weight %']
    
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
                results.append({'rule': rule, 'status': 'Error', 'details': 'Invalid format', 'severity': 'N/A'})
                continue
            
            op = parts[-2]
            if op not in ['>', '<', '>=', '<=', '=']:
                results.append({'rule': rule, 'status': 'Error', 'details': f"Invalid operator '{op}'", 'severity': 'N/A'})
                continue
            
            threshold = float(parts[-1].replace('%', ''))
            
            if rule_type == 'STOCK' and len(parts) == 4:
                symbol = parts[1].upper()
                if symbol in stock_weights.index:
                    actual_value = stock_weights.get(symbol, 0.0)
                    details = f"Actual for {symbol}: {actual_value:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Symbol '{symbol}' not found", 'severity': 'N/A'})
                    continue
            
            elif rule_type == 'SECTOR':
                sector_name = ' '.join(parts[1:-2]).upper()
                matching_sector = next((s for s in sector_weights.index if s.upper() == sector_name), None)
                if matching_sector:
                    actual_value = sector_weights.get(matching_sector, 0.0)
                    details = f"Actual for {matching_sector}: {actual_value:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Sector '{sector_name}' not found", 'severity': 'N/A'})
                    continue
            
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
            
            else:
                results.append({'rule': rule, 'status': 'Error', 'details': 'Unrecognized format', 'severity': 'N/A'})
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
            results.append({'rule': rule, 'status': 'Error', 'details': f"Parse error: {e}", 'severity': 'N/A'})
    
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
    from_date = datetime.now().date() - timedelta(days=366)
    to_date = datetime.now().date()
    
    returns_df = pd.DataFrame()
    failed_symbols = []
    
    progress_bar = st.progress(0, text="Fetching historical data...")
    
    for i, symbol in enumerate(symbols):
        hist_data = get_historical_data_cached(api_key, access_token, symbol, from_date, to_date, 'day')
        if not hist_data.empty and '_error' not in hist_data.columns:
            returns_df[symbol] = hist_data['close'].pct_change()
        else:
            failed_symbols.append(symbol)
        progress_bar.progress((i + 1) / len(symbols), text=f"Fetching {symbol}...")
    
    if failed_symbols:
        st.warning(f"Failed to fetch: {', '.join(failed_symbols)}")
    
    returns_df.dropna(how='all', inplace=True)
    returns_df.fillna(0, inplace=True)
    
    if returns_df.empty:
        st.error("Not enough data for metrics.")
        progress_bar.empty()
        return None
    
    successful_symbols = returns_df.columns.tolist()
    portfolio_df_success = portfolio_df.set_index('Symbol').reindex(successful_symbols).reset_index()
    total_value_success = portfolio_df_success['Real-time Value (Rs)'].sum()
    
    if total_value_success == 0:
        progress_bar.empty()
        return None
    
    weights = (portfolio_df_success['Real-time Value (Rs)'] / total_value_success).values
    portfolio_returns = returns_df.dot(weights)
    
    var_95 = portfolio_returns.quantile(0.05)
    var_99 = portfolio_returns.quantile(0.01)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    portfolio_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    progress_bar.empty()
    
    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "portfolio_volatility": portfolio_vol,
        "beta": None,
        "alpha": None,
        "tracking_error": None,
        "information_ratio": None,
        "sortino_ratio": None,
        "avg_correlation": None,
        "diversification_ratio": None
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
                reg_name = st.text_input("Full Name")
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
                            success, message = register_user(reg_email, reg_password, reg_name)
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                    else:
                        st.warning("Fill all fields.")


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
    st.markdown("### Saved Analyses")
    
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state["saved_analyses"] = get_user_analyses(st.session_state["user_id"])
    
    if not st.session_state.get("saved_analyses"):
        st.session_state["saved_analyses"] = get_user_analyses(st.session_state["user_id"])
    
    if st.session_state["saved_analyses"]:
        st.markdown(f"**{len(st.session_state['saved_analyses'])} saved**")
        
        for analysis in st.session_state["saved_analyses"][:5]:
            analysis_date = datetime.fromisoformat(analysis['analysis_date']).strftime('%Y-%m-%d %H:%M')
            with st.expander(f"📅 {analysis_date}"):
                if st.button(f"Load", key=f"load_{analysis['id']}", use_container_width=True):
                    loaded = load_analysis_from_supabase(analysis['id'])
                    if loaded:
                        if loaded.get('portfolio_data'):
                            st.session_state["compliance_results_df"] = pd.read_json(loaded['portfolio_data'])
                        if loaded.get('security_compliance'):
                            st.session_state["security_level_compliance"] = pd.read_json(loaded['security_compliance'])
                        if loaded.get('breach_alerts'):
                            st.session_state["breach_alerts"] = json.loads(loaded['breach_alerts'])
                        if loaded.get('advanced_metrics'):
                            st.session_state["advanced_metrics"] = json.loads(loaded['advanced_metrics'])
                        if loaded.get('ai_analysis'):
                            st.session_state["ai_analysis_response"] = loaded['ai_analysis']
                        
                        st.success("Loaded!")
                        st.rerun()
                
                if st.button(f"Delete", key=f"del_{analysis['id']}", use_container_width=True):
                    if delete_analysis(analysis['id']):
                        st.success("Deleted!")
                        st.session_state["saved_analyses"] = get_user_analyses(st.session_state["user_id"])
                        st.rerun()
    else:
        st.info("No saved analyses")


# --- Main Tabs ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]

tabs = st.tabs(["💼 Compliance", "🤖 AI Analysis", "📚 History"])


# --- TAB 1: Compliance ---
with tabs[0]:
    st.header("💼 Investment Compliance")
    
    if not k:
        st.warning("⚠️ Connect to Kite first")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Upload Portfolio")
        uploaded_file = st.file_uploader("CSV file", type="csv")
        
        with st.expander("⚙️ Thresholds"):
            single_stock_limit = st.number_input("Single Stock %", 1.0, 25.0, 10.0, 0.5)
            single_sector_limit = st.number_input("Single Sector %", 5.0, 50.0, 25.0, 1.0)
    
    with col2:
        st.subheader("Compliance Rules")
        rules_text = st.text_area("Rules (one per line)", height=200, 
                                   value="""# Example Rules
# STOCK RELIANCE < 10
# SECTOR BANKING < 25
# TOP_N_STOCKS 10 <= 50""")
    
    if uploaded_file and k:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [str(col).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_') for col in df.columns]
            
            header_map = {
                'symbol': 'Symbol', 'industry': 'Industry', 'quantity': 'Quantity',
                'name_of_the_instrument': 'Name',
                'market_fair_value(rs_in_lacs)': 'Uploaded Value (Lacs)'
            }
            df = df.rename(columns=header_map)
            
            if 'Industry' in df.columns:
                df['Industry'] = df['Industry'].fillna('UNKNOWN').str.strip().str.upper()
            
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    symbols = df['Symbol'].unique().tolist()
                    ltp_data = k.ltp([f"{DEFAULT_EXCHANGE}:{s}" for s in symbols])
                    prices = {sym: ltp_data.get(f"{DEFAULT_EXCHANGE}:{sym}", {}).get('last_price') for sym in symbols}
                    
                    df_results = df.copy()
                    df_results['LTP'] = df_results['Symbol'].map(prices)
                    df_results['Real-time Value (Rs)'] = (df_results['LTP'] * pd.to_numeric(df_results['Quantity'], errors='coerce')).fillna(0)
                    total_value = df_results['Real-time Value (Rs)'].sum()
                    df_results['Weight %'] = (df_results['Real-time Value (Rs)'] / total_value * 100) if total_value > 0 else 0
                    
                    rules_config = {'single_stock_limit': single_stock_limit}
                    security_compliance = calculate_security_level_compliance(df_results, rules_config)
                    
                    st.session_state.compliance_results_df = df_results
                    st.session_state.security_level_compliance = security_compliance
                    
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
                    
                    st.session_state.breach_alerts = breaches
                    st.success("✅ Analysis Complete!")
                    
                    # Save to Supabase
                    analysis_data = {
                        'portfolio_data': df_results.to_json(),
                        'compliance_results': json.dumps(parse_and_validate_rules_enhanced(rules_text, df_results)),
                        'compliance_rules': rules_text,
                        'breach_alerts': json.dumps(breaches),
                        'security_compliance': security_compliance.to_json(),
                        'advanced_metrics': None,
                        'ai_analysis': None,
                        'metadata': {
                            'total_value': float(total_value),
                            'holdings_count': len(df_results),
                            'single_stock_limit': single_stock_limit,
                            'single_sector_limit': single_sector_limit
                        }
                    }
                    
                    success, analysis_id = save_analysis_to_supabase(analysis_data)
                    if success:
                        st.success(f"💾 Auto-saved! ID: {analysis_id}")
        
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Display results
    results_df = st.session_state.get("compliance_results_df", pd.DataFrame())
    
    if not results_df.empty:
        st.markdown("---")
        
        if st.session_state.get("breach_alerts"):
            st.error("🚨 **Compliance Breaches**")
            breach_df = pd.DataFrame(st.session_state["breach_alerts"])
            st.dataframe(breach_df, use_container_width=True, hide_index=True)
        
        analysis_tabs = st.tabs([
            "📊 Dashboard",
            "🔍 Details",
            "📈 Metrics",
            "⚖️ Rules",
            "🔐 Security",
            "📊 Concentration",
            "📄 Report"
        ])
        
        with analysis_tabs[0]:
            st.subheader("Dashboard")
            total_value = results_df['Real-time Value (Rs)'].sum()
            
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Value", f"₹ {total_value:,.2f}")
            kpi_cols[1].metric("Holdings", f"{len(results_df)}")
            kpi_cols[2].metric("Sectors", f"{results_df['Industry'].nunique()}")
            kpi_cols[3].metric("Top Stock", f"{results_df['Weight %'].max():.2f}%")
            kpi_cols[4].metric("Status", "✅" if not st.session_state.get("breach_alerts") else f"❌ {len(st.session_state['breach_alerts'])}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_15 = results_df.nlargest(15, 'Weight %')
                fig_pie = px.pie(top_15, values='Weight %', names='Name', title='Top 15 Holdings', hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                sector_data = results_df.groupby('Industry')['Weight %'].sum().reset_index().sort_values('Weight %', ascending=False).head(10)
                fig_sector = px.bar(sector_data, x='Weight %', y='Industry', orientation='h',
                                   title='Top 10 Sectors', color='Weight %', color_continuous_scale='Blues')
                fig_sector.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_sector, use_container_width=True)
        
        with analysis_tabs[1]:
            st.subheader("Holdings Details")
            top_20 = results_df.nlargest(20, 'Weight %')[['Name', 'Symbol', 'Industry', 'Weight %', 'Real-time Value (Rs)', 'LTP']]
            st.dataframe(top_20.style.format({
                'Weight %': '{:.2f}%',
                'Real-time Value (Rs)': '₹{:,.2f}',
                'LTP': '₹{:,.2f}'
            }), use_container_width=True)
        
        with analysis_tabs[2]:
            st.subheader("Advanced Metrics")
            
            if st.button("🔄 Calculate", type="primary", use_container_width=True):
                with st.spinner("Calculating..."):
                    st.session_state.advanced_metrics = calculate_advanced_metrics(results_df, api_key, access_token)
            
            if st.session_state.advanced_metrics:
                metrics = st.session_state.advanced_metrics
                
                risk_cols = st.columns(4)
                risk_cols[0].metric("VaR (95%)", f"{metrics['var_95'] * 100:.2f}%")
                risk_cols[1].metric("VaR (99%)", f"{metrics['var_99'] * 100:.2f}%")
                risk_cols[2].metric("CVaR", f"{metrics['cvar_95'] * 100:.2f}%")
                risk_cols[3].metric("Volatility", f"{metrics['portfolio_volatility'] * 100:.2f}%" if metrics['portfolio_volatility'] else "N/A")
        
        with analysis_tabs[3]:
            st.subheader("Rule Validation")
            
            validation_results = parse_and_validate_rules_enhanced(rules_text, results_df)
            
            if validation_results:
                total_rules = len(validation_results)
                passed = sum(1 for r in validation_results if r['status'] == "✅ PASS")
                failed = sum(1 for r in validation_results if r['status'] == "❌ FAIL")
                
                summary_cols = st.columns(3)
                summary_cols[0].metric("Total", total_rules)
                summary_cols[1].metric("✅ Pass", passed)
                summary_cols[2].metric("❌ Fail", failed)
                
                st.markdown("---")
                
                for res in validation_results:
                    if res['status'] == "✅ PASS":
                        with st.expander(f"{res['status']} | `{res['rule']}`", expanded=False):
                            st.success(f"**Status:** {res['status']}")
                            st.write(f"**Details:** {res['details']}")
                    elif res['status'] == "❌ FAIL":
                        with st.expander(f"{res['status']} {res.get('severity', '')} | `{res['rule']}`", expanded=True):
                            st.error(f"**Status:** {res['status']}")
                            st.write(f"**Details:** {res['details']}")
        
        with analysis_tabs[4]:
            st.subheader("Security Compliance")
            
            security_df = st.session_state.get("security_level_compliance", pd.DataFrame())
            
            if not security_df.empty:
                breach_count = (security_df['Stock Limit Breach'] == '❌ Breach').sum()
                compliant_count = (security_df['Stock Limit Breach'] == '✅ Compliant').sum()
                
                summary_cols = st.columns(3)
                summary_cols[0].metric("Total", len(security_df))
                summary_cols[1].metric("✅", compliant_count)
                summary_cols[2].metric("❌", breach_count)
                
                st.dataframe(security_df[['Name', 'Symbol', 'Industry', 'Weight %', 'Stock Limit Breach', 'Concentration Risk']].style.format({
                    'Weight %': '{:.2f}%'
                }), use_container_width=True, height=500)
        
        with analysis_tabs[5]:
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
                xaxis_title='Holdings',
                yaxis_title='Cumulative %',
                height=400
            )
            st.plotly_chart(fig_lorenz, use_container_width=True)
            
            bench_cols = st.columns(5)
            bench_cols[0].metric("Top 1", f"{sorted_df.iloc[0]['Weight %']:.2f}%")
            bench_cols[1].metric("Top 3", f"{sorted_df.head(3)['Weight %'].sum():.2f}%")
            bench_cols[2].metric("Top 5", f"{sorted_df.head(5)['Weight %'].sum():.2f}%")
            bench_cols[3].metric("Top 10", f"{sorted_df.head(10)['Weight %'].sum():.2f}%")
            bench_cols[4].metric("Top 20", f"{sorted_df.head(20)['Weight %'].sum():.2f}%" if len(sorted_df) >= 20 else "N/A")
        
        with analysis_tabs[6]:
            st.subheader("Full Report")
            
            if st.button("📊 Generate Excel", type="primary", use_container_width=True):
                from io import BytesIO
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Holdings', index=False)
                    
                    sector_analysis = results_df.groupby('Industry').agg({
                        'Weight %': 'sum',
                        'Real-time Value (Rs)': 'sum',
                        'Symbol': 'count'
                    }).rename(columns={'Symbol': 'Count'})
                    sector_analysis.to_excel(writer, sheet_name='Sectors')
                
                output.seek(0)
                st.download_button(
                    "📥 Download",
                    output,
                    f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            st.markdown("---")
            st.dataframe(results_df, use_container_width=True, height=400)


# --- TAB 2: AI Analysis ---
with tabs[1]:
    st.header("🤖 AI Analysis")
    
    portfolio_df = st.session_state.get("compliance_results_df")
    
    if portfolio_df is None or portfolio_df.empty:
        st.warning("⚠️ Analyze portfolio first")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_docs = st.file_uploader(
                "📄 Upload Documents (SID/KIM)",
                type=["pdf", "txt"],
                accept_multiple_files=True
            )
        
        with col2:
            analysis_depth = st.select_slider(
                "Depth",
                options=["Quick", "Standard", "Comprehensive"],
                value="Standard"
            )
        
        if uploaded_docs:
            st.success(f"✅ {len(uploaded_docs)} uploaded")
            
            if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing..."):
                    try:
                        docs_text = extract_text_from_files(uploaded_docs)
                        portfolio_summary = get_portfolio_summary(portfolio_df)
                        breach_alerts = st.session_state.get("breach_alerts", [])
                        breach_summary = "\n".join([f"- {b['type']}: {b['details']}" for b in breach_alerts]) if breach_alerts else "No breaches."
                        
                        prompt = f"""You are an expert investment compliance analyst with SEBI regulations knowledge.

**TASK:** Comprehensive compliance analysis.

**PORTFOLIO:**
```
{portfolio_summary}
```

**DETECTED ISSUES:**
```
{breach_summary}
```

**SCHEME DOCUMENTS:**
```
{docs_text[:120000]}
```

**ANALYSIS FRAMEWORK:**

## 1. Executive Summary
- Overall compliance status
- Critical findings
- Key action items

## 2. Investment Alignment
- Portfolio vs stated philosophy
- Top holdings alignment
- Style drift

## 3. Regulatory Compliance

### 3.1 SEBI Regulations
- Single Issuer Limit (10%)
- Sectoral Concentration (25%)
- Group Exposure (25%)

### 3.2 Scheme-Specific
Verify against uploaded documents

## 4. Portfolio Quality & Risk
Comprehensive risk analysis

## 5. Violations & Concerns
List with severity, description, reference

## 6. Best Practices
Industry comparison

## 7. Recommendations
Specific actionable recommendations

## 8. Disclaimers
Missing information and assumptions

Begin analysis:
"""
                        
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        response = model.generate_content(
                            prompt,
                            generation_config={
                                'temperature': 0.3,
                                'top_p': 0.8,
                                'max_output_tokens': 8192,
                            }
                        )
                        
                        st.session_state.ai_analysis_response = response.text
                        st.success("✅ Complete!")
                        
                        # Save AI analysis
                        analysis_data = {
                            'portfolio_data': st.session_state["compliance_results_df"].to_json(),
                            'compliance_results': None,
                            'compliance_rules': None,
                            'breach_alerts': json.dumps(st.session_state.get("breach_alerts", [])),
                            'security_compliance': st.session_state.get("security_level_compliance", pd.DataFrame()).to_json(),
                            'advanced_metrics': json.dumps(st.session_state.get("advanced_metrics")),
                            'ai_analysis': response.text,
                            'metadata': {'analysis_type': 'AI', 'depth': analysis_depth}
                        }
                        save_analysis_to_supabase(analysis_data)
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        if st.session_state.get("ai_analysis_response"):
            st.markdown("---")
            st.markdown("## 📊 AI Analysis Report")
            st.markdown("---")
            st.markdown(st.session_state.ai_analysis_response)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                txt_data = st.session_state.ai_analysis_response.encode('utf-8')
                st.download_button(
                    "📄 Text",
                    txt_data,
                    f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    use_container_width=True
                )
            
            with col2:
                md_data = st.session_state.ai_analysis_response.encode('utf-8')
                st.download_button(
                    "📝 Markdown",
                    md_data,
                    f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    use_container_width=True
                )
            
            with col3:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.ai_analysis_response = None
                    st.rerun()


# --- TAB 3: History ---
with tabs[2]:
    st.header("📚 Analysis History")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"Total: **{len(st.session_state.get('saved_analyses', []))}** analyses")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state["saved_analyses"] = get_user_analyses(st.session_state["user_id"], limit=50)
            st.rerun()
    
    if not st.session_state.get("saved_analyses"):
        st.session_state["saved_analyses"] = get_user_analyses(st.session_state["user_id"], limit=50)
    
    if st.session_state["saved_analyses"]:
        st.markdown("---")
        
        for idx, analysis in enumerate(st.session_state["saved_analyses"][:20]):
            analysis_date = datetime.fromisoformat(analysis['analysis_date'])
            
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                
                with col1:
                    st.markdown(f"**📅 {analysis_date.strftime('%Y-%m-%d %H:%M')}**")
                    
                    if analysis.get('metadata'):
                        metadata = analysis['metadata']
                        if isinstance(metadata, str):
                            metadata = json.loads(metadata)
                        
                        info_text = []
                        if metadata.get('total_value'):
                            info_text.append(f"Value: ₹{metadata['total_value']:,.2f}")
                        if metadata.get('holdings_count'):
                            info_text.append(f"Holdings: {metadata['holdings_count']}")
                        
                        if info_text:
                            st.caption(" | ".join(info_text))
                
                with col2:
                    components = []
                    if analysis.get('portfolio_data'):
                        components.append("📊")
                    if analysis.get('compliance_results'):
                        components.append("⚖️")
                    if analysis.get('advanced_metrics'):
                        components.append("📈")
                    if analysis.get('ai_analysis'):
                        components.append("🤖")
                    
                    st.caption(" ".join(components))
                
                with col3:
                    if st.button("📂", key=f"load_h_{idx}", use_container_width=True):
                        loaded = load_analysis_from_supabase(analysis['id'])
                        if loaded:
                            if loaded.get('portfolio_data'):
                                st.session_state["compliance_results_df"] = pd.read_json(loaded['portfolio_data'])
                            if loaded.get('security_compliance'):
                                st.session_state["security_level_compliance"] = pd.read_json(loaded['security_compliance'])
                            if loaded.get('breach_alerts'):
                                st.session_state["breach_alerts"] = json.loads(loaded['breach_alerts'])
                            if loaded.get('advanced_metrics'):
                                st.session_state["advanced_metrics"] = json.loads(loaded['advanced_metrics'])
                            if loaded.get('ai_analysis'):
                                st.session_state["ai_analysis_response"] = loaded['ai_analysis']
                            
                            st.success("✅ Loaded!")
                            time.sleep(1)
                            st.rerun()
                
                with col4:
                    if st.button("🗑️", key=f"delete_h_{idx}", use_container_width=True):
                        if delete_analysis(analysis['id']):
                            st.success("Deleted!")
                            st.session_state["saved_analyses"] = get_user_analyses(st.session_state["user_id"], limit=50)
                            time.sleep(0.5)
                            st.rerun()
                
                st.markdown("---")
    
    else:
        st.info("📭 No saved analyses yet!")


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
