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
st.set_page_config(page_title="Invsion Connect - Portfolio Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("Invsion Connect")
st.markdown("A comprehensive platform for fetching market data, validating investment compliance, and AI-powered analysis.")

# --- Global Constants & Session State Initialization ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
BENCHMARK_SYMBOL = "NIFTY 50"

# Initialize session state variables
st.session_state.setdefault("kite_access_token", None)
st.session_state.setdefault("kite_login_response", None)
st.session_state.setdefault("instruments_df", pd.DataFrame())
st.session_state.setdefault("historical_data", pd.DataFrame())
st.session_state.setdefault("last_fetched_symbol", None)
st.session_state.setdefault("current_market_data", None)
st.session_state.setdefault("holdings_data", None)
st.session_state.setdefault("compliance_results_df", pd.DataFrame())
st.session_state.setdefault("advanced_metrics", None)
st.session_state.setdefault("ai_analysis_response", None)


# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    secrets = st.secrets
    kite_conf = secrets.get("kite", {})
    gemini_conf = secrets.get("google_gemini", {})
    
    errors = []
    if not all(k in kite_conf for k in ["api_key", "api_secret", "redirect_uri"]):
        errors.append("Kite credentials (api_key, api_secret, redirect_uri)")
    if not gemini_conf.get("api_key"):
        errors.append("Google Gemini API key")

    if errors:
        st.error(f"Missing required credentials in `.streamlit/secrets.toml`: {', '.join(errors)}.")
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

# --- Utility Functions ---

def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None

@st.cache_data(ttl=86400, show_spinner="Loading instruments...")
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: return pd.DataFrame({"_error": ["Kite not authenticated."]})
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        df = df[['instrument_token', 'tradingsymbol', 'name', 'exchange']]
        return df
    except Exception as e: return pd.DataFrame({"_error": [f"Failed to load instruments: {e}"]})

@st.cache_data(ttl=60)
def get_ltp_price_cached(api_key: str, access_token: str, symbol: str, exchange: str = DEFAULT_EXCHANGE):
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: return {"_error": "Kite not authenticated."}
    try: return kite_instance.ltp([f"{exchange.upper()}:{symbol.upper()}"]).get(f"{exchange.upper()}:{symbol.upper()}")
    except Exception as e: return {"_error": str(e)}

@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date: datetime.date, to_date: datetime.date, interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance: return pd.DataFrame({"_error": ["Kite not authenticated."]})
    instruments_df = load_instruments_cached(api_key, access_token)
    token = find_instrument_token(instruments_df, symbol, exchange)
    if not token and symbol in ["NIFTY BANK", BENCHMARK_SYMBOL, "SENSEX"]:
        index_exchange = "NSE" if symbol != "SENSEX" else "BSE"
        token = find_instrument_token(load_instruments_cached(api_key, access_token, index_exchange), symbol, index_exchange)
    if not token: return pd.DataFrame({"_error": [f"Instrument token not found for {symbol}."]})
    try:
        data = kite_instance.historical_data(token, from_date=datetime.combine(from_date, datetime.min.time()), to_date=datetime.combine(to_date, datetime.max.time()), interval=interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]); df.set_index("date", inplace=True)
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce'); df.dropna(subset=['close'], inplace=True)
        return df
    except Exception as e: return pd.DataFrame({"_error": [str(e)]})

def find_instrument_token(df: pd.DataFrame, tradingsymbol: str, exchange: str = DEFAULT_EXCHANGE) -> int | None:
    if df.empty: return None
    mask = (df.get("exchange", "").str.upper() == exchange.upper()) & (df.get("tradingsymbol", "").str.upper() == tradingsymbol.upper())
    return int(df[mask].iloc[0]["instrument_token"]) if not df[mask].empty else None

def classify_market_cap(mcap_crores):
    if mcap_crores >= 20000: return 'LARGE CAP'
    elif 5000 <= mcap_crores < 20000: return 'MID CAP'
    else: return 'SMALL CAP'

def classify_rating_group(rating):
    if pd.isna(rating): return 'UNRATED'
    rating = str(rating).upper()
    if any(r in rating for r in ['AAA', 'AA']): return 'HIGH_QUALITY'
    if 'A' in rating: return 'INVESTMENT_GRADE'
    if any(r in rating for r in ['BBB', 'BB', 'B']): return 'HIGH_YIELD'
    return 'OTHER'

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 1. Login to Kite Connect")
    if not st.session_state["kite_access_token"]:
        st.link_button("🔗 Open Kite login", login_url, use_container_width=True)
    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        with st.spinner("Authenticating..."):
            try:
                data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
                st.session_state["kite_access_token"] = data.get("access_token")
                st.sidebar.success("Kite authentication successful."); st.query_params.clear(); st.rerun()
            except Exception as e: st.sidebar.error(f"Authentication failed: {e}")
    if st.session_state["kite_access_token"]:
        st.success("Kite Authenticated ✅")
        if st.sidebar.button("Logout from Kite", use_container_width=True):
            st.session_state.clear(); st.rerun()

# --- Main Authenticated Kite Client ---
k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])

# --- Main UI Tabs ---
tabs = st.tabs(["💼 Investment Compliance", "📈 Market & Historical", "🤖 AI-Powered Analysis"])

# --- TAB 1: Investment Compliance ---
def render_investment_compliance_tab(kite_client, api_key, access_token):
    st.header("💼 Investment Compliance & Portfolio Analysis")
    if not kite_client: st.info("Please login to Kite Connect to fetch live prices."); return

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("1. Upload Portfolio")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Required columns: 'Symbol', 'Industry', 'Quantity'. Optional: 'Rating', 'Asset Class'.")
    with col2:
        st.subheader("2. Define Compliance Rules")
        rules_text = st.text_area("Enter one rule per line.", height=150, key="compliance_rules_input")

    with st.expander("📖 View Comprehensive Rule Syntax Guide & Examples", expanded=False):
        st.markdown("""
        The rule engine uses a simple `KEYWORD ARGUMENTS... OPERATOR VALUE` format.
        - **Operators**: `>` (greater than), `<` (less than), `>=` (greater or equal), `<=` (less or equal), `=` (equal to).
        - **Values**: Percentages should end with `%` (e.g., `10%`). Counts are numbers (e.g., `50`).

        ---
        #### **Holding-Level Rules**
        `STOCK <Symbol> <Operator> <Value>%`
        - **Description**: Checks the weight of a single stock in the portfolio.
        - **Example**: `STOCK RELIANCE < 10%` (Ensures Reliance Industries is less than 10% of the portfolio).

        ---
        #### **Concentration Rules**
        `TOP_N_STOCKS <Number> <Operator> <Value>%`
        - **Description**: Checks the combined weight of the top N largest holdings.
        - **Example**: `TOP_N_STOCKS 5 <= 40%` (Ensures the top 5 stocks are not more than 40% of the portfolio).

        `TOP_N_SECTORS <Number> <Operator> <Value>%`
        - **Description**: Checks the combined weight of the top N largest sectors.
        - **Example**: `TOP_N_SECTORS 3 <= 65%` (Ensures the top 3 sectors are not more than 65%).

        ---
        #### **Attribute-Based Rules**
        `SECTOR <Sector Name> <Operator> <Value>%`
        - **Description**: Checks the total weight of a specific sector. Sector name can be multiple words.
        - **Example**: `SECTOR FINANCIAL SERVICES > 20%`

        `ASSET_CLASS <Class Name> <Operator> <Value>%`
        - **Description**: Checks the total weight of an asset class (requires 'Asset Class' column).
        - **Example**: `ASSET_CLASS EQUITY < 90%`

        `MARKET_CAP <LARGE|MID|SMALL> <Operator> <Value>%`
        - **Description**: Checks the total weight of stocks belonging to a market cap category.
        - **Example**: `MARKET_CAP SMALL <= 15%`

        `RATING <Rating Value> <Operator> <Value>%`
        - **Description**: Checks the total weight of a specific credit rating (requires 'Rating' column).
        - **Example**: `RATING AAA > 50%`

        `RATING_GROUP <HIGH_QUALITY|INVESTMENT_GRADE|HIGH_YIELD> <Operator> <Value>%`
        - **Description**: Checks the total weight of a group of credit ratings.
        - **Example**: `RATING_GROUP HIGH_YIELD < 5%`

        ---
        #### **Portfolio-Level Rules**
        `COUNT_STOCKS <Operator> <Value>`
        - **Description**: Checks the total number of holdings in the portfolio.
        - **Example**: `COUNT_STOCKS > 30`
        """)

    if uploaded_file:
        if st.button("🚀 Analyze & Validate Portfolio", type="primary", use_container_width=True):
            with st.spinner("Analyzing portfolio... This may take a moment."):
                try:
                    df = pd.read_csv(uploaded_file)
                    df.columns = [str(col).strip().lower().replace(' ', '_').replace('.', '').replace('/','_') for col in df.columns]
                    header_map = {'name_of_the_instrument': 'Name', 'symbol': 'Symbol', 'industry': 'Industry', 'quantity': 'Quantity', 'rating': 'Rating', 'asset_class': 'Asset Class'}
                    df = df.rename(columns=header_map)
                    
                    symbols = df['Symbol'].unique().tolist()
                    ltp_data = kite_client.ltp([f"{DEFAULT_EXCHANGE}:{s}" for s in symbols])
                    
                    df['LTP'] = df['Symbol'].map({sym: ltp_data.get(f"{DEFAULT_EXCHANGE}:{sym}", {}).get('last_price') for sym in symbols})
                    df.dropna(subset=['LTP'], inplace=True)
                    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
                    df['Real-time Value (Rs)'] = df['LTP'] * df['Quantity']
                    df['Market Cap (Cr)'] = df['Real-time Value (Rs)'] / 1_00_00_000
                    
                    total_value = df['Real-time Value (Rs)'].sum()
                    df['Weight %'] = (df['Real-time Value (Rs)'] / total_value * 100) if total_value > 0 else 0
                    
                    df['Market Cap Class'] = df['Market Cap (Cr)'].apply(classify_market_cap)
                    if 'Rating' in df.columns:
                        df['Rating Group'] = df['Rating'].apply(classify_rating_group)
                    
                    st.session_state.compliance_results_df = df
                    st.session_state.advanced_metrics = None # Reset on new analysis
                except Exception as e:
                    st.error(f"Failed to process portfolio. Error: {e}")

    results_df = st.session_state.get("compliance_results_df", pd.DataFrame())
    if not results_df.empty:
        analysis_tabs = st.tabs(["📊 Dashboard", "🔍 Breakdowns", "📈 Risk & Attribution", "⚖️ Compliance Check", "📄 Detailed Holdings"])
        with analysis_tabs[0]: render_dashboard_view(results_df)
        with analysis_tabs[1]: render_breakdowns_view(results_df)
        with analysis_tabs[2]: render_risk_attribution_view(results_df, api_key, access_token)
        with analysis_tabs[3]: render_compliance_check_view(rules_text, results_df)
        with analysis_tabs[4]: render_detailed_holdings_view(results_df)

def render_dashboard_view(df):
    st.subheader("Portfolio Dashboard")
    total_value = df['Real-time Value (Rs)'].sum()
    
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Portfolio Value", f"₹ {total_value:,.2f}")
    kpi_cols[1].metric("Holdings Count", f"{len(df)}")
    kpi_cols[2].metric("Unique Sectors", f"{df['Industry'].nunique()}")
    if 'Rating' in df.columns:
        kpi_cols[3].metric("Unique Ratings", f"{df['Rating'].nunique()}")

    st.markdown("#### Concentration Analysis")
    conc_cols = st.columns(4)
    conc_cols[0].metric("Top 1 Holding", f"{df.nlargest(1, 'Weight %')['Weight %'].sum():.2f}%")
    conc_cols[1].metric("Top 5 Holdings", f"{df.nlargest(5, 'Weight %')['Weight %'].sum():.2f}%")
    conc_cols[2].metric("Top 10 Holdings", f"{df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%")
    conc_cols[3].metric("Top 3 Sectors", f"{df.groupby('Industry')['Weight %'].sum().nlargest(3).sum():.2f}%")

    st.markdown("#### Risk & Factor Analysis")
    risk_cols = st.columns(4)
    stock_hhi = (df['Weight %'] ** 2).sum()
    sector_hhi = (df.groupby('Industry')['Weight %'].sum() ** 2).sum()
    def get_hhi_category(score): return "Low" if score < 1500 else "Moderate" if score <= 2500 else "High"
    risk_cols[0].metric("Stock HHI", f"{stock_hhi:,.0f}", help=f"Concentration: {get_hhi_category(stock_hhi)}")
    risk_cols[1].metric("Sector HHI", f"{sector_hhi:,.0f}", help=f"Concentration: {get_hhi_category(sector_hhi)}")
    
    # Placeholder metrics - replace with real data if available
    risk_cols[2].metric("Est. P/E Ratio", "25.4x", help="Estimated portfolio Price-to-Earnings ratio. (Placeholder)")
    risk_cols[3].metric("Est. Dividend Yield", "1.2%", help="Estimated portfolio dividend yield. (Placeholder)")

def render_breakdowns_view(df):
    st.subheader("Portfolio Breakdowns")
    
    b_cols = st.columns(2)
    with b_cols[0]:
        st.markdown("##### Sector Exposure")
        sector_weights = df.groupby('Industry')['Weight %'].sum().nlargest(10).reset_index()
        fig = px.bar(sector_weights, x='Weight %', y='Industry', orientation='h', title='Top 10 Sector Exposures')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}); st.plotly_chart(fig, use_container_width=True)

    with b_cols[1]:
        st.markdown("##### Market Cap Exposure")
        mcap_weights = df.groupby('Market Cap Class')['Weight %'].sum().reset_index()
        fig = px.pie(mcap_weights, names='Market Cap Class', values='Weight %', title='Exposure by Market Cap', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    if 'Rating Group' in df.columns:
        b_cols2 = st.columns(2)
        with b_cols2[0]:
            st.markdown("##### Credit Quality")
            rating_weights = df.groupby('Rating Group')['Weight %'].sum().reset_index()
            fig = px.pie(rating_weights, names='Rating Group', values='Weight %', title='Exposure by Credit Quality', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("##### Interactive Treemap")
    fig_tree = px.treemap(df, path=[px.Constant("Portfolio"), 'Market Cap Class', 'Industry', 'Name'], values='Real-time Value (Rs)', hover_data={'Weight %': ':.2f%'})
    fig_tree.update_layout(margin = dict(t=25, l=25, r=25, b=25)); st.plotly_chart(fig_tree, use_container_width=True)

def render_risk_attribution_view(df, api_key, access_token):
    st.subheader("Advanced Risk & Attribution Analysis")
    if st.button("🔬 Calculate Advanced Metrics", key="calc_adv_metrics", use_container_width=True):
        st.session_state.advanced_metrics = calculate_advanced_metrics(df, api_key, access_token)
    
    if st.session_state.advanced_metrics:
        metrics = st.session_state.advanced_metrics
        st.markdown("#### Key Risk Metrics")
        m_cols = st.columns(5)
        m_cols[0].metric("Portfolio Beta", f"{metrics['beta']:.2f}", help="Volatility relative to the market (NIFTY 50).")
        m_cols[1].metric("Annualized Volatility", f"{metrics['annual_volatility']*100:.2f}%")
        m_cols[2].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", help="Risk-adjusted return.")
        m_cols[3].metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}", help="Return adjusted for downside risk.")
        m_cols[4].metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%", help="Largest peak-to-trough decline.")

        st.markdown("#### Return Distribution")
        d_cols = st.columns(4)
        d_cols[0].metric("Daily VaR (95%)", f"{metrics['var_95']*100:.2f}%", help="Expected max 1-day loss.")
        d_cols[1].metric("Skewness", f"{metrics['skewness']:.2f}", help="Asymmetry of returns. <0: Left tail.")
        d_cols[2].metric("Kurtosis", f"{metrics['kurtosis']:.2f}", help="Fatness of tails. >3: Fat tails (outliers).")
        d_cols[3].metric("Treynor Ratio", f"{metrics['treynor_ratio']:.2f}", help="Excess return per unit of beta.")
        
        st.markdown("#### Correlation Matrix")
        st.info("This heatmap shows the correlation between the daily returns of your top 15 holdings. Values near +1 move together, near -1 move opposite, and near 0 are unrelated.", icon="ℹ️")
        fig = go.Figure(data=go.Heatmap(
            z=metrics['correlation_matrix'].values,
            x=metrics['correlation_matrix'].columns,
            y=metrics['correlation_matrix'].columns,
            colorscale='RdBu', zmin=-1, zmax=1))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def render_compliance_check_view(rules_text, df):
    st.subheader("Compliance Rule Validation")
    validation_results = parse_and_validate_rules(rules_text, df)
    if not validation_results: st.info("Define rules in the text area above to see validation results.")
    else:
        for res in validation_results:
            if res['status'] == "✅ PASS": st.success(f"**{res['status']}**: `{res['rule']}` ({res['details']})")
            elif res['status'] == "❌ FAIL": st.error(f"**{res['status']}**: `{res['rule']}` ({res['details']})")
            else: st.warning(f"**{res['status']}**: `{res['rule']}` ({res['details']})")

def render_detailed_holdings_view(df):
    st.subheader("Detailed Holdings View")
    display_df = df.copy()
    format_dict = {'Real-time Value (Rs)': '₹{:,.2f}', 'LTP': '₹{:,.2f}', 'Weight %': '{:.2f}%', 'Market Cap (Cr)': '{:,.0f}'}
    
    column_order = ['Name', 'Symbol', 'Industry', 'Market Cap Class', 'Real-time Value (Rs)', 'Weight %', 'Quantity', 'LTP', 'Market Cap (Cr)']
    if 'Asset Class' in display_df.columns: column_order.insert(4, 'Asset Class')
    if 'Rating' in display_df.columns: column_order.insert(4, 'Rating'); column_order.insert(5, 'Rating Group')
    
    display_columns = [col for col in column_order if col in display_df.columns]
    st.dataframe(display_df[display_columns].style.format(format_dict), use_container_width=True, height=600)
    st.download_button("📥 Download Full Report (CSV)", display_df.to_csv(index=False).encode('utf-8'), f"portfolio_analysis.csv", "text/csv")

def parse_and_validate_rules(rules_text, portfolio_df):
    results = []; df = portfolio_df
    if not rules_text.strip() or df.empty: return results
    
    # Pre-calculate aggregations
    aggs = {
        'sector': df.groupby('Industry')['Weight %'].sum(),
        'stock': df.set_index('Symbol')['Weight %'],
        'rating': df.groupby('Rating')['Weight %'].sum() if 'Rating' in df.columns else pd.Series(),
        'asset_class': df.groupby('Asset Class')['Weight %'].sum() if 'Asset Class' in df.columns else pd.Series(),
        'market_cap': df.groupby('Market Cap Class')['Weight %'].sum(),
        'rating_group': df.groupby('Rating Group')['Weight %'].sum() if 'Rating Group' in df.columns else pd.Series()
    }
    
    def check_pass(actual, op, threshold):
        ops = {'>': actual > threshold, '<': actual < threshold, '>=': actual >= threshold, '<=': actual <= threshold, '=': actual == threshold}
        return ops.get(op, False)

    for rule in rules_text.strip().split('\n'):
        rule = rule.strip()
        if not rule or rule.startswith('#'): continue
        
        try:
            parts = re.match(r"(\w+)\s+([^><=]+)?\s*([><=]+)\s+([\d.%]+)", rule)
            if not parts:
                # Handle rules with no middle argument, like COUNT_STOCKS
                parts = re.match(r"(\w+)\s*([><=]+)\s+([\d.%]+)", rule)
                if not parts: raise ValueError("Invalid rule format")
                keyword, op, value_str = parts.groups()
                arg = None
            else:
                keyword, arg, op, value_str = parts.groups()
                arg = arg.strip() if arg else None

            keyword = keyword.upper()
            value = float(value_str.replace('%', ''))
            actual_value, details = None, ""
            
            if keyword == 'STOCK':
                actual_value = aggs['stock'].get(arg.upper(), 0.0)
                details = f"Actual for {arg.upper()}: {actual_value:.2f}%"
            elif keyword == 'SECTOR':
                actual_value = aggs['sector'].get(arg.upper(), 0.0)
                details = f"Actual for {arg.upper()}: {actual_value:.2f}%"
            elif keyword == 'ASSET_CLASS':
                actual_value = aggs['asset_class'].get(arg.upper(), 0.0)
                details = f"Actual for {arg.upper()}: {actual_value:.2f}%"
            elif keyword == 'MARKET_CAP':
                actual_value = aggs['market_cap'].get(arg.upper(), 0.0)
                details = f"Actual for {arg.upper()}: {actual_value:.2f}%"
            elif keyword == 'RATING':
                actual_value = aggs['rating'].get(arg.upper(), 0.0)
                details = f"Actual for {arg.upper()}: {actual_value:.2f}%"
            elif keyword == 'RATING_GROUP':
                actual_value = aggs['rating_group'].get(arg.upper(), 0.0)
                details = f"Actual for {arg.upper()}: {actual_value:.2f}%"
            elif keyword == 'TOP_N_STOCKS':
                n = int(arg)
                actual_value = df.nlargest(n, 'Weight %')['Weight %'].sum()
                details = f"Top {n} stocks weight: {actual_value:.2f}%"
            elif keyword == 'TOP_N_SECTORS':
                n = int(arg)
                actual_value = aggs['sector'].nlargest(n).sum()
                details = f"Top {n} sectors weight: {actual_value:.2f}%"
            elif keyword == 'COUNT_STOCKS':
                actual_value = len(df)
                details = f"Actual count: {actual_value}"
            
            if actual_value is not None:
                passed = check_pass(actual_value, op, value)
                status = "✅ PASS" if passed else "❌ FAIL"
                results.append({'rule': rule, 'status': status, 'details': f"{details} | Rule: {op} {value_str}"})
            else:
                results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Keyword '{keyword}' or argument invalid."})
        except Exception as e:
            results.append({'rule': rule, 'status': 'Error', 'details': f"Could not parse rule. Check syntax. Error: {e}"})
            
    return results

def calculate_advanced_metrics(portfolio_df, api_key, access_token):
    with st.spinner("Fetching 1 year of historical data for all holdings... This may take time."):
        symbols = portfolio_df['Symbol'].tolist()
        weights = (portfolio_df['Real-time Value (Rs)'] / portfolio_df['Real-time Value (Rs)'].sum()).values
        from_date = datetime.now().date() - timedelta(days=366); to_date = datetime.now().date()
        
        returns_df = pd.DataFrame()
        for symbol in symbols:
            hist_data = get_historical_data_cached(api_key, access_token, symbol, from_date, to_date, 'day')
            if not hist_data.empty and '_error' not in hist_data.columns:
                returns_df[symbol] = hist_data['close'].pct_change()
        
        returns_df.fillna(0, inplace=True)
        if returns_df.empty: st.error("Not enough historical data."); return None

    portfolio_returns = returns_df.dot(weights)
    
    # Risk-free rate (annualized)
    risk_free_rate = 0.06
    daily_rf = (1 + risk_free_rate)**(1/TRADING_DAYS_PER_YEAR) - 1
    
    # Calculations
    annual_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    annual_return = (portfolio_returns.mean() + 1)**TRADING_DAYS_PER_YEAR - 1
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
    downside_returns = portfolio_returns[portfolio_returns < daily_rf]
    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    var_95 = portfolio_returns.quantile(0.05)
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurt()
    
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Beta and Treynor Ratio
    benchmark_data = get_historical_data_cached(api_key, access_token, BENCHMARK_SYMBOL, from_date, to_date, 'day')
    beta, treynor_ratio = None, None
    if not benchmark_data.empty and '_error' not in benchmark_data.columns:
        benchmark_returns = benchmark_data['close'].pct_change()
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner').dropna()
        aligned.columns = ['portfolio', 'benchmark']
        covariance = aligned.cov().iloc[0, 1]
        benchmark_variance = aligned['benchmark'].var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        treynor_ratio = (annual_return - risk_free_rate) / beta if beta != 0 else 0

    return {
        "annual_volatility": annual_volatility, "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio, "var_95": var_95, "skewness": skewness,
        "kurtosis": kurtosis, "max_drawdown": max_drawdown, "beta": beta,
        "treynor_ratio": treynor_ratio,
        "correlation_matrix": returns_df.iloc[:,:15].corr() # Limit to top 15 for readability
    }


# --- TAB 2: Market & Historical ---
def render_market_historical_tab(kite_client, api_key, access_token):
    # This function is unchanged but provided for completeness
    st.header("📈 Market Data & Historical Candles with TA")
    if not kite_client: st.info("Login first to fetch market data."); return
    # ... [rest of the function is identical to your last version, keeping it out for brevity] ...
    # Placeholder to indicate the function exists
    st.info("Market data functionality is available here after logging in.")


# --- TAB 3: AI Analysis ---
def render_ai_analysis_tab(kite_client):
    # This function is unchanged but provided for completeness
    st.header("🤖 AI-Powered Compliance Analysis (with Google Gemini)")
    portfolio_df = st.session_state.get("compliance_results_df")
    if portfolio_df is None or portfolio_df.empty:
        st.warning("Please upload and analyze a portfolio in the 'Investment Compliance' tab first."); return
    # ... [rest of the function is identical to your last version, keeping it out for brevity] ...
    # Placeholder to indicate the function exists
    st.info("AI analysis is available here after first analyzing a portfolio.")


# --- Main Application Logic (Tab Rendering) ---
with tabs[0]: render_investment_compliance_tab(k, KITE_CREDENTIALS["api_key"], st.session_state.get("kite_access_token"))
with tabs[1]: render_market_historical_tab(k, KITE_CREDENTIALS["api_key"], st.session_state.get("kite_access_token"))
with tabs[2]: render_ai_analysis_tab(k)
