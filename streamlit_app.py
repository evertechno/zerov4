# app.py
# Invsion Connect — Single-file Streamlit app
# - Supabase Auth (login/signup) + per-user persistent saves
# - KiteConnect login (to fetch holdings & prices)
# - Compliance engine + advanced risk analytics
# - Gemini AI (Google GenerativeAI) for SID/KIM-based analysis
# - FULL payload save: portfolio, breaches, rules, security-level checks, risk metrics, treemap & concentration, AI output

import os
import re
import json
import base64
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ta  # pip install ta
import fitz  # PyMuPDF

# --- External SDKs (hard dependencies) ---
try:
    import google.generativeai as genai
except ImportError:
    st.error("Missing dependency: google-generativeai. Install with: pip install google-generativeai")
    st.stop()

try:
    from kiteconnect import KiteConnect
except ImportError:
    st.error("Missing dependency: kiteconnect. Install with: pip install kiteconnect")
    st.stop()

try:
    from supabase import create_client, Client
except ImportError:
    st.error("Missing dependency: supabase. Install with: pip install supabase")
    st.stop()

# --- Page ---
st.set_page_config(
    page_title="Invsion Connect - Compliance & AI Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Invsion Connect")
st.caption("Professional Portfolio Compliance & Analysis Platform (Supabase + Kite + Gemini)")

# --- Constants / Session Defaults ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_EXCHANGE = "NSE"
BENCHMARK_SYMBOL = "NIFTY 50"

_defaults = {
    "kite_access_token": None,
    "kite_login_response": None,
    "holdings_data": None,
    "compliance_results_df": pd.DataFrame(),
    "security_level_compliance": pd.DataFrame(),
    "advanced_metrics": None,
    "ai_analysis_response": None,
    "breach_alerts": [],
    "cfg_thresholds": {},
    "last_concentration": {},
    "last_validation_results": [],
    "sb_client": None,
    "sb_user": None,
    "sb_session": None,
    "last_saved_run_id": None
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Credentials loader: supports Streamlit secrets or ENV ---
def _get_conf():
    # Prefer st.secrets, fallback to env
    def gs(path, default=None):
        try:
            return st.secrets.get(path.split(":")[0], {}).get(path.split(":")[1], default)
        except Exception:
            return default

    kite_api_key = gs("kite:api_key", os.getenv("KITE_API_KEY"))
    kite_api_secret = gs("kite:api_secret", os.getenv("KITE_API_SECRET"))
    kite_redirect = gs("kite:redirect_uri", os.getenv("KITE_REDIRECT_URI"))

    gemini_key = gs("google_gemini:api_key", os.getenv("GEMINI_API_KEY"))

    sb_url = gs("supabase:url", os.getenv("SUPABASE_URL"))
    sb_anon = gs("supabase:anon_key", os.getenv("SUPABASE_ANON_KEY"))

    missing = []
    if not kite_api_key or not kite_api_secret or not kite_redirect:
        missing.append("Kite (api_key, api_secret, redirect_uri)")
    if not gemini_key:
        missing.append("Gemini (api_key)")
    if not sb_url or not sb_anon:
        missing.append("Supabase (url, anon_key)")

    if missing:
        st.error("Missing credentials. Provide via `.streamlit/secrets.toml` or environment variables.\nMissing: " + ", ".join(missing))
        st.stop()

    return {
        "kite": {"api_key": kite_api_key, "api_secret": kite_api_secret, "redirect_uri": kite_redirect},
        "gemini": {"api_key": gemini_key},
        "supabase": {"url": sb_url, "anon_key": sb_anon}
    }

CONF = _get_conf()
genai.configure(api_key=CONF["gemini"]["api_key"])

# --- Supabase client ---
@st.cache_resource(show_spinner=False)
def _sb_client(url, anon) -> Client:
    return create_client(url, anon)

st.session_state["sb_client"] = _sb_client(CONF["supabase"]["url"], CONF["supabase"]["anon_key"])

# --- Supabase auth helpers ---
def sb_signup(email: str, password: str):
    try:
        return st.session_state["sb_client"].auth.sign_up({"email": email, "password": password})
    except Exception as e:
        st.error(f"Sign up failed: {e}")
        return None

def sb_login(email: str, password: str):
    try:
        res = st.session_state["sb_client"].auth.sign_in_with_password({"email": email, "password": password})
        st.session_state["sb_session"] = res.session
        st.session_state["sb_user"] = res.user
        return res
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None

def sb_logout():
    try:
        st.session_state["sb_client"].auth.sign_out()
    except Exception as e:
        st.warning(f"Sign out warning: {e}")
    st.session_state["sb_session"] = None
    st.session_state["sb_user"] = None

def require_sb_user() -> bool:
    if not st.session_state["sb_user"]:
        st.warning("Please login to Supabase to save your runs.")
        return False
    return True

def dictify_df(df):
    if df is None or isinstance(df, bool):
        return None
    if isinstance(df, pd.DataFrame):
        return df.to_dict(orient="records")
    return None

def save_full_payload_to_supabase(run_label: str | None, payload: dict):
    if not require_sb_user():
        return None
    try:
        data = {
            "user_id": st.session_state["sb_user"].id,
            "run_label": run_label,
            "payload": payload
        }
        # Requires table: analysis_runs(id, user_id uuid, created_at timestamptz default now(), run_label text, payload jsonb)
        res = st.session_state["sb_client"].table("analysis_runs").insert(data).execute()
        if hasattr(res, "data") and res.data:
            rid = res.data[0].get("id")
            st.session_state["last_saved_run_id"] = rid
            st.success(f"✅ Saved analysis run (ID: {rid})")
            return rid
        st.warning("Insert returned no data — check Supabase RLS/permissions.")
        return None
    except Exception as e:
        st.error(f"Supabase save failed: {e}")
        return None

# --- KiteConnect ---
@st.cache_resource(ttl=3600)
def _kc(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

kite_unauth = _kc(CONF["kite"]["api_key"])
login_url = kite_unauth.login_url()

def get_kite_auth_client(api_key: str, access_token: str | None) -> KiteConnect | None:
    if not access_token:
        return None
    kc = KiteConnect(api_key=api_key)
    kc.set_access_token(access_token)
    return kc

@st.cache_data(ttl=86400, show_spinner=False)
def load_instruments_cached(api_key: str, access_token: str, exchange: str | None = None) -> pd.DataFrame:
    kc = get_kite_auth_client(api_key, access_token)
    if not kc:
        return pd.DataFrame({"_error": ["Kite not authenticated."]})
    try:
        data = kc.instruments(exchange) if exchange else kc.instruments()
        df = pd.DataFrame(data)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        sel = ['instrument_token', 'tradingsymbol', 'name', 'exchange']
        df = df[[c for c in sel if c in df.columns]]
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [f"Failed to load instruments: {e}"]})

def _find_instrument_token(df: pd.DataFrame, symbol: str, exchange: str = DEFAULT_EXCHANGE) -> int | None:
    if df.empty:
        return None
    ex = df.get("exchange", pd.Series(dtype=str)).astype(str).str.upper()
    ts = df.get("tradingsymbol", pd.Series(dtype=str)).astype(str).str.upper()
    mask = (ex == exchange.upper()) & (ts == symbol.upper())
    hits = df[mask]
    if hits.empty:
        return None
    return int(hits.iloc[0]["instrument_token"])

@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str,
                               from_date: datetime.date, to_date: datetime.date,
                               interval: str, exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    kc = get_kite_auth_client(api_key, access_token)
    if not kc:
        return pd.DataFrame({"_error": ["Kite not authenticated."]})
    instruments_df = load_instruments_cached(api_key, access_token)
    token = _find_instrument_token(instruments_df, symbol, exchange)
    if not token and symbol in ["NIFTY BANK", "NIFTYBANK", "BANKNIFTY", BENCHMARK_SYMBOL, "SENSEX"]:
        index_exchange = "NSE" if symbol != "SENSEX" else "BSE"
        instruments_secondary = load_instruments_cached(api_key, access_token, index_exchange)
        token = _find_instrument_token(instruments_secondary, symbol, index_exchange)
    if not token:
        return pd.DataFrame({"_error": [f"Instrument token not found for {symbol}."]})
    try:
        data = kc.historical_data(
            token,
            from_date=datetime.combine(from_date, datetime.min.time()),
            to_date=datetime.combine(to_date, datetime.max.time()),
            interval=interval
        )
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=['close'])
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [str(e)]})

# --- Performance metrics (basic) ---
def calculate_performance_metrics(returns_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    if returns_series.empty or len(returns_series) < 2:
        return {}
    daily = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    if daily.empty:
        return {}
    cumret = (1 + daily).cumprod() - 1
    total_return = float((cumret.iloc[-1] * 100) if not cumret.empty else 0)
    ann_return = float(((1 + daily.mean()) ** TRADING_DAYS_PER_YEAR - 1) * 100)
    ann_vol = float(daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)
    sharpe = float((ann_return - risk_free_rate) / ann_vol) if ann_vol > 0 else np.nan
    if not cumret.empty:
        mdd = float((((1 + cumret).cummax() - (1 + cumret)) / (1 + cumret).cummax()).max() * 100)
    else:
        mdd = np.nan
    def _r(x):
        if isinstance(x, float) and np.isnan(x): return np.nan
        return round(x, 4)
    return {
        "Total Return (%)": _r(total_return),
        "Annualized Return (%)": _r(ann_return),
        "Annualized Volatility (%)": _r(ann_vol),
        "Sharpe Ratio": _r(sharpe),
        "Max Drawdown (%)": _r(mdd)
    }

# --- Compliance engine ---
def parse_and_validate_rules_enhanced(rules_text: str, portfolio_df: pd.DataFrame):
    results = []
    if not rules_text or portfolio_df.empty:
        return results

    sector_wt = portfolio_df.groupby('Industry')['Weight %'].sum()
    stock_wt = portfolio_df.set_index('Symbol')['Weight %']
    rating_wt = portfolio_df.groupby('Rating')['Weight %'].sum() if 'Rating' in portfolio_df.columns else pd.Series(dtype=float)
    asset_wt = portfolio_df.groupby('Asset Class')['Weight %'].sum() if 'Asset Class' in portfolio_df.columns else pd.Series(dtype=float)
    mcap_wt = portfolio_df.groupby('Market Cap')['Weight %'].sum() if 'Market Cap' in portfolio_df.columns else pd.Series(dtype=float)

    def chk(actual, op, thr):
        if op == '>': return actual > thr
        if op == '<': return actual < thr
        if op == '>=': return actual >= thr
        if op == '<=': return actual <= thr
        if op == '=': return actual == thr
        return False

    for line in rules_text.strip().split('\n'):
        rule = line.strip()
        if not rule or rule.startswith('#'):
            continue
        parts = re.split(r'\s+', rule)
        rtype = parts[0].upper()
        try:
            if len(parts) < 3:
                results.append({'rule': rule, 'status': 'Error', 'details': 'Invalid format.', 'severity': 'N/A'})
                continue
            op = parts[-2]
            if op not in ['>', '<', '>=', '<=', '=']:
                results.append({'rule': rule, 'status': 'Error', 'details': f"Invalid operator '{op}'.", 'severity': 'N/A'})
                continue
            thr = float(parts[-1].replace('%', ''))
            actual = None
            details = ""

            if rtype == 'STOCK' and len(parts) == 4:
                sym = parts[1].upper()
                if sym in stock_wt.index:
                    actual = float(stock_wt.get(sym, 0.0))
                    details = f"Actual {sym}: {actual:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Symbol '{sym}' not found.", 'severity': 'N/A'})
                    continue

            elif rtype == 'SECTOR':
                sec_name = ' '.join(parts[1:-2]).upper()
                match = next((s for s in sector_wt.index if str(s).upper() == sec_name), None)
                if match:
                    actual = float(sector_wt.get(match, 0.0))
                    details = f"Actual {match}: {actual:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Sector '{sec_name}' not found.", 'severity': 'N/A'})
                    continue

            elif rtype == 'RATING':
                r = ' '.join(parts[1:-2]).upper()
                actual = float(rating_wt.get(r, 0.0))
                details = f"Actual {r}: {actual:.2f}%"

            elif rtype == 'ASSET_CLASS':
                c = ' '.join(parts[1:-2]).upper()
                actual = float(asset_wt.get(c, 0.0))
                details = f"Actual {c}: {actual:.2f}%"

            elif rtype == 'MARKET_CAP':
                c = ' '.join(parts[1:-2]).upper()
                actual = float(mcap_wt.get(c, 0.0))
                details = f"Actual {c}: {actual:.2f}%"

            elif rtype == 'TOP_N_STOCKS' and len(parts) == 4:
                n = int(parts[1])
                actual = float(portfolio_df.nlargest(n, 'Weight %')['Weight %'].sum())
                details = f"Top {n} stocks: {actual:.2f}%"

            elif rtype == 'TOP_N_SECTORS' and len(parts) == 4:
                n = int(parts[1])
                actual = float(sector_wt.nlargest(n).sum())
                details = f"Top {n} sectors: {actual:.2f}%"

            elif rtype == 'COUNT_STOCKS' and len(parts) == 3:
                actual = float(len(portfolio_df))
                details = f"Actual count: {int(actual)}"

            elif rtype == 'COUNT_SECTORS' and len(parts) == 3:
                actual = float(portfolio_df['Industry'].nunique())
                details = f"Actual count: {int(actual)}"

            elif rtype == 'ISSUER_GROUP':
                g = ' '.join(parts[1:-2]).upper()
                if 'Issuer Group' in portfolio_df.columns:
                    actual = float(portfolio_df[portfolio_df['Issuer Group'].str.upper() == g]['Weight %'].sum())
                    details = f"Actual {g}: {actual:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Issuer Group' not found.", 'severity': 'N/A'})
                    continue

            elif rtype == 'MIN_LIQUIDITY' and len(parts) == 4:
                sym = parts[1].upper()
                if 'Avg Volume (90d)' in portfolio_df.columns:
                    row = portfolio_df[portfolio_df['Symbol'] == sym]
                    if not row.empty:
                        actual = float(row['Avg Volume (90d)'].values[0])
                        details = f"Volume {sym}: {actual:,.0f}"
                    else:
                        results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': f"Symbol '{sym}' not found.", 'severity': 'N/A'})
                        continue
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Avg Volume (90d)' not found.", 'severity': 'N/A'})
                    continue

            elif rtype == 'UNRATED_EXPOSURE' and len(parts) == 3:
                if 'Rating' in portfolio_df.columns:
                    unr = portfolio_df['Rating'].isin(['UNRATED', 'NR', 'NOT RATED', ''])
                    actual = float(portfolio_df[unr]['Weight %'].sum())
                    details = f"Unrated: {actual:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Rating' not found.", 'severity': 'N/A'})
                    continue

            elif rtype == 'FOREIGN_EXPOSURE' and len(parts) == 3:
                if 'Country' in portfolio_df.columns:
                    foreign = portfolio_df['Country'].str.upper() != 'INDIA'
                    actual = float(portfolio_df[foreign]['Weight %'].sum())
                    details = f"Foreign: {actual:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Country' not found.", 'severity': 'N/A'})
                    continue

            elif rtype == 'DERIVATIVES_EXPOSURE' and len(parts) == 3:
                if 'Instrument Type' in portfolio_df.columns:
                    dd = portfolio_df['Instrument Type'].str.upper().isin(['FUTURES', 'OPTIONS', 'SWAPS'])
                    actual = float(portfolio_df[dd]['Weight %'].sum())
                    details = f"Derivatives: {actual:.2f}%"
                else:
                    results.append({'rule': rule, 'status': '⚠️ Invalid', 'details': "Column 'Instrument Type' not found.", 'severity': 'N/A'})
                    continue

            else:
                results.append({'rule': rule, 'status': 'Error', 'details': 'Unrecognized rule format.', 'severity': 'N/A'})
                continue

            passed = chk(actual, op, thr)
            status = "✅ PASS" if passed else "❌ FAIL"
            if not passed:
                diff = abs(actual - thr)
                if diff > thr * 0.2: sev = "🔴 Critical"
                elif diff > thr * 0.1: sev = "🟠 High"
                else: sev = "🟡 Medium"
            else:
                sev = "✅ Compliant"

            results.append({
                "rule": rule,
                "status": status,
                "details": f"{details} | Rule: {op} {thr}",
                "severity": sev,
                "actual_value": actual,
                "threshold": thr,
                "breach_amount": (actual - thr) if not passed else 0
            })

        except (ValueError, IndexError) as e:
            results.append({'rule': rule, 'status': 'Error', 'details': f"Parse error: {e}", 'severity': 'N/A'})

    return results

def calculate_security_level_compliance(portfolio_df: pd.DataFrame, rules_config: dict):
    if portfolio_df.empty:
        return pd.DataFrame()
    out = portfolio_df.copy()
    single_lim = float(rules_config.get('single_stock_limit', 10.0))
    out['Stock Limit Breach'] = out['Weight %'].apply(lambda x: '❌ Breach' if x > single_lim else '✅ Compliant')
    out['Stock Limit Gap (%)'] = single_lim - out['Weight %']

    if 'Avg Volume (90d)' in out.columns:
        minliq = float(rules_config.get('min_liquidity', 100000))
        out['Liquidity Status'] = out['Avg Volume (90d)'].apply(lambda x: '✅ Adequate' if float(x) >= minliq else '⚠️ Low')

    if 'Rating' in out.columns:
        ok = rules_config.get('min_rating', ['AAA', 'AA+', 'AA', 'AA-', 'A+'])
        out['Rating Compliance'] = out['Rating'].apply(lambda x: '✅ Compliant' if str(x).upper() in ok else '⚠️ Below Threshold')

    out['Concentration Risk'] = out['Weight %'].apply(lambda x: '🔴 High' if x > 8 else ('🟡 Medium' if x > 5 else '🟢 Low'))
    return out

def calculate_advanced_metrics(portfolio_df: pd.DataFrame, api_key: str, access_token: str):
    symbols = portfolio_df['Symbol'].tolist()
    from_date = datetime.now().date() - timedelta(days=366)
    to_date = datetime.now().date()

    rets = pd.DataFrame()
    failed = []
    prog = st.progress(0, text="Fetching historical data...")

    for i, sym in enumerate(symbols):
        df = get_historical_data_cached(api_key, access_token, sym, from_date, to_date, 'day')
        if not df.empty and '_error' not in df.columns:
            rets[sym] = df['close'].pct_change()
        else:
            failed.append(sym)
        prog.progress((i + 1) / max(1, len(symbols)), text=f"Fetched: {sym}")

    if failed:
        st.warning("No history for: " + ", ".join(failed))

    rets.dropna(how='all', inplace=True)
    rets.fillna(0, inplace=True)
    if rets.empty:
        prog.empty()
        st.error("Insufficient data for metrics.")
        return None

    succ_syms = rets.columns.tolist()
    port_ok = portfolio_df.set_index('Symbol').reindex(succ_syms).reset_index()
    tot_val = port_ok['Real-time Value (Rs)'].sum()
    if tot_val == 0:
        prog.empty()
        st.error("Zero total value for successful symbols.")
        return None

    weights = (port_ok['Real-time Value (Rs)'] / tot_val).values
    port_rets = rets.dot(weights)

    var_95 = float(port_rets.quantile(0.05))
    var_99 = float(port_rets.quantile(0.01))
    cvar_95 = float(port_rets[port_rets <= var_95].mean())

    bench = get_historical_data_cached(api_key, access_token, BENCHMARK_SYMBOL, from_date, to_date, 'day')
    beta = alpha = te = ir = None
    if not bench.empty and '_error' not in bench.columns:
        b = bench['close'].pct_change()
        aligned = pd.concat([port_rets, b], axis=1, join='inner').dropna()
        aligned.columns = ['port', 'bench']
        if not aligned.empty:
            cov = aligned.cov().iloc[0, 1]
            varb = aligned['bench'].var()
            beta = float(cov / varb) if varb > 0 else None
            pr = ((1 + aligned['port'].mean()) ** TRADING_DAYS_PER_YEAR - 1)
            br = ((1 + aligned['bench'].mean()) ** TRADING_DAYS_PER_YEAR - 1)
            rf = 0.06
            if beta is not None:
                alpha = float(pr - (rf + beta * (br - rf)))
            diff = aligned['port'] - aligned['bench']
            te_val = diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            te = float(te_val)
            if te > 0:
                ir = float((pr - br) / te_val)
    else:
        st.error(f"Benchmark {BENCHMARK_SYMBOL} unavailable — Beta/Alpha skipped.")

    ddown = port_rets[port_rets < 0]
    dstd = ddown.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(ddown) > 0 else 0
    pr_ann = ((1 + port_rets.mean()) ** TRADING_DAYS_PER_YEAR - 1)
    sortino = float((pr_ann - 0.06) / dstd) if dstd > 0 else None

    corr = rets.corr()
    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    avg_corr = None if np.isnan(avg_corr) else float(avg_corr)

    pvol = float(port_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    wvol = float(np.sum(weights * rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)))
    div_ratio = float(wvol / pvol) if pvol > 0 else None

    prog.empty()
    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "beta": beta,
        "alpha": alpha,
        "tracking_error": te,
        "information_ratio": ir,
        "sortino_ratio": sortino,
        "avg_correlation": avg_corr,
        "diversification_ratio": div_ratio,
        "portfolio_volatility": pvol
    }

# --- Sidebar: Supabase + Kite ---
with st.sidebar:
    st.subheader("🔐 Supabase")
    if not st.session_state["sb_user"]:
        with st.form("sb_auth", clear_on_submit=False):
            e = st.text_input("Email")
            p = st.text_input("Password", type="password")
            c1, c2 = st.columns(2)
            do_login = c1.form_submit_button("Login", use_container_width=True)
            do_signup = c2.form_submit_button("Sign Up", use_container_width=True)
        if do_signup and e and p:
            r = sb_signup(e, p)
            if r and getattr(r, "user", None):
                st.success("Signup successful. Login now.")
        if do_login and e and p:
            r = sb_login(e, p)
            if r and getattr(r, "user", None):
                st.success(f"Welcome, {e}!")
    else:
        st.success(f"Logged in: {st.session_state['sb_user'].email}")
        if st.button("Logout", use_container_width=True):
            sb_logout()
            st.rerun()

    st.divider()
    st.subheader("Kite Connect")
    if not st.session_state["kite_access_token"]:
        st.link_button("🔗 Open Kite login", login_url, use_container_width=True)
    req_tok = st.query_params.get("request_token")
    if req_tok and not st.session_state["kite_access_token"]:
        with st.spinner("Authenticating with Kite..."):
            try:
                data = kite_unauth.generate_session(req_tok, api_secret=CONF["kite"]["api_secret"])
                st.session_state["kite_access_token"] = data.get("access_token")
                st.session_state["kite_login_response"] = data
                st.success("Kite auth successful.")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Kite auth failed: {e}")
    if st.session_state["kite_access_token"]:
        st.success("Kite ✅")
        if st.button("Logout Kite", use_container_width=True):
            st.session_state["kite_access_token"] = None
            st.session_state["kite_login_response"] = None
            st.rerun()
    else:
        st.info("Not authenticated with Kite.")

    st.divider()
    if st.session_state["kite_access_token"]:
        if st.button("Fetch Current Holdings", use_container_width=True):
            kc = get_kite_auth_client(CONF["kite"]["api_key"], st.session_state["kite_access_token"])
            try:
                holds = kc.holdings()
                st.session_state["holdings_data"] = pd.DataFrame(holds)
                st.success(f"Fetched {len(holds)} holdings.")
            except Exception as e:
                st.error(f"Error fetching holdings: {e}")
        if st.session_state["holdings_data"] is not None and not st.session_state["holdings_data"].empty:
            with st.expander("Show Holdings"):
                st.dataframe(st.session_state["holdings_data"])
                st.download_button(
                    "Download Holdings (CSV)",
                    st.session_state["holdings_data"].to_csv(index=False).encode('utf-8'),
                    file_name="kite_holdings.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# --- Main Tabs (NO Market/Historical tab) ---
tabs = st.tabs([
    "💼 Investment Compliance",
    "🤖 AI-Powered Analysis",
])

tab_compliance, tab_ai = tabs

# --- Helpers for AI tab ---
def extract_text_from_files(uploaded_files) -> str:
    txt = ""
    for f in uploaded_files:
        txt += f"\n\n--- DOCUMENT: {f.name} ---\n\n"
        if f.type == "application/pdf":
            with fitz.open(stream=f.getvalue(), filetype="pdf") as doc:
                for page in doc:
                    txt += page.get_text()
        else:
            try:
                txt += f.getvalue().decode("utf-8", errors="ignore")
            except Exception:
                txt += ""
    return txt

def portfolio_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No portfolio data available."
    total_value = df['Real-time Value (Rs)'].sum()
    top10 = df.nlargest(10, 'Weight %')[['Name', 'Weight %']]
    sector_top = df.groupby('Industry')['Weight %'].sum().nlargest(10)
    s = [
        f"**Portfolio Snapshot (as of {datetime.now().strftime('%Y-%m-%d')})**",
        f"- **Total Value:** ₹ {total_value:,.2f}",
        f"- **Number of Holdings:** {len(df)}",
        f"- **Top Stock Weight:** {df['Weight %'].max():.2f}%",
        f"- **Top 10 Combined Weight:** {df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%",
        "",
        "**Top 10 Holdings:**"
    ]
    for _, r in top10.iterrows():
        s.append(f"- {r['Name']}: {r['Weight %']:.2f}%")
    s.append("")
    s.append("**Top 10 Sector Exposures:**")
    for sector, wt in sector_top.items():
        s.append(f"- {sector}: {wt:.2f}%")
    return "\n".join(s)

# --- COMPLIANCE TAB ---
with tab_compliance:
    st.header("💼 Enhanced Investment Compliance & Portfolio Analysis")
    st.write("Upload portfolio, run validations, compute risk analytics, and save the entire run to Supabase per-user.")

    if not st.session_state["kite_access_token"]:
        st.info("Login to Kite Connect for live prices.")
    kc = get_kite_auth_client(CONF["kite"]["api_key"], st.session_state["kite_access_token"])

    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("1) Upload Portfolio CSV")
        up = st.file_uploader("CSV with Symbol, Industry, Quantity, etc.", type="csv")

        st.markdown("##### Compliance Thresholds")
        with st.expander("⚙️ Configure", expanded=True):
            single_stock_limit = st.number_input("Single Stock Limit (%)", 1.0, 25.0, 10.0, 0.5)
            single_sector_limit = st.number_input("Single Sector Limit (%)", 5.0, 50.0, 25.0, 1.0)
            top_10_limit = st.number_input("Top 10 Holdings Limit (%)", 20.0, 80.0, 50.0, 5.0)
            min_holdings = st.number_input("Minimum Holdings Count", 10, 200, 30, 5)
            unrated_limit = st.number_input("Unrated Securities Limit (%)", 0.0, 30.0, 10.0, 1.0)

    with c2:
        st.subheader("2) Custom Rules")
        rules_text = st.text_area(
            "One rule per line",
            height=200,
            value=(
                "# Examples\n"
                "# STOCK RELIANCE < 10\n"
                "# SECTOR BANKING < 25\n"
                "# TOP_N_STOCKS 10 <= 50\n"
                "# RATING AAA >= 30\n"
                "# UNRATED_EXPOSURE <= 10\n"
            )
        )
        with st.expander("📖 Syntax Guide"):
            st.markdown(
                "- STOCK [Symbol] <op> [Value]%\n"
                "- SECTOR [Name] <op> [Value]%\n"
                "- TOP_N_STOCKS [N] <op> [Value]%\n"
                "- TOP_N_SECTORS [N] <op> [Value]%\n"
                "- COUNT_STOCKS <op> [Value]\n"
                "- COUNT_SECTORS <op> [Value]\n"
                "- RATING [Rating] <op> [Value]%\n"
                "- UNRATED_EXPOSURE <op> [Value]%\n"
                "- ASSET_CLASS [Class] <op> [Value]%\n"
                "- MARKET_CAP [Cap] <op> [Value]%\n"
                "- ISSUER_GROUP [Group] <op> [Value]%\n"
                "- MIN_LIQUIDITY [Symbol] >= [Volume]\n"
                "- FOREIGN_EXPOSURE <op> [Value]%\n"
                "- DERIVATIVES_EXPOSURE <op> [Value]%"
            )

    if up is not None and kc is not None:
        try:
            df = pd.read_csv(up)
            # normalize headers
            df.columns = [str(c).strip().lower().replace(' ', '_').replace('.', '').replace('/', '_') for c in df.columns]
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
                'market_fair_value(rs_in_lacs)': 'Uploaded Value (Lacs)',
            }
            df = df.rename(columns=header_map)
            for col in ['Rating', 'Asset Class', 'Industry', 'Market Cap', 'Issuer Group', 'Country', 'Instrument Type']:
                if col in df.columns:
                    df[col] = df[col].fillna('UNKNOWN').astype(str).str.strip().str.upper()

            if st.button("🔍 Analyze & Validate", type="primary"):
                with st.spinner("Fetching prices and computing analytics..."):
                    symbols = df['Symbol'].astype(str).str.upper().unique().tolist()
                    try:
                        ltp = kc.ltp([f"{DEFAULT_EXCHANGE}:{s}" for s in symbols])
                    except Exception as e:
                        st.error(f"LTP fetch failed: {e}")
                        ltp = {}

                    price_map = {s: ltp.get(f"{DEFAULT_EXCHANGE}:{s}", {}).get('last_price') for s in symbols}
                    res = df.copy()
                    res['Symbol'] = res['Symbol'].astype(str).str.upper()
                    res['LTP'] = res['Symbol'].map(price_map)
                    res['Quantity'] = pd.to_numeric(res.get('Quantity', 0), errors='coerce').fillna(0)
                    res['Real-time Value (Rs)'] = (res['LTP'] * res['Quantity']).fillna(0)
                    tot = float(res['Real-time Value (Rs)'].sum())
                    res['Weight %'] = (res['Real-time Value (Rs)'] / tot * 100) if tot > 0 else 0.0

                    st.session_state["cfg_thresholds"] = {
                        "single_stock_limit": float(single_stock_limit),
                        "single_sector_limit": float(single_sector_limit),
                        "top_10_limit": float(top_10_limit),
                        "min_holdings": int(min_holdings),
                        "unrated_limit": float(unrated_limit),
                    }

                    sec_comp = calculate_security_level_compliance(res, {
                        'single_stock_limit': single_stock_limit,
                        'single_sector_limit': single_sector_limit,
                        'min_liquidity': 100000
                    })

                    st.session_state["compliance_results_df"] = res
                    st.session_state["security_level_compliance"] = sec_comp
                    st.session_state["advanced_metrics"] = None

                    breaches = []
                    if (res['Weight %'] > single_stock_limit).any():
                        for _, r in res[res['Weight %'] > single_stock_limit].iterrows():
                            breaches.append({
                                "type": "Single Stock Limit",
                                "severity": "🔴 Critical",
                                "details": f"{r['Symbol']} at {r['Weight %']:.2f}% (Limit: {single_stock_limit}%)"
                            })
                    sec_w = res.groupby('Industry')['Weight %'].sum()
                    if (sec_w > single_sector_limit).any():
                        for sec, wt in sec_w[sec_w > single_sector_limit].items():
                            breaches.append({
                                "type": "Sector Limit",
                                "severity": "🟠 High",
                                "details": f"{sec} at {wt:.2f}% (Limit: {single_sector_limit}%)"
                            })
                    st.session_state["breach_alerts"] = breaches

                    st.success("✅ Analysis complete.")
                    if breaches:
                        st.warning(f"⚠️ {len(breaches)} breach(es) detected.")

        except Exception as e:
            st.error(f"CSV parse/analysis error: {e}")
            st.exception(e)

    results_df = st.session_state["compliance_results_df"]

    if not results_df.empty and 'Weight %' in results_df.columns:
        st.divider()
        if st.session_state["breach_alerts"]:
            st.error("🚨 Compliance Breaches")
            st.dataframe(pd.DataFrame(st.session_state["breach_alerts"]), hide_index=True, use_container_width=True)

        subtabs = st.tabs([
            "📊 Executive Dashboard",
            "🔍 Detailed Breakdowns",
            "📈 Advanced Risk Analytics",
            "⚖️ Rule Validation",
            "🔐 Security-Level Compliance",
            "📊 Concentration Analysis",
            "📄 Full Report",
            "💾 Save / Export",
        ])

        # Executive
        with subtabs[0]:
            st.subheader("Executive Dashboard")
            total_value = results_df['Real-time Value (Rs)'].sum()
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Portfolio Value", f"₹ {total_value:,.2f}")
            k2.metric("Holdings", f"{len(results_df)}")
            k3.metric("Sectors", f"{results_df['Industry'].nunique()}")
            if 'Rating' in results_df.columns:
                k4.metric("Ratings", f"{results_df['Rating'].nunique()}")
            stat = "✅ Pass" if not st.session_state["breach_alerts"] else f"❌ {len(st.session_state['breach_alerts'])} Breaches"
            k5.metric("Compliance", stat)

            st.markdown("#### Concentration")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Top Stock", f"{results_df['Weight %'].max():.2f}%")
            c1.metric("Top 5", f"{results_df.nlargest(5, 'Weight %')['Weight %'].sum():.2f}%")
            c2.metric("Top 10", f"{results_df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%")
            c2.metric("Top 3 Sectors", f"{results_df.groupby('Industry')['Weight %'].sum().nlargest(3).sum():.2f}%")
            stock_hhi = float((results_df['Weight %'] ** 2).sum())
            sector_hhi = float((results_df.groupby('Industry')['Weight %'].sum() ** 2).sum())
            def _hhi_b(x): return "🟢 Low" if x < 1500 else ("🟡 Moderate" if x <= 2500 else "🔴 High")
            c3.metric("Stock HHI", f"{stock_hhi:,.0f}", help=_hhi_b(stock_hhi))
            c3.metric("Sector HHI", f"{sector_hhi:,.0f}", help=_hhi_b(sector_hhi))
            effN = 1 / ((results_df['Weight %'] / 100) ** 2).sum()
            secW = results_df.groupby('Industry')['Weight %'].sum() / 100
            effN_sec = 1 / (secW ** 2).sum()
            c4.metric("Effective N (Stocks)", f"{effN:.1f}")
            c4.metric("Effective N (Sectors)", f"{effN_sec:.1f}")

            st.markdown("#### Composition")
            cc = st.columns(2)
            with cc[0]:
                top15 = results_df.nlargest(15, 'Weight %')
                others = results_df.nsmallest(max(len(results_df) - 15, 0), 'Weight %')['Weight %'].sum()
                pie_df = pd.concat([top15[['Name', 'Weight %']], pd.DataFrame([{'Name': 'Others', 'Weight %': others}])])
                st.plotly_chart(px.pie(pie_df, values='Weight %', names='Name', title='Top 15 + Others', hole=0.4), use_container_width=True)
            with cc[1]:
                sec_df = results_df.groupby('Industry')['Weight %'].sum().reset_index().sort_values('Weight %', ascending=False)
                fig = px.bar(sec_df.head(10), x='Weight %', y='Industry', orientation='h', title='Top Sectors', color='Weight %', color_continuous_scale='Blues')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        # Detailed
        with subtabs[1]:
            st.subheader("Detailed Breakdowns")
            btabs = st.tabs(["Holdings", "Sectors", "Ratings", "Market Cap", "Asset Class"])
            with btabs[0]:
                top20 = results_df.nlargest(20, 'Weight %')[['Name', 'Symbol', 'Industry', 'Weight %', 'Real-time Value (Rs)', 'LTP']]
                fig = px.bar(top20, x='Weight %', y='Name', orientation='h', color='Industry', title='Top 20 by Weight', hover_data=['Symbol', 'Real-time Value (Rs)'])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(top20.style.format({'Weight %': '{:.2f}%', 'Real-time Value (Rs)': '₹{:,.2f}', 'LTP': '₹{:,.2f}'}), use_container_width=True)

            with btabs[1]:
                sec_agg = results_df.groupby('Industry').agg({'Weight %': 'sum', 'Real-time Value (Rs)': 'sum', 'Symbol': 'count'}).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                c = st.columns(2)
                c[0].plotly_chart(px.bar(sec_agg.reset_index().head(15), x='Weight %', y='Industry', orientation='h', title='Top Sectors by Weight'), use_container_width=True)
                c[1].plotly_chart(px.bar(sec_agg.reset_index().head(15), x='Count', y='Industry', orientation='h', title='Sectors by Count', color='Count', color_continuous_scale='Greens'), use_container_width=True)
                st.dataframe(sec_agg.style.format({'Weight %': '{:.2f}%', 'Real-time Value (Rs)': '₹{:,.2f}'}), use_container_width=True)

            with btabs[2]:
                if 'Rating' in results_df.columns:
                    rat = results_df.groupby('Rating').agg({'Weight %': 'sum', 'Symbol': 'count'}).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                    c = st.columns(2)
                    c[0].plotly_chart(px.pie(rat.reset_index(), values='Weight %', names='Rating', title='Rating by Weight', hole=0.3), use_container_width=True)
                    c[1].plotly_chart(px.bar(rat.reset_index(), x='Weight %', y='Rating', orientation='h', title='Rating Exposure', color='Weight %', color_continuous_scale='RdYlGn_r'), use_container_width=True)
                    st.dataframe(rat.style.format({'Weight %': '{:.2f}%'}), use_container_width=True)
                else:
                    st.info("No Rating column")

            with btabs[3]:
                if 'Market Cap' in results_df.columns:
                    mc = results_df.groupby('Market Cap').agg({'Weight %': 'sum', 'Symbol': 'count'}).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                    c = st.columns(2)
                    c[0].plotly_chart(px.pie(mc.reset_index(), values='Weight %', names='Market Cap', title='Market Cap Mix', hole=0.3), use_container_width=True)
                    st.dataframe(mc.style.format({'Weight %': '{:.2f}%'}), use_container_width=True)
                else:
                    st.info("No Market Cap column")

            with btabs[4]:
                if 'Asset Class' in results_df.columns:
                    ac = results_df.groupby('Asset Class').agg({'Weight %': 'sum', 'Symbol': 'count'}).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                    c = st.columns(2)
                    c[0].plotly_chart(px.pie(ac.reset_index(), values='Weight %', names='Asset Class', title='Asset Class Mix', hole=0.3), use_container_width=True)
                    st.dataframe(ac.style.format({'Weight %': '{:.2f}%'}), use_container_width=True)
                else:
                    st.info("No Asset Class column")

            st.markdown("---")
            st.markdown("##### Treemap")
            tcols = st.columns([3, 1])
            with tcols[1]:
                depth = st.radio("Hierarchy", ["Industry → Stock", "Industry → Rating → Stock"])
            with tcols[0]:
                if depth == "Industry → Stock":
                    fig = px.treemap(results_df, path=[px.Constant("Portfolio"), 'Industry', 'Name'], values='Real-time Value (Rs)', hover_data={'Weight %': ':.2f'})
                else:
                    if 'Rating' in results_df.columns:
                        fig = px.treemap(results_df, path=[px.Constant("Portfolio"), 'Industry', 'Rating', 'Name'], values='Real-time Value (Rs)', hover_data={'Weight %': ':.2f'})
                    else:
                        st.warning("Rating not available; using Industry → Stock")
                        fig = px.treemap(results_df, path=[px.Constant("Portfolio"), 'Industry', 'Name'], values='Real-time Value (Rs)', hover_data={'Weight %': ':.2f'})
                fig.update_layout(margin=dict(t=40, l=25, r=25, b=25), height=600)
                st.plotly_chart(fig, use_container_width=True)

        # Risk
        with subtabs[2]:
            st.subheader("Advanced Risk Analytics")
            lc = st.columns([2, 1])
            with lc[1]:
                if st.button("🔄 Calculate Advanced Metrics", type="primary", use_container_width=True):
                    with st.spinner("Computing..."):
                        st.session_state["advanced_metrics"] = calculate_advanced_metrics(
                            results_df, CONF["kite"]["api_key"], st.session_state["kite_access_token"]
                        )
            with lc[0]:
                st.info("Uses 1-year daily data per holding and NIFTY 50 as benchmark.")
            m = st.session_state["advanced_metrics"]
            if m:
                rc = st.columns(4)
                rc[0].metric("VaR (95%) Daily", f"{m['var_95'] * 100:.2f}%")
                rc[1].metric("VaR (99%) Daily", f"{m['var_99'] * 100:.2f}%")
                rc[2].metric("CVaR (95%)", f"{m['cvar_95'] * 100:.2f}%")
                rc[3].metric("Volatility (Ann.)", f"{m['portfolio_volatility'] * 100:.2f}%" if m['portfolio_volatility'] is not None else "N/A")

                pc = st.columns(4)
                pc[0].metric(f"Beta vs {BENCHMARK_SYMBOL}", f"{m['beta']:.3f}" if m['beta'] is not None else "N/A")
                pc[1].metric("Alpha (Ann.)", f"{m['alpha'] * 100:.2f}%" if m['alpha'] is not None else "N/A")
                pc[2].metric("Tracking Error", f"{m['tracking_error'] * 100:.2f}%" if m['tracking_error'] is not None else "N/A")
                pc[3].metric("Information Ratio", f"{m['information_ratio']:.3f}" if m['information_ratio'] is not None else "N/A")

                dc = st.columns(3)
                dc[0].metric("Sortino", f"{m['sortino_ratio']:.3f}" if m['sortino_ratio'] is not None else "N/A")
                dc[1].metric("Avg Correlation", f"{m['avg_correlation']:.3f}" if m['avg_correlation'] is not None else "N/A")
                dc[2].metric("Diversification Ratio", f"{m['diversification_ratio']:.3f}" if m['diversification_ratio'] is not None else "N/A")

        # Rule validation
        with subtabs[3]:
            st.subheader("Rule Validation")
            st.write("Results based on your custom rules.")
            vals = parse_and_validate_rules_enhanced(rules_text, results_df)
            st.session_state["last_validation_results"] = vals

            if not vals:
                st.info("Add rules to see results.")
            else:
                total = len(vals)
                passed = sum(1 for r in vals if r['status'] == "✅ PASS")
                failed = sum(1 for r in vals if r['status'] == "❌ FAIL")
                errors = sum(1 for r in vals if 'Error' in r['status'] or 'Invalid' in r['status'])
                mc = st.columns(4)
                mc[0].metric("Total Rules", total)
                mc[1].metric("✅ Passed", passed)
                mc[2].metric("❌ Failed", failed)
                mc[3].metric("⚠️ Errors", errors)

                fc = st.columns([2, 1])
                with fc[0]:
                    sf = st.multiselect("Status filter", ["✅ PASS", "❌ FAIL", "⚠️ Invalid", "Error"], default=["✅ PASS", "❌ FAIL", "⚠️ Invalid", "Error"])
                with fc[1]:
                    svf = st.multiselect("Severity", ["🔴 Critical", "🟠 High", "🟡 Medium", "✅ Compliant"], default=["🔴 Critical", "🟠 High", "🟡 Medium", "✅ Compliant"])

                st.markdown("---")
                for r in vals:
                    if r['status'] not in sf:
                        continue
                    if r.get('severity') not in svf and r.get('severity') != 'N/A':
                        continue
                    sev = r.get('severity', 'N/A')
                    if r['status'] == "✅ PASS":
                        with st.expander(f"{r['status']} {sev} | `{r['rule']}`", expanded=False):
                            st.success(r['details'])
                    elif r['status'] == "❌ FAIL":
                        with st.expander(f"{r['status']} {sev} | `{r['rule']}`", expanded=True):
                            st.error(r['details'])
                            if 'actual_value' in r:
                                st.metric("Actual", f"{r['actual_value']:.2f}%")
                                st.metric("Threshold", f"{r['threshold']:.2f}%")
                                st.metric("Breach", f"{r['breach_amount']:.2f}%", delta=f"{r['breach_amount']:.2f}%", delta_color="inverse")
                    else:
                        with st.expander(f"{r['status']} | `{r['rule']}`"):
                            st.warning(r['details'])

                st.markdown("---")
                if st.button("📥 Download Validation CSV", use_container_width=True):
                    csv = pd.DataFrame(vals).to_csv(index=False).encode('utf-8')
                    st.download_button("Save CSV", csv, file_name=f"rule_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)

        # Security-level
        with subtabs[4]:
            st.subheader("Security-Level Compliance")
            sdf = st.session_state["security_level_compliance"]
            if sdf.empty:
                st.info("No data — run analysis first.")
            else:
                b = (sdf['Stock Limit Breach'] == '❌ Breach').sum()
                g = (sdf['Stock Limit Breach'] == '✅ Compliant').sum()
                mc = st.columns(4)
                mc[0].metric("Total", len(sdf))
                mc[1].metric("✅ Compliant", g)
                mc[2].metric("❌ Breach", b)
                mc[3].metric("Breach Rate", f"{(b/len(sdf)*100):.1f}%")

                st.markdown("---")
                fc = st.columns(3)
                with fc[0]:
                    cf = st.multiselect("Stock Limit Status", sdf['Stock Limit Breach'].unique(), default=list(sdf['Stock Limit Breach'].unique()))
                with fc[1]:
                    rf = st.multiselect("Concentration Risk", sdf['Concentration Risk'].unique(), default=list(sdf['Concentration Risk'].unique()))
                with fc[2]:
                    rcf = st.multiselect("Rating Compliance" if 'Rating Compliance' in sdf.columns else "—", list(sdf.get('Rating Compliance', pd.Series(dtype=str)).unique()) if 'Rating Compliance' in sdf.columns else [], default=list(sdf.get('Rating Compliance', pd.Series(dtype=str)).unique()) if 'Rating Compliance' in sdf.columns else [])

                f = sdf[sdf['Stock Limit Breach'].isin(cf)]
                if 'Concentration Risk' in f.columns:
                    f = f[f['Concentration Risk'].isin(rf)]
                if 'Rating Compliance' in f.columns and len(rcf) > 0:
                    f = f[f['Rating Compliance'].isin(rcf)]

                show_cols = ['Name', 'Symbol', 'Industry', 'Weight %', 'Stock Limit Breach', 'Stock Limit Gap (%)', 'Concentration Risk']
                for extra in ['Liquidity Status', 'Rating Compliance', 'Rating', 'Real-time Value (Rs)']:
                    if extra in f.columns and extra not in show_cols:
                        show_cols.append(extra)

                def _hl(row):
                    if row['Stock Limit Breach'] == '❌ Breach':
                        return ['background-color: #ffcccc'] * len(row)
                    return [''] * len(row)
                st.dataframe(f[show_cols].style.apply(_hl, axis=1).format({'Weight %': '{:.2f}%', 'Stock Limit Gap (%)': '{:.2f}%', 'Real-time Value (Rs)': '₹{:,.2f}'}), use_container_width=True, height=520)

                if b > 0:
                    st.markdown("---")
                    st.markdown("#### Breach Details")
                    limit_val = st.session_state["cfg_thresholds"].get("single_stock_limit", 10.0)
                    for _, row in sdf[sdf['Stock Limit Breach'] == '❌ Breach'].sort_values('Weight %', ascending=False).iterrows():
                        with st.expander(f"🔴 {row['Symbol']} - {row['Name']} ({row['Weight %']:.2f}%)"):
                            cols = st.columns(3)
                            cols[0].metric("Current", f"{row['Weight %']:.2f}%")
                            cols[1].metric("Limit", f"{limit_val:.2f}%")
                            cols[2].metric("Excess", f"{row['Weight %'] - float(limit_val):.2f}%", delta=f"{row['Weight %'] - float(limit_val):.2f}%", delta_color="inverse")
                            st.write(f"Industry: {row['Industry']}")
                            st.write(f"Value: ₹{row['Real-time Value (Rs)']:,.2f}" if 'Real-time Value (Rs)' in row else "")

                st.markdown("---")
                st.download_button(
                    "📥 Export Security Compliance CSV",
                    sdf.to_csv(index=False).encode('utf-8'),
                    file_name=f"security_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # Concentration
        with subtabs[5]:
            st.subheader("Concentration Analysis")
            srt = results_df.sort_values('Weight %', ascending=False).reset_index(drop=True)
            srt['Cumulative Weight %'] = srt['Weight %'].cumsum()
            srt['Rank'] = range(1, len(srt) + 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=srt['Rank'], y=srt['Cumulative Weight %'], mode='lines+markers', name='Portfolio', line=dict(width=2)))
            fig.add_trace(go.Scatter(x=[0, len(srt)], y=[0, 100], mode='lines', name='Equality', line=dict(dash='dash')))
            fig.update_layout(title="Lorenz Curve", xaxis_title="Holdings (ranked)", yaxis_title="Cumulative Weight %", height=500)
            st.plotly_chart(fig, use_container_width=True)

            bc = st.columns(5)
            top_1 = float(srt.iloc[0]['Weight %'])
            top_3 = float(srt.head(3)['Weight %'].sum())
            top_5 = float(srt.head(5)['Weight %'].sum())
            top_10 = float(srt.head(10)['Weight %'].sum())
            top_20 = float(srt.head(20)['Weight %'].sum() if len(srt) >= 20 else srt['Weight %'].sum())
            bc[0].metric("Top 1", f"{top_1:.2f}%"); bc[1].metric("Top 3", f"{top_3:.2f}%")
            bc[2].metric("Top 5", f"{top_5:.2f}%"); bc[3].metric("Top 10", f"{top_10:.2f}%")
            bc[4].metric("Top 20", f"{top_20:.2f}%")

            st.markdown("---")
            st.markdown("#### Sector vs Market Cap Heatmap")
            if 'Market Cap' in results_df.columns:
                pivot = results_df.pivot_table(values='Weight %', index='Industry', columns='Market Cap', aggfunc='sum', fill_value=0)
                h = px.imshow(pivot, labels=dict(x="Market Cap", y="Industry", color="Weight %"), title="Allocation Heatmap", color_continuous_scale='RdYlGn_r')
                h.update_layout(height=max(400, len(pivot) * 30))
                st.plotly_chart(h, use_container_width=True)
            else:
                st.info("No Market Cap column for heatmap.")
            st.session_state["last_concentration"] = {
                "sorted": dictify_df(srt),
                "benchmarks": {"top_1": top_1, "top_3": top_3, "top_5": top_5, "top_10": top_10, "top_20": top_20},
                "heatmap": pivot.to_dict() if 'Market Cap' in results_df.columns else {}
            }

        # Full report
        with subtabs[6]:
            st.subheader("Full Report (Download)")
            opt = st.multiselect(
                "Sections",
                ["Executive Summary", "Holdings Detail", "Sector Analysis", "Risk Metrics", "Compliance Validation", "Security-Level Compliance"],
                default=["Executive Summary", "Holdings Detail", "Sector Analysis", "Compliance Validation"]
            )
            fmt = st.radio("Format", ["Excel", "CSV"], horizontal=True)
            if st.button("📊 Generate Report", type="primary"):
                with st.spinner("Building report..."):
                    data_map = {}
                    total_value = results_df['Real-time Value (Rs)'].sum()
                    if "Executive Summary" in opt:
                        data_map['Executive Summary'] = pd.DataFrame({
                            "Metric": ["Total Value", "Holdings", "Sectors", "Top Stock", "Top 10", "Stock HHI", "Sector HHI"],
                            "Value": [
                                f"₹{total_value:,.2f}", len(results_df), results_df['Industry'].nunique(),
                                f"{results_df['Weight %'].max():.2f}%",
                                f"{results_df.nlargest(10, 'Weight %')['Weight %'].sum():.2f}%",
                                f"{(results_df['Weight %'] ** 2).sum():.0f}",
                                f"{(results_df.groupby('Industry')['Weight %'].sum() ** 2).sum():.0f}"
                            ]
                        })
                    if "Holdings Detail" in opt:
                        data_map["Holdings Detail"] = results_df[['Name', 'Symbol', 'Industry', 'Quantity', 'LTP', 'Real-time Value (Rs)', 'Weight %']]
                    if "Sector Analysis" in opt:
                        sa = results_df.groupby('Industry').agg({'Weight %': 'sum', 'Real-time Value (Rs)': 'sum', 'Symbol': 'count'}).rename(columns={'Symbol': 'Count'}).sort_values('Weight %', ascending=False)
                        data_map["Sector Analysis"] = sa
                    if "Risk Metrics" in opt and st.session_state["advanced_metrics"]:
                        m = st.session_state["advanced_metrics"]
                        data_map["Risk Metrics"] = pd.DataFrame([{"Metric": k.replace('_', ' ').title(), "Value": ("N/A" if v is None else f"{v:.6f}")} for k, v in m.items()])
                    if "Compliance Validation" in opt and st.session_state["last_validation_results"]:
                        data_map["Compliance Validation"] = pd.DataFrame(st.session_state["last_validation_results"])
                    if "Security-Level Compliance" in opt and not st.session_state["security_level_compliance"].empty:
                        data_map["Security Compliance"] = st.session_state["security_level_compliance"]

                    if fmt == "Excel":
                        from io import BytesIO
                        out = BytesIO()
                        with pd.ExcelWriter(out, engine="openpyxl") as w:
                            for name, df_s in data_map.items():
                                df_s.to_excel(w, sheet_name=name[:31], index=False)
                        out.seek(0)
                        st.download_button("📥 Download Excel", out, file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                    else:
                        comb = pd.concat([df.assign(Section=name) for name, df in data_map.items()], ignore_index=True)
                        st.download_button("📥 Download CSV", comb.to_csv(index=False).encode('utf-8'), file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
                    st.success("✅ Report ready.")

        # Save / Export
        with subtabs[7]:
            st.subheader("💾 Save Everything (Supabase per-user)")
            run_label = st.text_input("Run label (optional)", value=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            payload = {
                "created_at": datetime.utcnow().isoformat(),
                "user_email": st.session_state["sb_user"].email if st.session_state["sb_user"] else None,
                "thresholds": st.session_state["cfg_thresholds"],
                "portfolio": dictify_df(results_df),
                "breaches": st.session_state["breach_alerts"],
                "security_level_compliance": dictify_df(st.session_state["security_level_compliance"]),
                "rule_validations": st.session_state["last_validation_results"],
                "advanced_metrics": st.session_state["advanced_metrics"],
                "concentration": st.session_state["last_concentration"],
                "ai_analysis": st.session_state["ai_analysis_response"],
                "meta": {
                    "kite_authenticated": bool(st.session_state["kite_access_token"]),
                    "benchmark": BENCHMARK_SYMBOL
                }
            }
            c = st.columns(2)
            with c[0]:
                if st.button("🟩 Save to Supabase (analysis_runs)", type="primary", use_container_width=True):
                    _ = save_full_payload_to_supabase(run_label.strip() or None, payload)
            with c[1]:
                b = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
                st.download_button("📥 Download Full JSON", b, file_name=f"analysis_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", use_container_width=True)

# --- AI-Powered Analysis (Gemini) ---
with tab_ai:
    st.header("🤖 AI-Powered Compliance Analysis (Gemini)")
    dfp = st.session_state["compliance_results_df"]
    if dfp is None or dfp.empty:
        st.warning("Upload & analyze a portfolio first in the Investment Compliance tab.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            files = st.file_uploader("Upload SID / KIM / Policy (PDF or TXT — multiple allowed)", type=["pdf", "txt"], accept_multiple_files=True)
        with c2:
            depth = st.select_slider("Analysis Depth", options=["Quick", "Standard", "Comprehensive"], value="Standard")
            incl_reco = st.checkbox("Include Recommendations", value=True)
            incl_risk = st.checkbox("Include Risk Assessment", value=True)

        if files:
            st.success(f"Uploaded {len(files)} document(s).")
            if st.button("🚀 Run AI Analysis", type="primary"):
                with st.spinner("Reading documents & generating analysis..."):
                    try:
                        docs_text = extract_text_from_files(files)
                        psum = portfolio_summary(dfp)
                        breaches = st.session_state["breach_alerts"]
                        breach_summary = "\n".join([f"- {b['type']}: {b['details']}" for b in breaches]) if breaches else "No immediate breaches detected."

                        if depth == "Quick":
                            depth_msg = "Provide a concise analysis focusing on critical compliance issues only."
                        elif depth == "Standard":
                            depth_msg = "Provide a balanced analysis covering key compliance areas and major risks."
                        else:
                            depth_msg = "Provide an exhaustive, detailed analysis covering all aspects of compliance, risk, and regulatory requirements."

prompt = f"""
You are an expert investment compliance analyst for an Indian AMC with deep knowledge of SEBI regulations, MF guidelines, and portfolio best practices.

YOUR TASK:
Perform a comprehensive compliance analysis of the given investment portfolio against the provided scheme documents (SID/KIM/policy) and SEBI/AMFI regulations.

{depth_msg}

PORTFOLIO DATA:
