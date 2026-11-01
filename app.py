import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Air Quality Forecast", page_icon="🌐", layout="wide")

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
TARGET = FEATURES.index("AQI")
WINDOW = 30
MIN_R2 = 0.5          # isse kam R2 wali city hide hogi
DAY_START = 6         # Auto mode: subah 6 se
NIGHT_START = 19      # Auto mode: shaam 7 se night

COORDS = {
    "Ahmedabad": (23.02, 72.57), "Aizawl": (23.73, 92.72), "Amaravati": (16.51, 80.52),
    "Amritsar": (31.63, 74.87), "Bengaluru": (12.97, 77.59), "Bhopal": (23.26, 77.41),
    "Brajrajnagar": (21.82, 83.92), "Chandigarh": (30.73, 76.78), "Chennai": (13.08, 80.27),
    "Coimbatore": (11.02, 76.96), "Delhi": (28.61, 77.21), "Ernakulam": (9.98, 76.28),
    "Gurugram": (28.46, 77.03), "Guwahati": (26.14, 91.74), "Hyderabad": (17.39, 78.49),
    "Jaipur": (26.91, 75.79), "Jorapokhar": (23.70, 86.41), "Kochi": (9.93, 76.27),
    "Kolkata": (22.57, 88.36), "Lucknow": (26.85, 80.95), "Mumbai": (19.08, 72.88),
    "Patna": (25.59, 85.14), "Shillong": (25.58, 91.89), "Talcher": (20.95, 85.22),
    "Thiruvananthapuram": (8.52, 76.94), "Visakhapatnam": (17.69, 83.22),
}

STATE_MAP = {
    "Ahmedabad": "Gujarat", "Aizawl": "Mizoram", "Amaravati": "Andhra Pradesh",
    "Amritsar": "Punjab", "Bengaluru": "Karnataka", "Bhopal": "Madhya Pradesh",
    "Brajrajnagar": "Odisha", "Chandigarh": "Chandigarh", "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu", "Delhi": "Delhi", "Ernakulam": "Kerala",
    "Gurugram": "Haryana", "Guwahati": "Assam", "Hyderabad": "Telangana",
    "Jaipur": "Rajasthan", "Jorapokhar": "Jharkhand", "Kochi": "Kerala",
    "Kolkata": "West Bengal", "Lucknow": "Uttar Pradesh", "Mumbai": "Maharashtra",
    "Patna": "Bihar", "Shillong": "Meghalaya", "Talcher": "Odisha",
    "Thiruvananthapuram": "Kerala", "Visakhapatnam": "Andhra Pradesh",
}

INDIA_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
]

# (upper limit, label, colour, text colour, advice)
BANDS = [
    (50, "Good", "#00B050", "#FFFFFF", "Air quality is good. Outdoor activity is fine."),
    (100, "Satisfactory", "#92D050", "#1A1A1A", "Acceptable for most people. Sensitive people should be careful."),
    (200, "Moderate", "#FFE600", "#1A1A1A", "People with asthma or heart disease should limit time outdoors."),
    (300, "Poor", "#FFA500", "#1A1A1A", "Avoid long outdoor activity. Wear a mask."),
    (400, "Very Poor", "#E53935", "#FFFFFF", "Avoid going out. An N95 mask is necessary."),
    (10000, "Severe", "#A00000", "#FFFFFF", "Health emergency level. Stay indoors."),
]
RANGES = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500)]

# ---------------- Day / Night themes ----------------
THEMES = {
    "Day": dict(bg="#F0F2F5", card="#FFFFFF", text="#050505", muted="#65676B", border="#CED0D4",
                input="#FFFFFF", hover="#E7F3FF", advice="#E7F3FF", metric="#F0F2F5", sidebar="#FFFFFF",
                shadow="rgba(0,0,0,0.12)", navbar="#1877F2", accent="#1877F2", template="plotly_white",
                land="#F0F2F5", ocean="#DCEBFF", country="#999999", mline="#333333", gauge_bar="#1A1A1A"),
    "Night": dict(bg="#18191A", card="#242526", text="#E4E6EB", muted="#B0B3B8", border="#3E4042",
                  input="#3A3B3C", hover="#3A3B3C", advice="#263951", metric="#303132", sidebar="#242526",
                  shadow="rgba(0,0,0,0.5)", navbar="#1B3A6B", accent="#4599FF", template="plotly_dark",
                  land="#2F3031", ocean="#17263A", country="#6A6C70", mline="#E4E6EB", gauge_bar="#E4E6EB"),
}


def now_ist():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        return datetime.now()


def auto_theme():
    return "Day" if DAY_START <= now_ist().hour < NIGHT_START else "Night"


mode = st.session_state.get("theme_mode", "Auto")
theme_name = auto_theme() if mode == "Auto" else mode
T = THEMES[theme_name]
st.session_state["applied_theme"] = theme_name
ACC = T["accent"]

PLOT = dict(template=T["template"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=T["text"]))


def band(v):
    for hi, label, color, txt, advice in BANDS:
        if v <= hi:
            return label, color, txt, advice
    return BANDS[-1][1:]


def pill(text, bg, fg):
    return (f"<span style='background:{bg};color:{fg} !important;padding:3px 12px;border-radius:20px;"
            f"font-size:0.78rem;font-weight:600'>{text}</span>")


def aqi_pill(v):
    l, c, t, _ = band(v)
    return pill(l, c, t)


def stat_card(label, value, pill_html=""):
    return (f"<div class='stat'><div class='stat-l'>{label}</div>"
            f"<div class='stat-v'>{value}</div><div>{pill_html}</div></div>")


def html_table(df):
    return "<div class='tbl-wrap'>" + df.to_html(classes="aq", border=0, escape=True) + "</div>"


CSS = """
<style>
html, body, [class*="css"] {font-family: 'Segoe UI', Helvetica, Arial, sans-serif;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background: __BG__ !important;}
[data-testid="stHeader"] {background: transparent !important;}
[data-testid="stHeader"] *, [data-testid="stToolbar"] * {color: __TEXT__ !important;}
.block-container {padding-top: 0.6rem; max-width: 1250px;}

.stApp p, .stApp span, .stApp label, .stApp li, .stApp div[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] p {color: __TEXT__;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {color: __MUTED__ !important;}
[data-testid="stSlider"] div {color: __TEXT__;}

[data-testid="stSidebar"] {background: __SIDEBAR__ !important; border-right: 1px solid __BORDER__;}
[data-testid="stSidebar"] * {color: __TEXT__;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color: __MUTED__ !important;}
[data-testid="stExpander"] {background: __CARD__; border: 1px solid __BORDER__ !important; border-radius: 10px;}

/* dropdowns */
[data-baseweb="select"], [data-baseweb="select"] * {background-color: __INPUT__ !important;}
[data-baseweb="select"] > div {border: 1px solid __BORDER__ !important; border-radius: 8px;}
[data-baseweb="select"] [data-baseweb="tag"] {background-color: __HOVER__ !important; border-radius: 6px;}
[data-baseweb="select"] [data-baseweb="tag"] * {background-color: transparent !important;}
[data-baseweb="select"] div, [data-baseweb="select"] span, [data-baseweb="select"] input {
    color: __TEXT__ !important; -webkit-text-fill-color: __TEXT__ !important;}
[data-baseweb="select"] svg {fill: __TEXT__ !important;}
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"],
ul[role="listbox"] {background-color: __CARD__ !important;}
[data-baseweb="popover"] li, [role="option"] {background-color: __CARD__ !important;}
[data-baseweb="popover"] li *, [role="option"] * {background: transparent !important;
    color: __TEXT__ !important; -webkit-text-fill-color: __TEXT__ !important;}
[data-baseweb="popover"] li:hover, [role="option"]:hover,
[role="option"][aria-selected="true"] {background-color: __HOVER__ !important;}

/* navbar */
.navbar {background: __NAVBAR__; padding: 14px 24px; border-radius: 12px; margin-bottom: 16px;
         display: flex; align-items: center; gap: 16px; box-shadow: 0 2px 8px __SHADOW__;}
.logo {width: 46px; height: 46px; border-radius: 50%; background: #FFFFFF; color: __NAVBAR__ !important;
       font-weight: 800; font-size: 1.15rem; display: flex; align-items: center; justify-content: center;}
.navbar h1 {margin: 0; font-size: 1.55rem; color: #FFFFFF !important; font-weight: 700; border: none; padding: 0;}
.navbar p {margin: 2px 0 0 0; font-size: 0.88rem; color: #E7F0FF !important;}
.mode-badge {margin-left: auto; background: rgba(255,255,255,0.18); color: #FFFFFF !important;
             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; white-space: nowrap;}

/* stat cards */
.stat {background: __CARD__; border-radius: 12px; padding: 16px 18px; height: 100%;
       box-shadow: 0 1px 3px __SHADOW__;}
.stat-l {color: __MUTED__ !important; font-size: 0.85rem; font-weight: 600;}
.stat-v {color: __TEXT__ !important; font-size: 2.2rem; font-weight: 700; line-height: 1.25;}

/* tabs */
button[data-baseweb="tab"] {background: transparent !important; font-weight: 600; font-size: 0.98rem;}
button[data-baseweb="tab"] p {color: __MUTED__ !important;}
button[data-baseweb="tab"][aria-selected="true"] p {color: __ACCENT__ !important;}
[data-baseweb="tab-highlight"] {background-color: __ACCENT__ !important; height: 3px;}
[data-baseweb="tab-border"] {background-color: __BORDER__ !important;}

/* cards */
[class*="st-key-card"] {background: __CARD__; border-radius: 12px; border: none !important;
                        box-shadow: 0 1px 3px __SHADOW__; padding: 8px 10px;}
h2, h3, h4 {color: __TEXT__ !important; font-weight: 700 !important;}

.day-card {padding: 14px 4px; text-align: center; border-radius: 14px;
           box-shadow: 0 2px 6px rgba(0,0,0,0.25); transition: transform .15s;}
.day-card:hover {transform: translateY(-3px);}
.day-card .d {font-weight: 700; font-size: 1rem;}
.day-card .dt {font-size: 0.76rem; opacity: 0.9;}
.day-card .v {font-size: 2.1rem; font-weight: 800; line-height: 1.2;}
.day-card .l {font-size: 0.8rem; font-weight: 700;}
.day-card div {color: inherit !important;}

.chip {display: inline-block; padding: 8px 14px; border-radius: 12px; margin: 4px 8px 4px 0;
       font-weight: 600; box-shadow: 0 1px 3px rgba(0,0,0,0.25);}
.chip b {font-size: 1.1rem; margin-left: 10px;}

.advice {padding: 12px 16px; margin-top: 14px; border-radius: 10px; background: __ADVICE__;
         border-left: 6px solid __ACCENT__; color: __TEXT__ !important; font-size: 0.95rem;}
.footer {margin-top: 26px; padding: 12px; text-align: center; font-size: 0.8rem; color: __MUTED__ !important;}
[data-testid="stMetric"] {background: __METRIC__; border-radius: 10px; padding: 10px 14px;}
[data-testid="stMetricLabel"] p {color: __MUTED__ !important;}
[data-testid="stMetricValue"] {color: __TEXT__ !important;}
.stDownloadButton button {background: __ACCENT__; color: #FFFFFF !important; border: none; border-radius: 8px;}

/* custom table */
.tbl-wrap {overflow-x: auto; border: 1px solid __BORDER__; border-radius: 10px;}
table.aq {width: 100%; border-collapse: collapse; font-size: 0.88rem;}
table.aq th {background: __METRIC__; color: __TEXT__; text-align: left; padding: 8px 10px; white-space: nowrap;}
table.aq td {padding: 7px 10px; border-top: 1px solid __BORDER__; color: __TEXT__; white-space: nowrap;}
table.aq tr:hover td {background: __HOVER__;}
</style>
"""
for _k, _v in {"__BG__": T["bg"], "__CARD__": T["card"], "__TEXT__": T["text"], "__MUTED__": T["muted"],
               "__BORDER__": T["border"], "__INPUT__": T["input"], "__HOVER__": T["hover"],
               "__ADVICE__": T["advice"], "__METRIC__": T["metric"], "__SIDEBAR__": T["sidebar"],
               "__SHADOW__": T["shadow"], "__NAVBAR__": T["navbar"], "__ACCENT__": T["accent"]}.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)

# Auto mode: har 5 minute mein time check, theme badalna ho toh page rerun
if hasattr(st, "fragment"):
    @st.fragment(run_every="5m")
    def _theme_watcher():
        if (st.session_state.get("theme_mode", "Auto") == "Auto"
                and auto_theme() != st.session_state.get("applied_theme")):
            st.rerun()
    try:
        _theme_watcher()
    except Exception:
        pass


# ---------------- Data / model ----------------
@st.cache_data
def load_summary():
    with open("models/summary.json") as f:
        return json.load(f)


@st.cache_resource
def get_model(city):
    return tf.keras.models.load_model(f"models/{city}.keras"), joblib.load(f"models/{city}_scaler.pkl")


@st.cache_data
def get_recent(city):
    return pd.read_csv(f"models/{city}_recent.csv", index_col=0, parse_dates=True)


@st.cache_data
def get_backtest(city):
    return pd.read_csv(f"models/{city}_backtest.csv", parse_dates=["Date"])


def inv(sc, v):
    dummy = np.zeros((len(v), len(FEATURES)))
    dummy[:, TARGET] = v
    return sc.inverse_transform(dummy)[:, TARGET]


@st.cache_data(show_spinner=False)
def run_forecast(city, steps=7):
    model, sc = get_model(city)
    recent = get_recent(city)
    window = sc.transform(recent[FEATURES].iloc[-WINDOW:])
    preds = []
    for _ in range(steps):
        x = window[np.newaxis].astype("float32")
        p = float(model(x, training=False).numpy()[0, 0])   # model.predict se tez
        preds.append(p)
        row = window[-1].copy()
        row[TARGET] = p
        window = np.vstack([window[1:], row])
    vals = np.clip(inv(sc, np.array(preds)), 0, 500)
    dates = pd.date_range(recent.index[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.Series(vals, index=dates)


SUMMARY = load_summary()


def state_of(c):
    return SUMMARY.get(c, {}).get("state") or STATE_MAP.get(c, "Other")


ALL_CITIES = sorted(SUMMARY, key=lambda c: SUMMARY[c]["avg_aqi"], reverse=True)
CITIES = [c for c in ALL_CITIES if SUMMARY[c]["r2"] >= MIN_R2]
EXCLUDED = [c for c in ALL_CITIES if c not in CITIES]
STATES = sorted({state_of(c) for c in CITIES})

# ---------------- Sidebar ----------------
st.sidebar.markdown("### Settings")
st.sidebar.radio("Theme", ["Auto", "Day", "Night"], horizontal=True, key="theme_mode",
                 help="Auto: subah 6 se shaam 7 tak Day, baaki Night (IST)")
pick_mode = st.sidebar.radio("Select by", ["State → City", "All cities"], horizontal=True)
if pick_mode == "State → City":
    state = st.sidebar.selectbox("State / UT", STATES, index=STATES.index(state_of(CITIES[0])))
    city_opts = [c for c in CITIES if state_of(c) == state]          # most polluted pehle
    city = st.sidebar.selectbox("City (most polluted first)", city_opts)
else:
    city = st.sidebar.selectbox("City (most polluted first)", CITIES,
                                format_func=lambda c: f"{c} ({state_of(c)})")
    state = state_of(city)
steps = st.sidebar.slider("Forecast days", 1, 7, 7)

recent = get_recent(city)
hist_opts = [d for d in [30, 60, 90, 180, 365, 730] if d <= len(recent)] or [len(recent)]
hist_days = st.sidebar.select_slider("History shown (days)", options=hist_opts,
                                     value=60 if 60 in hist_opts else hist_opts[-1])
st.sidebar.markdown("---")
st.sidebar.caption("Model: LSTM (32 units), 30-day window  \nData: CPCB daily AQI, 2015-2020  \n"
                   "Forecast is for the days after the last record in the data.")
with st.sidebar.expander("Coverage"):
    st.write(f"{len(STATES)} states/UTs and {len(CITIES)} cities have forecasts.")
    missing = [s for s in INDIA_STATES if s not in STATES]
    if missing:
        st.caption("No data in this dataset for: " + ", ".join(missing))
    if EXCLUDED:
        st.caption("Hidden (low model accuracy): " + ", ".join(EXCLUDED))

# ---------------- Navbar ----------------
badge = ("☀ Day mode" if theme_name == "Day" else "☾ Night mode") + (" (auto)" if mode == "Auto" else "")
st.markdown(f"""
<div class='navbar'>
  <div class='logo'>AQ</div>
  <div>
    <h1>Air Quality Forecast</h1>
    <p>LSTM deep learning forecast for the most polluted cities</p>
  </div>
  <div class='mode-badge'>{badge}</div>
</div>
""", unsafe_allow_html=True)

fc = run_forecast(city, 7).iloc[:steps]
cur = float(recent["AQI"].iloc[-1])
s = SUMMARY[city]
trend = fc.iloc[0] - cur

m1, m2, m3, m4 = st.columns(4)
m1.markdown(stat_card(f"{city}, {state}: last recorded AQI", f"{cur:.0f}", aqi_pill(cur)), unsafe_allow_html=True)
m2.markdown(stat_card("Forecast peak", f"{fc.max():.0f}", aqi_pill(fc.max())), unsafe_allow_html=True)
m3.markdown(stat_card("Forecast average", f"{fc.mean():.0f}", aqi_pill(fc.mean())), unsafe_allow_html=True)
m4.markdown(stat_card("Model R² (test)", f"{s['r2']:.2f}", pill("LSTM", T["hover"], ACC)),
            unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "City / State comparison", "Pollutants", "Model accuracy"])

# ---------------- Tab 1: Forecast ----------------
with tab1:
    with st.container(key="card_forecast"):
        st.subheader(f"{city} ({state}): next {steps} days")
        cols = st.columns(steps)
        for i, (d, v) in enumerate(fc.items()):
            label, color, txt, _ = band(v)
            cols[i].markdown(
                f"<div class='day-card' style='background:{color};color:{txt}'>"
                f"<div class='d'>{d:%a}</div><div class='dt'>{d:%d %b %Y}</div>"
                f"<div class='v'>{v:.0f}</div><div class='l'>{label}</div></div>",
                unsafe_allow_html=True)

        label0, _, _, advice0 = band(fc.iloc[0])
        arrow = "rising" if trend > 5 else "falling" if trend < -5 else "about the same"
        st.markdown(f"<div class='advice'><b>Health advice ({label0}):</b> {advice0}<br>"
                    f"<span style='color:{T['muted']}'>Next day is {arrow} compared to the last record "
                    f"({cur:.0f} → {fc.iloc[0]:.0f}).</span></div>", unsafe_allow_html=True)
        out = fc.round(1).rename("Forecast AQI").to_frame()
        out["Category"] = [band(v)[0] for v in fc.values]
        out.index.name = "Date"
        st.download_button("Download forecast (CSV)", out.to_csv().encode(), f"{city}_forecast.csv", "text/csv")

    st.write("")
    with st.container(key="card_state"):
        st_cities = [c for c in CITIES if state_of(c) == state]
        vals = {c: float(run_forecast(c, 7).iloc[0]) for c in st_cities}
        avg_v = sum(vals.values()) / len(vals)
        st.markdown(f"**{state}: city-wise forecast for tomorrow**")
        chips = "".join(
            f"<span class='chip' style='background:{band(v)[1]};color:{band(v)[2]} !important'>{c}<b>{v:.0f}</b></span>"
            for c, v in vals.items())
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
        st.caption(f"State average: {avg_v:.0f} ({band(avg_v)[0]}) | {len(st_cities)} city/cities with data in {state}.")

    st.write("")
    with st.container(key="card_chart"):
        g, c = st.columns([1, 2.4])
        with g:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=cur,
                title=dict(text="Last recorded AQI", font=dict(size=14, color=T["muted"])),
                number=dict(font=dict(color=T["text"])),
                gauge=dict(axis=dict(range=[0, 500], tickcolor=T["muted"]),
                           bar=dict(color=T["gauge_bar"], thickness=0.25),
                           steps=[dict(range=list(r), color=b[2]) for r, b in zip(RANGES, BANDS)])))
            gauge.update_layout(**PLOT, height=330, margin=dict(l=20, r=20, t=60, b=10))
            st.plotly_chart(gauge, use_container_width=True)
        with c:
            hist = recent["AQI"].iloc[-hist_days:]
            fig = go.Figure()
            for (lo, hi), b in zip(RANGES, BANDS):
                fig.add_hrect(y0=lo, y1=hi, fillcolor=b[2], opacity=0.16, line_width=0)
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name=f"Actual (last {hist_days} days)",
                                     line=dict(color=ACC, width=3)))
            upper = fc.values + s["rmse"]
            lower = np.maximum(fc.values - s["rmse"], 0)
            fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]),
                                     y=list(upper) + list(lower[::-1]), fill="toself",
                                     fillcolor="rgba(231,76,60,0.18)", line=dict(width=0),
                                     name=f"±{s['rmse']:.0f} (typical model error)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[hist.index[-1]] + list(fc.index), y=[hist.iloc[-1]] + list(fc.values),
                                     name="LSTM Forecast", mode="lines+markers",
                                     line=dict(color="#E74C3C", width=3, dash="dash"), marker=dict(size=9)))
            fig.update_layout(**PLOT, height=330, margin=dict(l=10, r=10, t=40, b=10),
                              yaxis_title="AQI", legend=dict(orientation="h", y=1.18), hovermode="x unified")
            fig.update_yaxes(range=[0, max(hist.max(), upper.max()) * 1.15])
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Coloured bands show the AQI category (Good to Severe). The red shaded area is the range "
                   "based on the model's test error (RMSE).")

# ---------------- Tab 2: City / State comparison ----------------
with tab2:
    with st.container(key="card_compare"):
        st.subheader("Which place will be the most polluted?")
        o1, o2 = st.columns(2)
        level = o1.radio("Show by", ["City", "State"], horizontal=True)
        metric = o2.radio("Rank by", ["Tomorrow", "Week avg"], horizontal=True)
        with st.spinner("Forecasting all cities (pehli baar thoda time lagega)..."):
            rows = []
            for ct in CITIES:
                f = run_forecast(ct, 7)
                val = f.iloc[0] if metric == "Tomorrow" else f.mean()
                rows.append({"Name": ct, "State": state_of(ct), "Value": float(val),
                             "lat": COORDS[ct][0], "lon": COORDS[ct][1]})
        cdf = pd.DataFrame(rows)
        if level == "State":
            cdf = cdf.groupby("State", as_index=False).agg(
                Value=("Value", "mean"), lat=("lat", "mean"), lon=("lon", "mean"))
            cdf["Name"] = cdf["State"]
        cdf["Label"] = cdf["Value"].apply(lambda v: band(v)[0])
        cdf["Color"] = cdf["Value"].apply(lambda v: band(v)[1])
        cdf = cdf.sort_values("Value", ascending=False)

        left, right = st.columns(2)
        with left:
            figm = go.Figure(go.Scattergeo(
                lat=cdf["lat"], lon=cdf["lon"],
                text=cdf["Name"] + ": " + cdf["Value"].round(0).astype(int).astype(str),
                hoverinfo="text", mode="markers+text", textposition="top center",
                textfont=dict(color=T["text"], size=10),
                marker=dict(size=cdf["Value"] / 6 + 10, color=cdf["Color"],
                            line=dict(width=1.5, color=T["mline"]))))
            figm.update_geos(lataxis_range=[7, 36], lonaxis_range=[67, 98], showcountries=True,
                             countrycolor=T["country"], coastlinecolor=T["country"], landcolor=T["land"],
                             showocean=True, oceancolor=T["ocean"], showland=True, showframe=False,
                             bgcolor="rgba(0,0,0,0)")
            figm.update_layout(**PLOT, height=520, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(figm, use_container_width=True)
        with right:
            figb = go.Figure(go.Bar(x=cdf["Value"][::-1], y=cdf["Name"][::-1], orientation="h",
                                    marker_color=list(cdf["Color"][::-1]), marker_line_color=T["mline"],
                                    marker_line_width=1,
                                    text=cdf["Value"][::-1].round(0).astype(int), textposition="outside"))
            figb.update_layout(**PLOT, height=max(520, 26 * len(cdf) + 80),
                               xaxis_title=f"Predicted AQI ({metric.lower()})",
                               margin=dict(l=0, r=30, t=10, b=0))
            st.plotly_chart(figb, use_container_width=True)

        worst, best = cdf.iloc[0], cdf.iloc[-1]
        st.markdown(f"<div class='advice'><b>Highest forecast:</b> {worst['Name']} "
                    f"(AQI {worst['Value']:.0f}, {worst['Label']}) &nbsp;|&nbsp; "
                    f"<b>Lowest:</b> {best['Name']} (AQI {best['Value']:.0f}, {best['Label']})</div>",
                    unsafe_allow_html=True)
        if level == "State":
            st.caption("State value = average of the forecasts of that state's cities in the dataset.")

    st.write("")
    with st.container(key="card_multi"):
        st.markdown("**Compare forecast of selected cities**")
        pick = st.multiselect("Cities", CITIES, default=[city] + [c for c in CITIES[:2] if c != city])
        figc = go.Figure()
        for ct in pick:
            f = run_forecast(ct, 7)
            figc.add_trace(go.Scatter(x=f.index, y=f.values, mode="lines+markers", name=ct))
        figc.update_layout(**PLOT, height=360, yaxis_title="Forecast AQI", hovermode="x unified",
                           margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.15))
        st.plotly_chart(figc, use_container_width=True)

# ---------------- Tab 3: Pollutants ----------------
with tab3:
    with st.container(key="card_poll"):
        st.subheader(f"{city} ({state}): pollutant levels")
        pols = [f for f in FEATURES if f != "AQI"]
        latest = recent[pols].iloc[-1]
        prev = recent[pols].iloc[-8] if len(recent) > 8 else latest
        mc = st.columns(len(pols))
        for col, p in zip(mc, pols):
            col.metric(p, f"{latest[p]:.1f}", f"{latest[p] - prev[p]:+.1f} vs 7d ago", delta_color="inverse")

        sel = st.multiselect("Show pollutants", pols, default=["PM2.5", "PM10"])
        norm = st.checkbox("Scale each to 0-100% (easy to compare shapes)", value=False)
        win = recent[pols].iloc[-hist_days:]
        figp = go.Figure()
        for p in sel:
            y = win[p] / win[p].max() * 100 if norm and win[p].max() > 0 else win[p]
            figp.add_trace(go.Scatter(x=win.index, y=y, name=p, mode="lines"))
        figp.update_layout(**PLOT, height=380, yaxis_title="% of max" if norm else "Concentration",
                           hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10),
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(figp, use_container_width=True)

        corr = recent[FEATURES].iloc[-365:].corr()["AQI"].drop("AQI").sort_values()
        figr = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h", marker_color=ACC,
                                text=corr.round(2), textposition="outside"))
        figr.update_layout(**PLOT, height=300, xaxis_title="Correlation with AQI (last 365 days)",
                           margin=dict(l=10, r=30, t=10, b=10))
        st.plotly_chart(figr, use_container_width=True)

# ---------------- Tab 4: Model accuracy ----------------
with tab4:
    with st.container(key="card_acc"):
        st.subheader(f"{city}: model predictions on the last 90 test days")
        bt = get_backtest(city)
        figt = go.Figure()
        figt.add_trace(go.Scatter(x=bt["Date"], y=bt["Actual"], name="Actual", line=dict(color=ACC, width=3)))
        figt.add_trace(go.Scatter(x=bt["Date"], y=bt["Predicted"], name="LSTM Predicted",
                                  line=dict(color="#F39C12", width=2.5)))
        figt.update_layout(**PLOT, height=360, yaxis_title="AQI", hovermode="x unified",
                           margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.14))
        st.plotly_chart(figt, use_container_width=True)

        gain = (s["naive_rmse"] - s["rmse"]) / s["naive_rmse"] * 100
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("RMSE", f"{s['rmse']:.1f}")
        a2.metric("MAE", f"{s['mae']:.1f}")
        a3.metric("R²", f"{s['r2']:.3f}")
        a4.metric("Better than naive", f"{gain:.0f}%")

        l2, r2c = st.columns(2)
        err = bt["Predicted"] - bt["Actual"]
        with l2:
            fh = go.Figure(go.Histogram(x=err, nbinsx=25, marker_color=ACC))
            fh.update_layout(**PLOT, height=300, xaxis_title="Prediction error (Predicted - Actual)",
                             yaxis_title="Days", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fh, use_container_width=True)
        with r2c:
            lo = float(min(bt["Actual"].min(), bt["Predicted"].min()))
            hi = float(max(bt["Actual"].max(), bt["Predicted"].max()))
            fs = go.Figure()
            fs.add_trace(go.Scatter(x=bt["Actual"], y=bt["Predicted"], mode="markers",
                                    marker=dict(color=ACC, size=7, opacity=0.7), name="Days"))
            fs.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                    line=dict(color="#E74C3C", dash="dash"), name="Perfect"))
            fs.update_layout(**PLOT, height=300, xaxis_title="Actual AQI", yaxis_title="Predicted AQI",
                             margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fs, use_container_width=True)

    st.write("")
    with st.container(key="card_table"):
        st.markdown("**Accuracy of all trained cities**")
        tbl = pd.DataFrame(SUMMARY).T.loc[ALL_CITIES].copy()
        for col in ["rmse", "mae", "r2", "naive_rmse", "avg_aqi"]:
            tbl[col] = pd.to_numeric(tbl[col]).round(3)
        tbl["state"] = [state_of(c) for c in tbl.index]
        tbl["improvement_vs_naive_%"] = ((tbl["naive_rmse"] - tbl["rmse"]) / tbl["naive_rmse"] * 100).round(1)
        tbl["status"] = ["OK" if r >= MIN_R2 else "Low accuracy (hidden)" for r in tbl["r2"]]
        tbl.index.name = "City"
        st.markdown(html_table(tbl[["state", "rmse", "mae", "r2", "naive_rmse", "improvement_vs_naive_%",
                                    "n_days", "status"]]), unsafe_allow_html=True)

st.markdown("<div class='footer'>Academic project (Deep Learning, BTech Sem V). Data source: CPCB daily AQI, "
            "2015-2020. Forecast is for days after the end of the historical data, not live.</div>",
            unsafe_allow_html=True)