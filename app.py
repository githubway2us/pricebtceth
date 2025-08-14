import time
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh
import numpy as np

st.set_page_config(page_title="BTC & ETH Realtime + Alerts", layout="centered")

API = "https://api.binance.com/api/v3/ticker/price"
KLINE_API = "https://api.binance.com/api/v3/klines"
ORDER_BOOK_API = "https://api.binance.com/api/v3/depth"

# ---------- Timeframe Map ----------
timeframe_map = {
    "4 ชั่วโมง": {"seconds": 4 * 3600, "binance_interval": "4h"},
    "8 ชั่วโมง": {"seconds": 8 * 3600, "binance_interval": "8h"},
    "1 วัน": {"seconds": 24 * 3600, "binance_interval": "1d"},
    "1 สัปดาห์": {"seconds": 7 * 24 * 3600, "binance_interval": "1w"}
}

# ---------- ตลาดหุ้นทั่วโลก ----------
markets = [
    {"name": "New York (NYSE/Nasdaq)", "tz": "America/New_York", "open": "09:30", "close": "16:00"},
    {"name": "London (LSE)", "tz": "Europe/London", "open": "08:00", "close": "16:30"},
    {"name": "Tokyo (TSE)", "tz": "Asia/Tokyo", "open": "09:00", "close": "15:00"},
    {"name": "Sydney (ASX)", "tz": "Australia/Sydney", "open": "10:00", "close": "16:00"},
    {"name": "Hong Kong (HKEX)", "tz": "Asia/Hong_Kong", "open": "09:30", "close": "16:00"},
    {"name": "Shanghai (SSE)", "tz": "Asia/Shanghai", "open": "09:30", "close": "15:00"},
]

def get_market_status():
    rows = []
    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
    for m in markets:
        tz = pytz.timezone(m["tz"])
        local_now = utc_now.astimezone(tz)
        open_time = tz.localize(datetime.combine(local_now.date(), datetime.strptime(m["open"], "%H:%M").time()))
        close_time = tz.localize(datetime.combine(local_now.date(), datetime.strptime(m["close"], "%H:%M").time()))
        status = "เปิด" if open_time <= local_now <= close_time else "ปิด"
        rows.append({
            "ตลาด": m["name"],
            "เวลาเปิด": m["open"],
            "เวลาปิด": m["close"],
            "เวลาปัจจุบัน": local_now.strftime("%H:%M"),
            "สถานะ": status
        })
    return pd.DataFrame(rows)

# ---------- Helpers ----------
def get_price(symbol: str):
    try:
        r = requests.get(API, params={"symbol": symbol}, timeout=5)
        r.raise_for_status()
        return float(r.json()["price"])
    except:
        return None

def get_kline_data(symbol: str, interval: str, limit: int):
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(KLINE_API, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df
    except:
        return None

def get_order_book(symbol: str, limit: int = 100):
    try:
        params = {"symbol": symbol, "limit": limit}
        r = requests.get(ORDER_BOOK_API, params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except:
        return None

def color_by_target(price, target):
    if price is None or target is None:
        return "#888888"
    return "#00ff99" if price >= target else "#ff3366"

def format_big_price(price):
    if price is None:
        return "-"
    return f"{price:,.2f}"

def diff_text(price, target):
    if price is None or target is None:
        return "(n/a)"
    diff = price - target
    sign = "+" if diff >= 0 else ""
    arrow = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"
    return f"{arrow} {sign}{diff:,.2f}"

def get_countdown_str(tf_seconds):
    now = datetime.utcnow()
    epoch_now = int(now.timestamp())
    next_close_epoch = ((epoch_now // tf_seconds) + 1) * tf_seconds
    remaining = next_close_epoch - epoch_now
    h = str(remaining // 3600).zfill(2)
    m = str((remaining % 3600) // 60).zfill(2)
    s = str(remaining % 60).zfill(2)
    return f"{h}:{m}:{s}"

# ---------- Technical Analysis ----------
def calculate_fibonacci_levels(high, low):
    diff = high - low
    levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "100.0%": low
    }
    return levels

def calculate_macd(prices, fast=12, slow=26, signal=9):
    prices = pd.Series(prices)
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_rsi(prices, period=14):
    prices = pd.Series(prices)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_trend(symbol, kline_data):
    if kline_data is None or len(kline_data) < 2:
        return "N/A"
    closes = kline_data["close"].tail(2)
    if len(closes) < 2:
        return "N/A"
    price_diff = closes.iloc[-1] - closes.iloc[-2]
    x = np.arange(len(closes))
    slope, _ = np.polyfit(x, closes, 1)
    if price_diff > 0 and slope > 0:
        return "ขาขึ้น (Bullish)"
    elif price_diff < 0 and slope < 0:
        return "ขาลง (Bearish)"
    else:
        return "Sideways"

def detect_volume_spike(kline_data, threshold=2.5):
    if kline_data is None or len(kline_data) < 10:
        return False, 0
    volumes = kline_data["volume"]
    mean_vol = volumes[:-1].mean()
    std_vol = volumes[:-1].std()
    current_vol = volumes.iloc[-1]
    if std_vol == 0:
        return False, current_vol
    z_score = (current_vol - mean_vol) / std_vol
    return z_score > threshold, current_vol

# ---------- Price Prediction Game ----------
def check_prediction(symbol, prediction, pred_price, current_price):
    if pred_price is None or current_price is None or prediction is None:
        return None
    if prediction == "Higher" and current_price > pred_price:
        return True
    if prediction == "Lower" and current_price < pred_price:
        return True
    return False

def update_score(current_score, is_correct):
    if is_correct is None:
        return current_score
    if is_correct:
        return current_score * 18 if current_score > 0 else 18
    else:
        return max(0, current_score - 2)

def get_game_countdown(end_time):
    now = datetime.utcnow().replace(tzinfo=pytz.utc)
    remaining = (end_time - now).total_seconds()
    if remaining <= 0:
        return "00:00:00"
    h = str(int(remaining // 3600)).zfill(2)
    m = str(int((remaining % 3600) // 60)).zfill(2)
    s = str(int(remaining % 60)).zfill(2)
    return f"{h}:{m}:{s}"

# ---------- Styling ----------
st.markdown("""
    <style>
        .stApp {
            background-color: black;
        }
        .stDataFrame, .stMarkdown, .stPlotlyChart {
            background-color: black !important;
        }
        body, .stMarkdown, .stDataFrame, .stPlotlyChart {
            color: white !important;
        }
        .alert-box {
            background-color: #ff3366;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        .game-result {
            background-color: #00ccff;
            color: black;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Session State Initialization ----------
if "btc_history" not in st.session_state:
    st.session_state.btc_history = []
if "eth_history" not in st.session_state:
    st.session_state.eth_history = []
if "btc_entry" not in st.session_state:
    st.session_state.btc_entry = None
if "eth_entry" not in st.session_state:
    st.session_state.eth_entry = None
if "btc_pred_price" not in st.session_state:
    st.session_state.btc_pred_price = None
if "eth_pred_price" not in st.session_state:
    st.session_state.eth_pred_price = None
if "btc_prediction" not in st.session_state:
    st.session_state.btc_prediction = None
if "eth_prediction" not in st.session_state:
    st.session_state.eth_prediction = None
if "btc_session_score" not in st.session_state:
    st.session_state.btc_session_score = 0
if "eth_session_score" not in st.session_state:
    st.session_state.eth_session_score = 0
if "btc_total_score" not in st.session_state:
    st.session_state.btc_total_score = 0
if "eth_total_score" not in st.session_state:
    st.session_state.eth_total_score = 0
if "game_end_time" not in st.session_state:
    st.session_state.game_end_time = None
if "game_duration" not in st.session_state:
    st.session_state.game_duration = None
if "game_active" not in st.session_state:
    st.session_state.game_active = False

# ---------- Sidebar ----------
st.sidebar.markdown("## ⚙️ ตั้งค่า Alert")
btc_target = st.sidebar.number_input("🎯 ราคาเป้าหมาย BTC", min_value=0.0, value=70000.0, step=100.0, format="%.2f")
eth_target = st.sidebar.number_input("🎯 ราคาเป้าหมาย ETH", min_value=0.0, value=3500.0, step=10.0, format="%.2f")
refresh_sec = st.sidebar.slider("⏱ รอบอัปเดต (วินาที)", 1, 10, 2)
history_len = st.sidebar.slider("📊 จำนวนจุดกราฟ", 10, 200, 50)
vol_threshold = st.sidebar.slider("📈 เกณฑ์ Volume Spike (Z-score)", 1.0, 5.0, 2.5, 0.1)
selected_tf = st.sidebar.selectbox("🕒 เลือกไทม์เฟรม", list(timeframe_map.keys()))

st.sidebar.markdown("## 🎮 Price Prediction Game")
st.sidebar.markdown("Predict whether the next price will be higher or lower!")
game_duration_options = {
    "5 นาที": 5 * 60,
    "10 นาที": 10 * 60,
    "30 นาที": 30 * 60,
    "1 ชั่วโมง": 1 * 3600,
    "4 ชั่วโมง": 4 * 3600,
    "8 ชั่วโมง": 8 * 3600,
    "1 วัน": 24 * 3600,
    "1 สัปดาห์": 7 * 24 * 3600
}
game_duration_label = st.sidebar.selectbox("⏳ ระยะเวลาเกม", list(game_duration_options.keys()))
btc_pred = st.sidebar.selectbox("BTC Prediction", ["Higher", "Lower"], key="btc_pred")
eth_pred = st.sidebar.selectbox("ETH Prediction", ["Higher", "Lower"], key="eth_pred")
if st.sidebar.button("เริ่มเกม"):
    st.session_state.game_duration = game_duration_options[game_duration_label]
    st.session_state.game_end_time = datetime.utcnow().replace(tzinfo=pytz.utc) + timedelta(seconds=st.session_state.game_duration)
    st.session_state.game_active = True
    st.session_state.btc_session_score = 0
    st.session_state.eth_session_score = 0
    st.session_state.btc_prediction = btc_pred
    st.session_state.eth_prediction = eth_pred
    st.session_state.btc_pred_price = get_price("BTCUSDT")
    st.session_state.eth_pred_price = get_price("ETHUSDT")

# New button to submit new predictions without restarting the session
if st.session_state.game_active and st.sidebar.button("ส่งการทายใหม่"):
    st.session_state.btc_prediction = btc_pred
    st.session_state.eth_prediction = eth_pred

if st.session_state.game_active:
    countdown = get_game_countdown(st.session_state.game_end_time)
    st.sidebar.markdown(f"**เวลาคงเหลือ**: {countdown}")
    st.sidebar.markdown(f"**BTC Prediction**: {st.session_state.btc_prediction or 'None'}")
    st.sidebar.markdown(f"**BTC ราคาที่ทาย**: {format_big_price(st.session_state.btc_pred_price)}")
    st.sidebar.markdown(f"**ETH Prediction**: {st.session_state.eth_prediction or 'None'}")
    st.sidebar.markdown(f"**ETH ราคาที่ทาย**: {format_big_price(st.session_state.eth_pred_price)}")
    st.sidebar.markdown(f"**BTC Session Score**: {st.session_state.btc_session_score}")
    st.sidebar.markdown(f"**ETH Session Score**: {st.session_state.eth_session_score}")
    st.sidebar.markdown(f"**BTC Total Score**: {st.session_state.btc_total_score}")
    st.sidebar.markdown(f"**ETH Total Score**: {st.session_state.eth_total_score}")

st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")

# ---------- Game Session Logic ----------
if st.session_state.game_active and datetime.utcnow().replace(tzinfo=pytz.utc) >= st.session_state.game_end_time:
    st.session_state.btc_total_score += st.session_state.btc_session_score
    st.session_state.eth_total_score += st.session_state.eth_session_score
    st.markdown(
        f"<div class='game-result'>🎮 เกมจบแล้ว! BTC Session Score: {st.session_state.btc_session_score}, "
        f"ETH Session Score: {st.session_state.eth_session_score}</div>",
        unsafe_allow_html=True
    )
    st.session_state.game_active = False
    st.session_state.btc_session_score = 0
    st.session_state.eth_session_score = 0
    st.session_state.btc_prediction = None
    st.session_state.eth_prediction = None
    st.session_state.btc_pred_price = None
    st.session_state.eth_pred_price = None
    st.session_state.game_end_time = None

# ---------- Fetch Data ----------
btc = get_price("BTCUSDT")
eth = get_price("ETHUSDT")
btc_kline = get_kline_data("BTCUSDT", "1d", 2)
eth_kline = get_kline_data("ETHUSDT", "1d", 2)
btc_kline_tf = get_kline_data("BTCUSDT", timeframe_map.get(selected_tf, {"binance_interval": "4h"})["binance_interval"], history_len)
eth_kline_tf = get_kline_data("ETHUSDT", timeframe_map.get(selected_tf, {"binance_interval": "4h"})["binance_interval"], history_len)

# ---------- Price Prediction Game Logic ----------
if st.session_state.game_active:
    if btc is not None and st.session_state.btc_pred_price is not None and st.session_state.btc_prediction is not None:
        btc_result = check_prediction("BTC", st.session_state.btc_prediction, st.session_state.btc_pred_price, btc)
        st.session_state.btc_session_score = update_score(st.session_state.btc_session_score, btc_result)
        if btc_result is not None:
            st.session_state.btc_prediction = None  # Reset only prediction
    if eth is not None and st.session_state.eth_pred_price is not None and st.session_state.eth_prediction is not None:
        eth_result = check_prediction("ETH", st.session_state.eth_prediction, st.session_state.eth_pred_price, eth)
        st.session_state.eth_session_score = update_score(st.session_state.eth_session_score, eth_result)
        if eth_result is not None:
            st.session_state.eth_prediction = None  # Reset only prediction

# เก็บประวัติราคา
if btc is not None:
    st.session_state.btc_history.append(btc)
    if len(st.session_state.btc_history) > history_len:
        st.session_state.btc_history.pop(0)
if eth is not None:
    st.session_state.eth_history.append(eth)
    if len(st.session_state.eth_history) > history_len:
        st.session_state.eth_history.pop(0)

# ตั้งจุดเข้าเมื่อไม่มี
if st.session_state.btc_entry is None and btc is not None:
    st.session_state.btc_entry = btc
if st.session_state.eth_entry is None and eth is not None:
    st.session_state.eth_entry = eth

# ---------- Technical Analysis ----------
if btc_kline_tf is not None:
    btc_high = btc_kline_tf["high"].max()
    btc_low = btc_kline_tf["low"].min()
    btc_fib = calculate_fibonacci_levels(btc_high, btc_low)
else:
    btc_fib = {}

if eth_kline_tf is not None:
    eth_high = eth_kline_tf["high"].max()
    eth_low = eth_kline_tf["low"].min()
    eth_fib = calculate_fibonacci_levels(eth_high, eth_low)
else:
    eth_fib = {}

if len(st.session_state.btc_history) >= 26:
    btc_macd, btc_signal, btc_histogram = calculate_macd(st.session_state.btc_history)
    btc_rsi = calculate_rsi(st.session_state.btc_history)
else:
    btc_macd, btc_signal, btc_histogram, btc_rsi = None, None, None, None

if len(st.session_state.eth_history) >= 26:
    eth_macd, eth_signal, eth_histogram = calculate_macd(st.session_state.eth_history)
    eth_rsi = calculate_rsi(st.session_state.eth_history)
else:
    eth_macd, eth_signal, eth_histogram, eth_rsi = None, None, None, None

btc_trend = calculate_trend("BTCUSDT", btc_kline)
eth_trend = calculate_trend("ETHUSDT", eth_kline)

btc_vol_spike, btc_current_vol = detect_volume_spike(btc_kline_tf, vol_threshold) if btc_kline_tf is not None else (False, 0)
eth_vol_spike, eth_current_vol = detect_volume_spike(eth_kline_tf, vol_threshold) if eth_kline_tf is not None else (False, 0)

# ---------- Layout ----------
col1, col2 = st.columns(2)

# ---------- BTC Box ----------
btc_color = color_by_target(btc, btc_target)
btc_box = f"""
<div style='background:linear-gradient(135deg, {btc_color}33, #000000); 
            border: 2px solid {btc_color}; 
            padding: 16px; border-radius: 15px; color:white; 
            text-align:center; font-family:Arial; box-shadow:0 0 20px {btc_color}66'>
    <div style='font-size:20px;'>BTC / USDT</div>
    <div style='font-size:45px; font-weight:bold; color:{btc_color};'>{format_big_price(btc)}</div>
    <div style='font-size:20px;'>{diff_text(btc, btc_target)}</div>
    <div style='font-size:14px; color:#aaa;'>⏳ {selected_tf} ปิดแท่งใน {get_countdown_str(timeframe_map[selected_tf]['seconds'])}</div>
    <div style='font-size:16px; color:#00ccff;'>Trend (1 วัน): {btc_trend}</div>
</div>
"""
col1.markdown(btc_box, unsafe_allow_html=True)
if btc_vol_spike:
    col1.markdown(f"<div class='alert-box'>⚠️ BTC Volume Spike Detected! Current Volume: {btc_current_vol:,.2f}</div>", unsafe_allow_html=True)

# ---------- ETH Box ----------
eth_color = color_by_target(eth, eth_target)
eth_box = f"""
<div style='background:linear-gradient(135deg, {eth_color}33, #000000); 
            border: 2px solid {eth_color}; 
            padding: 16px; border-radius: 15px; color:white; 
            text-align:center; font-family:Arial; box-shadow:0 0 20px {eth_color}66'>
    <div style='font-size:20px;'>ETH / USDT</div>
    <div style='font-size:45px; font-weight:bold; color:{eth_color};'>{format_big_price(eth)}</div>
    <div style='font-size:20px;'>{diff_text(eth, eth_target)}</div>
    <div style='font-size:14px; color:#aaa;'>⏳ {selected_tf} ปิดแท่งใน {get_countdown_str(timeframe_map[selected_tf]['seconds'])}</div>
    <div style='font-size:16px; color:#00ccff;'>Trend (1 วัน): {eth_trend}</div>
</div>
"""
col2.markdown(eth_box, unsafe_allow_html=True)
if eth_vol_spike:
    col2.markdown(f"<div class='alert-box'>⚠️ ETH Volume Spike Detected! Current Volume: {eth_current_vol:,.2f}</div>", unsafe_allow_html=True)

# ---------- Charts ----------
btc_chart = go.Figure()
btc_chart.add_trace(go.Scatter(y=st.session_state.btc_history, mode="lines+markers", name="BTC", line=dict(color=btc_color)))
btc_chart.add_hline(y=st.session_state.btc_entry, line_dash="dot", line_color="#00ffff", annotation_text="Entry Point")
for level, price in btc_fib.items():
    btc_chart.add_hline(y=price, line_dash="dash", line_color="#FFD700", annotation_text=f"Fib {level}", annotation_position="right")
if st.session_state.btc_history:
    btc_min = min(st.session_state.btc_history)
    btc_max = max(st.session_state.btc_history)
    padding = (btc_max - btc_min) * 0.02
    btc_chart.update_layout(yaxis_range=[btc_min - padding, btc_max + padding])
btc_chart.update_layout(title="BTC / USDT", height=300, margin=dict(l=0,r=0,t=30,b=0),
                        paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))

eth_chart = go.Figure()
eth_chart.add_trace(go.Scatter(y=st.session_state.eth_history, mode="lines+markers", name="ETH", line=dict(color=eth_color)))
eth_chart.add_hline(y=st.session_state.eth_entry, line_dash="dot", line_color="#00ffff", annotation_text="Entry Point")
for level, price in eth_fib.items():
    eth_chart.add_hline(y=price, line_dash="dash", line_color="#FFD700", annotation_text=f"Fib {level}", annotation_position="right")
if st.session_state.eth_history:
    eth_min = min(st.session_state.eth_history)
    eth_max = max(st.session_state.eth_history)
    padding = (eth_max - eth_min) * 0.02
    eth_chart.update_layout(yaxis_range=[eth_min - padding, eth_max + padding])
eth_chart.update_layout(title="ETH / USDT", height=300, margin=dict(l=0,r=0,t=30,b=0),
                        paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))

col1.plotly_chart(btc_chart, use_container_width=True)
col2.plotly_chart(eth_chart, use_container_width=True)

# ---------- Indicator Charts ----------
if btc_macd is not None:
    btc_indicator_fig = go.Figure()
    btc_indicator_fig.add_trace(go.Scatter(y=btc_macd, mode="lines", name="MACD", line=dict(color="#00ff99")))
    btc_indicator_fig.add_trace(go.Scatter(y=btc_signal, mode="lines", name="Signal", line=dict(color="#ff3366")))
    btc_indicator_fig.add_trace(go.Bar(y=btc_histogram, name="Histogram", marker_color="#888888"))
    btc_indicator_fig.update_layout(title="BTC MACD", height=200, margin=dict(l=0,r=0,t=30,b=0),
                                   paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))
    col1.plotly_chart(btc_indicator_fig, use_container_width=True)

    btc_rsi_fig = go.Figure()
    btc_rsi_fig.add_trace(go.Scatter(y=btc_rsi, mode="lines", name="RSI", line=dict(color="#00ccff")))
    btc_rsi_fig.add_hline(y=70, line_dash="dash", line_color="#ff3366", annotation_text="Overbought")
    btc_rsi_fig.add_hline(y=30, line_dash="dash", line_color="#00ff99", annotation_text="Oversold")
    btc_rsi_fig.update_layout(title="BTC RSI", height=200, margin=dict(l=0,r=0,t=30,b=0),
                              paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))
    col1.plotly_chart(btc_rsi_fig, use_container_width=True)

    if btc_kline_tf is not None:
        btc_vol_fig = go.Figure()
        btc_vol_fig.add_trace(go.Bar(y=btc_kline_tf["volume"], name="Volume", marker_color="#888888"))
        btc_vol_fig.update_layout(title="BTC Volume", height=200, margin=dict(l=0,r=0,t=30,b=0),
                                  paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))
        col1.plotly_chart(btc_vol_fig, use_container_width=True)

if eth_macd is not None:
    eth_indicator_fig = go.Figure()
    eth_indicator_fig.add_trace(go.Scatter(y=eth_macd, mode="lines", name="MACD", line=dict(color="#00ff99")))
    eth_indicator_fig.add_trace(go.Scatter(y=eth_signal, mode="lines", name="Signal", line=dict(color="#ff3366")))
    eth_indicator_fig.add_trace(go.Bar(y=eth_histogram, name="Histogram", marker_color="#888888"))
    eth_indicator_fig.update_layout(title="ETH MACD", height=200, margin=dict(l=0,r=0,t=30,b=0),
                                   paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))
    col2.plotly_chart(eth_indicator_fig, use_container_width=True)

    eth_rsi_fig = go.Figure()
    eth_rsi_fig.add_trace(go.Scatter(y=eth_rsi, mode="lines", name="RSI", line=dict(color="#00ccff")))
    eth_rsi_fig.add_hline(y=70, line_dash="dash", line_color="#ff3366", annotation_text="Overbought")
    eth_rsi_fig.add_hline(y=30, line_dash="dash", line_color="#00ff99", annotation_text="Oversold")
    eth_rsi_fig.update_layout(title="ETH RSI", height=200, margin=dict(l=0,r=0,t=30,b=0),
                              paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))
    col2.plotly_chart(eth_rsi_fig, use_container_width=True)

    if eth_kline_tf is not None:
        eth_vol_fig = go.Figure()
        eth_vol_fig.add_trace(go.Bar(y=eth_kline_tf["volume"], name="Volume", marker_color="#888888"))
        eth_vol_fig.update_layout(title="ETH Volume", height=200, margin=dict(l=0,r=0,t=30,b=0),
                                  paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))
        col2.plotly_chart(eth_vol_fig, use_container_width=True)

# ---------- Market Status Table (at the bottom) ----------
st.markdown("## 🌎 เวลาเปิด–ปิดตลาดหุ้นทั่วโลก")
st.dataframe(get_market_status(), use_container_width=True)