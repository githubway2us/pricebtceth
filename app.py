import time
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pytz
from streamlit_autorefresh import st_autorefresh
import numpy as np
import sqlite3
import bcrypt
import uuid
import logging
import os
from dotenv import load_dotenv
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ตั้งค่า logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# โหลด environment variables
load_dotenv()

st.set_page_config(page_title="BTC & ETH Realtime + Alerts", layout="centered")

# API Endpoints
API = "https://api.binance.com/api/v3/ticker/price"
KLINE_API = "https://api.binance.com/api/v3/klines"
ORDER_BOOK_API = "https://api.binance.com/api/v3/depth"

# ---------- SQLite Database Setup ----------
def connect_db():
    conn = sqlite3.connect('game_database.db')
    # หากใช้ pysqlcipher3: conn.execute(f"PRAGMA key='{os.getenv('DB_KEY')}'")
    return conn

def init_db():
    conn = connect_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        chat_level INTEGER DEFAULT 1,
        created_at TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER,
        created_at TEXT,
        expires_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        session_number INTEGER,
        money REAL,
        credits REAL,
        btc_prediction TEXT,
        eth_prediction TEXT,
        btc_pred_price REAL,
        eth_pred_price REAL,
        btc_final_price REAL,
        eth_final_price REAL,
        timestamp TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS login_attempts (
        username TEXT PRIMARY KEY,
        attempts INTEGER DEFAULT 0,
        last_attempt TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        chat_count INTEGER,
        timestamp TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()

# เรียกใช้เพื่อสร้างฐานข้อมูล
init_db()

# ---------- Leaderboard Function ----------
def get_leaderboard(limit=10):
    conn = connect_db()
    c = conn.cursor()
    c.execute('''SELECT u.username, s.money, s.credits, s.timestamp
                 FROM users u
                 JOIN scores s ON u.id = s.user_id
                 WHERE s.id = (
                     SELECT MAX(id)
                     FROM scores
                     WHERE user_id = u.id
                 )
                 ORDER BY s.money DESC
                 LIMIT ?''', (limit,))
    leaderboard = c.fetchall()
    conn.close()
    return pd.DataFrame(leaderboard, columns=["ชื่อผู้ใช้", "เงิน (USDT)", "เครดิต", "เวลาล่าสุด"]).reset_index().rename(columns={"index": "อันดับ"})

# ---------- Authentication Functions ----------
def hash_password(password):
    try:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error hashing password: {e}")
        raise

def register_user(username, password):
    if not (3 <= len(username) <= 50):
        logging.warning(f"Invalid username length for {username}")
        return False, "ชื่อผู้ใช้ต้องมีความยาว 3-50 ตัวอักษร"
    if not (6 <= len(password) <= 50):
        logging.warning(f"Invalid password length for {username}")
        return False, "รหัสผ่านต้องมีความยาว 6-50 ตัวอักษร"
    if not username.isalnum():
        logging.warning(f"Invalid characters in username: {username}")
        return False, "ชื่อผู้ใช้ต้องประกอบด้วยตัวอักษรและตัวเลขเท่านั้น"
    
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        logging.warning(f"Username {username} already exists")
        return False, "ชื่อผู้ใช้นี้มีอยู่แล้ว"
    
    try:
        hashed_password = hash_password(password)
        c.execute("INSERT INTO users (username, password, chat_level, created_at) VALUES (?, ?, ?, ?)",
                  (username, hashed_password, 1, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        logging.info(f"User {username} registered successfully")
        return True, "ลงทะเบียนสำเร็จ"
    except sqlite3.Error as e:
        logging.error(f"Database error during registration for {username}: {e}")
        return False, f"เกิดข้อผิดพลาดในฐานข้อมูล: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error during registration for {username}: {e}")
        return False, f"เกิดข้อผิดพลาด: {str(e)}"
    finally:
        conn.close()

def check_login_attempts(username):
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT attempts, last_attempt FROM login_attempts WHERE username = ?", (username,))
    result = c.fetchone()
    if result and result[0] >= 5:
        last_attempt = datetime.strptime(result[1], "%Y-%m-%d %H:%M:%S")
        if (datetime.utcnow() - last_attempt).total_seconds() < 3600:
            logging.warning(f"Account {username} is locked due to too many failed attempts")
            return False
    return True

def update_login_attempts(username, success):
    conn = connect_db()
    c = conn.cursor()
    if success:
        c.execute("DELETE FROM login_attempts WHERE username = ?", (username,))
    else:
        c.execute("INSERT OR REPLACE INTO login_attempts (username, attempts, last_attempt) VALUES (?, COALESCE((SELECT attempts + 1 FROM login_attempts WHERE username = ?), 1), ?)",
                  (username, username, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def login_user(username, password):
    if not check_login_attempts(username):
        return None
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user:
        try:
            if bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
                update_login_attempts(username, True)
                logging.info(f"User {username} logged in successfully")
                return user[0]
            else:
                update_login_attempts(username, False)
                logging.warning(f"Failed login attempt for username: {username} (wrong password)")
                return None
        except ValueError as e:
            logging.error(f"Invalid password hash for {username}: {e}")
            update_login_attempts(username, False)
            return None
    else:
        update_login_attempts(username, False)
        logging.warning(f"Failed login attempt for username: {username} (user not found)")
        return None

def create_session(user_id):
    token = str(uuid.uuid4())
    expires_at = (datetime.utcnow() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
              (token, user_id, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), expires_at))
    conn.commit()
    conn.close()
    return token

def validate_session(token):
    if not token:
        return None
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,))
    session = c.fetchone()
    conn.close()
    if session and datetime.strptime(session[1], "%Y-%m-%d %H:%M:%S") > datetime.utcnow():
        return session[0]
    return None

def save_score(user_id, session_number, money, credits, btc_prediction, eth_prediction, btc_pred_price, eth_pred_price, btc_final_price, eth_final_price):
    conn = connect_db()
    c = conn.cursor()
    c.execute('''INSERT INTO scores (user_id, session_number, money, credits, btc_prediction, eth_prediction,
                 btc_pred_price, eth_pred_price, btc_final_price, eth_final_price, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (user_id, session_number, money, credits, btc_prediction, eth_prediction,
               btc_pred_price, eth_pred_price, btc_final_price, eth_final_price, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    logging.info(f"Saved score for user_id {user_id}, session {session_number}")

def get_score_history(user_id):
    conn = connect_db()
    c = conn.cursor()
    c.execute('''SELECT session_number, money, credits, btc_prediction, eth_prediction,
                 btc_pred_price, eth_pred_price, btc_final_price, eth_final_price, timestamp
                 FROM scores WHERE user_id = ? ORDER BY timestamp DESC''', (user_id,))
    history = c.fetchall()
    conn.close()
    return pd.DataFrame(history, columns=["เซสชัน", "เงิน", "เครดิต", "BTC การทาย", "ETH การทาย",
                                         "BTC ราคาทาย", "ETH ราคาทาย", "BTC ราคาจริง", "ETH ราคาจริง", "เวลา"])

# ---------- Chat Level Functions ----------
def log_chat(user_id):
    conn = connect_db()
    c = conn.cursor()
    c.execute("INSERT INTO chat_logs (user_id, chat_count, timestamp) VALUES (?, 1, ?)",
              (user_id, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    
    # นับจำนวนแชททั้งหมดของผู้ใช้
    c.execute("SELECT SUM(chat_count) FROM chat_logs WHERE user_id = ?", (user_id,))
    total_chats = c.fetchone()[0] or 0
    
    # กำหนดระดับตามจำนวนแชท (ทุก 10 แชทเพิ่ม 1 ระดับ)
    new_level = max(1, total_chats // 10)
    
    # อัปเดตระดับในตาราง users
    c.execute("UPDATE users SET chat_level = ? WHERE id = ?", (new_level, user_id))
    conn.commit()
    conn.close()
    return new_level

def get_chat_level(user_id):
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT chat_level FROM users WHERE id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 1

def get_total_users():
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    total = c.fetchone()[0]
    conn.close()
    return total

def get_chat_history(user_id):
    conn = connect_db()
    c = conn.cursor()
    c.execute("SELECT chat_count, timestamp FROM chat_logs WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    history = c.fetchall()
    conn.close()
    return pd.DataFrame(history, columns=["จำนวนแชท", "เวลา"])

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
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    try:
        r = session.get(API, params={"symbol": symbol}, timeout=5)
        r.raise_for_status()
        if r.status_code == 429:
            time.sleep(60)
            r = session.get(API, params={"symbol": symbol}, timeout=5)
        return float(r.json()["price"])
    except Exception as e:
        logging.error(f"Failed to fetch price for {symbol}: {e}")
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
    except Exception as e:
        logging.error(f"Failed to fetch kline data for {symbol}: {e}")
        return None

def get_order_book(symbol: str, limit: int = 100):
    try:
        params = {"symbol": symbol, "limit": limit}
        r = requests.get(ORDER_BOOK_API, params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"Failed to fetch order book for {symbol}: {e}")
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
def check_prediction(symbol, prediction, pred_price, final_price):
    if pred_price is None or final_price is None or prediction is None:
        return None, 0
    price_diff = abs(final_price - pred_price)
    if prediction == "แทงขึ้น" and final_price > pred_price:
        return True, price_diff
    if prediction == "แทงลง" and final_price < pred_price:
        return True, price_diff
    return False, price_diff

def update_money(current_money, is_correct, price_diff):
    if is_correct is None:
        return current_money, 0
    score = price_diff * 125 if is_correct else price_diff * 25
    if is_correct:
        return current_money + score, score
    else:
        return current_money - score, -score

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
        .bankrupt-box {
            background-color: #ff0000;
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            margin-top: 10px;
        }
        .profit-positive {
            color: #00ff99;
        }
        .profit-negative {
            color: #ff3366;
        }
        .leaderboard-box {
            background-color: #1a1a1a;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .leaderboard-box table {
            width: 100%;
            border-collapse: collapse;
        }
        .leaderboard-box th, .leaderboard-box td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #444;
        }
        .leaderboard-box th {
            background-color: #00ccff;
            color: black;
        }
        .leaderboard-box tr:nth-child(even) {
            background-color: #2a2a2a;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Session State Initialization ----------
if "session_token" not in st.session_state:
    st.session_state.session_token = None
if "username" not in st.session_state:
    st.session_state.username = None
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
if "money" not in st.session_state:
    st.session_state.money = 10000000
if "credits" not in st.session_state:
    st.session_state.credits = 100
if "score_history" not in st.session_state:
    st.session_state.score_history = []
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "session_number" not in st.session_state:
    st.session_state.session_number = 0
if "game_end_time" not in st.session_state:
    st.session_state.game_end_time = None
if "game_duration" not in st.session_state:
    st.session_state.game_duration = None
if "game_active" not in st.session_state:
    st.session_state.game_active = False
if "bankrupt" not in st.session_state:
    st.session_state.bankrupt = False
if "btc_final_price" not in st.session_state:
    st.session_state.btc_final_price = None
if "eth_final_price" not in st.session_state:
    st.session_state.eth_final_price = None

# ---------- Login/Register UI ----------
st.sidebar.markdown("## 🔐 เข้าสู่ระบบ / ลงทะเบียน")
if not st.session_state.session_token:
    tab1, tab2 = st.sidebar.tabs(["เข้าสู่ระบบ", "ลงทะเบียน"])
    
    with tab1:
        login_username = st.text_input("ชื่อผู้ใช้", key="login_username")
        login_password = st.text_input("รหัสผ่าน", type="password", key="login_password")
        if st.button("เข้าสู่ระบบ"):
            if not check_login_attempts(login_username):
                st.error("บัญชีถูกล็อกชั่วคราว กรุณาลองใหม่ใน 1 ชั่วโมง")
            else:
                user_id = login_user(login_username, login_password)
                if user_id:
                    st.session_state.session_token = create_session(user_id)
                    st.session_state.username = login_username
                    st.success(f"ยินดีต้อนรับ {login_username}!")
                    score_history = get_score_history(user_id)
                    if not score_history.empty:
                        last_session = score_history.iloc[0]
                        st.session_state.session_number = last_session["เซสชัน"]
                        st.session_state.money = last_session["เงิน"]
                        st.session_state.credits = last_session["เครดิต"]
                    st.rerun()
                else:
                    st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    
    with tab2:
        reg_username = st.text_input("ชื่อผู้ใช้ใหม่", key="reg_username")
        reg_password = st.text_input("รหัสผ่านใหม่", type="password", key="reg_password")
        if st.button("ลงทะเบียน"):
            success, message = register_user(reg_username, reg_password)
            if success:
                st.success(message)
            else:
                st.error(message)
else:
    user_id = validate_session(st.session_state.session_token)
    if user_id:
        st.sidebar.markdown(f"**ผู้ใช้**: {st.session_state.username}")
        st.sidebar.markdown(f"**ระดับการแชท**: {get_chat_level(user_id)}")
        st.sidebar.markdown(f"**จำนวนผู้ใช้ที่สมัคร**: {get_total_users()}")
        
        # เพิ่มช่องแชท
        #st.sidebar.markdown("## 💬 แชท")
        #hat_input = st.sidebar.text_input("พิมพ์ข้อความ:", key="chat_input")
        #if st.sidebar.button("ส่งข้อความ"):
            #if chat_input:
                #log_chat(user_id)
                #st.sidebar.success(f"ข้อความส่งแล้ว! ระดับปัจจุบัน: {get_chat_level(user_id)}")
            #else:
                #st.sidebar.error("กรุณาพิมพ์ข้อความก่อนส่ง")
        
        # แสดงประวัติการแชท
        st.sidebar.markdown("## 📜 ฝึกให้มีวินัย")
        #chat_history = get_chat_history(user_id)
        #if not chat_history.empty:
            #st.sidebar.dataframe(chat_history, use_container_width=True)
        
        if st.sidebar.button("ออกจากระบบ"):
            conn = connect_db()
            c = conn.cursor()
            c.execute("DELETE FROM sessions WHERE token = ?", (st.session_state.session_token,))
            conn.commit()
            conn.close()
            logging.info(f"User {st.session_state.username} logged out")
            st.session_state.session_token = None
            st.session_state.username = None
            st.session_state.money = 10000000
            st.session_state.credits = 100
            st.session_state.score_history = []
            st.session_state.prediction_history = []
            st.session_state.session_number = 0
            st.session_state.game_active = False
            st.session_state.bankrupt = False
            st.rerun()
    else:
        st.session_state.session_token = None
        st.session_state.username = None
        st.markdown("<div class='alert-box'>⚠️ เซสชันหมดอายุ กรุณาเข้าสู่ระบบใหม่</div>", unsafe_allow_html=True)
        st.rerun()


# แนะนำโปรเจคและวิธีใช้
st.markdown("""
### ยินดีต้อนรับสู่ #PUK คัมภีร์สายกระบี่คริปโต
""")

# Restrict game access if not logged in
if not st.session_state.session_token or not validate_session(st.session_state.session_token):
    st.markdown("""
            ### โปรเจคนี้
            คุณสามารถเข้าดูและดาวน์โหลดโปรเจคได้ที่: [GitHub Repository](https://github.com/githubway2us/pricebtceth)

            ### วิธีใช้โปรแกรม
            1. ลงทะเบียนและเข้าสู่ระบบก่อนเล่นเกม
            2. เมื่อเข้าสู่ระบบแล้ว คุณจะสามารถเข้าถึงฟีเจอร์ของเกมได้
            3. หากยังไม่ได้เข้าสู่ระบบ โปรแกรมจะแจ้งเตือนให้ล็อกอินก่อนเล่น
            <div class='alert-box'>⚠️ กรุณาเข้าสู่ระบบเพื่อเล่นเกม!</div>
            """, unsafe_allow_html=True)
    st.stop()

# ---------- Sidebar ----------
user_id = validate_session(st.session_state.session_token)
st.sidebar.markdown("## ⚙️ ตั้งค่า Alert")
btc_target = st.sidebar.number_input("🎯 ราคาเป้าหมาย BTC", min_value=0.0, value=120000.0, step=100.0, format="%.2f")
eth_target = st.sidebar.number_input("🎯 ราคาเป้าหมาย ETH", min_value=0.0, value=4500.0, step=10.0, format="%.2f")
refresh_sec = st.sidebar.slider("⏱ รอบอัปเดต (วินาที)", 1, 10, 2)
history_len = st.sidebar.slider("📊 จำนวนจุดกราฟ", 10, 200, 50)
vol_threshold = st.sidebar.slider("📈 เกณฑ์ Volume Spike (Z-score)", 1.0, 5.0, 2.5, 0.1)
selected_tf = st.sidebar.selectbox("🕒 เลือกไทม์เฟรม", list(timeframe_map.keys()))

st.sidebar.markdown("## 🎮 Price Prediction Game")
st.sidebar.markdown("Predict whether the next price will be higher or lower!")
st.sidebar.markdown(f"**เงินคงเหลือ**: {st.session_state.money:,.2f} USDT")
profit_loss = st.session_state.money - 10000000
profit_class = "profit-positive" if profit_loss >= 0 else "profit-negative"
st.sidebar.markdown(f"<div class='{profit_class}'>**กำไร/ขาดทุน**: {profit_loss:+,.2f} USDT</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"**เครดิตคงเหลือ**: {st.session_state.credits}")

# Exchange money for credits
exchange_amount = st.sidebar.number_input("แลกเงินเป็นเครดิต (1,000 USDT = 1 เครดิต)", min_value=0.0, step=1000.0, format="%.2f")
if st.sidebar.button("แลกเครดิต") and exchange_amount > 0:
    credits_to_add = exchange_amount / 1000
    if exchange_amount <= st.session_state.money:
        st.session_state.credits += credits_to_add
        st.session_state.money -= exchange_amount
        st.session_state.prediction_history.append({
            "เซสชัน": st.session_state.session_number,
            "เหรียญ": "N/A",
            "สถานะ": "แลกเครดิต",
            "ราคาที่ทาย": "-",
            "ราคาจริง": "-",
            "เปลี่ยนแปลงเงิน": -exchange_amount,
            "เวลา": datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
        })
        save_score(
            user_id,
            st.session_state.session_number,
            st.session_state.money,
            st.session_state.credits,
            None, None, None, None, None, None
        )
    else:
        st.sidebar.error("เงินไม่พอสำหรับการแลกเครดิต!")

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
btc_pred = st.sidebar.selectbox("BTC Prediction", ["แทงขึ้น", "แทงลง", "ไม่มี"], key="btc_pred", index=2)
eth_pred = st.sidebar.selectbox("ETH Prediction", ["แทงขึ้น", "แทงลง", "ไม่มี"], key="eth_pred", index=2)

# Check if enough credits to start game
can_start = (st.session_state.credits >= 10 if btc_pred != "ไม่มี" and eth_pred != "ไม่มี" else
             st.session_state.credits >= 5 if btc_pred != "ไม่มี" or eth_pred != "ไม่มี" else False)
if st.sidebar.button("เริ่มเกม", disabled=not can_start or st.session_state.bankrupt):
    st.session_state.game_duration = game_duration_options[game_duration_label]
    st.session_state.game_end_time = datetime.utcnow().replace(tzinfo=pytz.utc) + timedelta(seconds=st.session_state.game_duration)
    st.session_state.game_active = True
    st.session_state.btc_prediction = btc_pred if btc_pred != "ไม่มี" else None
    st.session_state.eth_prediction = eth_pred if eth_pred != "ไม่มี" else None
    st.session_state.btc_pred_price = get_price("BTCUSDT") if btc_pred != "ไม่มี" else None
    st.session_state.eth_pred_price = get_price("ETHUSDT") if eth_pred != "ไม่มี" else None
    st.session_state.btc_final_price = None
    st.session_state.eth_final_price = None
    st.session_state.credits -= 10 if btc_pred != "ไม่มี" and eth_pred != "ไม่มี" else 5
    st.session_state.session_number += 1
    st.session_state.bankrupt = False
    save_score(
        user_id,
        st.session_state.session_number,
        st.session_state.money,
        st.session_state.credits,
        st.session_state.btc_prediction,
        st.session_state.eth_prediction,
        st.session_state.btc_pred_price,
        st.session_state.eth_pred_price,
        None, None
    )
    logging.info(f"User {st.session_state.username} started game session {st.session_state.session_number}")

# Submit new prediction with credit check
can_submit = (st.session_state.game_active and not st.session_state.bankrupt and
              ((btc_pred != "ไม่มี" and st.session_state.credits >= 5) or
               (eth_pred != "ไม่มี" and st.session_state.credits >= 5) or
               (btc_pred != "ไม่มี" and eth_pred != "ไม่มี" and st.session_state.credits >= 10)))
if st.session_state.game_active and st.sidebar.button("ส่งการทายใหม่", disabled=not can_submit):
    if btc_pred != "ไม่มี" and st.session_state.btc_prediction != btc_pred:
        st.session_state.btc_prediction = btc_pred
        st.session_state.btc_pred_price = get_price("BTCUSDT")
        st.session_state.credits -= 5
    if eth_pred != "ไม่มี" and st.session_state.eth_prediction != eth_pred:
        st.session_state.eth_prediction = eth_pred
        st.session_state.eth_pred_price = get_price("ETHUSDT")
        st.session_state.credits -= 5 if btc_pred == "ไม่มี" or st.session_state.btc_prediction == btc_pred else 0
    save_score(
        user_id,
        st.session_state.session_number,
        st.session_state.money,
        st.session_state.credits,
        st.session_state.btc_prediction,
        st.session_state.eth_prediction,
        st.session_state.btc_pred_price,
        st.session_state.eth_pred_price,
        None, None
    )
    logging.info(f"User {st.session_state.username} submitted new prediction for session {st.session_state.session_number}")

if st.session_state.game_active and not st.session_state.bankrupt:
    countdown = get_game_countdown(st.session_state.game_end_time)
    st.sidebar.markdown(f"**เวลาคงเหลือ**: {countdown}")
    st.sidebar.markdown(f"**BTC สถานะ**: {st.session_state.btc_prediction or 'ไม่มี'}")
    st.sidebar.markdown(f"**BTC ราคาที่ทาย**: {format_big_price(st.session_state.btc_pred_price)}")
    st.sidebar.markdown(f"**BTC ราคาเข้า**: {format_big_price(st.session_state.btc_entry)}")
    st.sidebar.markdown(f"**ETH สถานะ**: {st.session_state.eth_prediction or 'ไม่มี'}")
    st.sidebar.markdown(f"**ETH ราคาที่ทาย**: {format_big_price(st.session_state.eth_pred_price)}")
    st.sidebar.markdown(f"**ETH ราคาเข้า**: {format_big_price(st.session_state.eth_entry)}")
    st.sidebar.markdown(f"**เงินคงเหลือ**: {st.session_state.money:,.2f} USDT")
    st.sidebar.markdown(f"<div class='{profit_class}'>**กำไร/ขาดทุน**: {profit_loss:+,.2f} USDT</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**เครดิตคงเหลือ**: {st.session_state.credits}")

# Display session history
if st.session_state.score_history:
    st.sidebar.markdown("## 📜 ประวัติเซสชัน")
    score_df = pd.DataFrame(st.session_state.score_history, columns=["เซสชัน", "BTC Money Change", "ETH Money Change", "เวลา"])
    score_df.index = score_df.index + 1
    st.sidebar.dataframe(score_df, use_container_width=True)

# Display prediction history
if st.session_state.prediction_history:
    st.sidebar.markdown("## 📜 ประวัติการทาย")
    pred_df = pd.DataFrame(st.session_state.prediction_history, columns=["เซสชัน", "เหรียญ", "สถานะ", "ราคาที่ทาย", "ราคาจริง", "เปลี่ยนแปลงเงิน", "เวลา"])
    pred_df.index = pred_df.index + 1
    st.sidebar.dataframe(pred_df, use_container_width=True)

# Display user score history from database
if user_id:
    st.sidebar.markdown("## 📊 ประวัติคะแนนย้อนหลัง")
    score_history = get_score_history(user_id)
    if not score_history.empty:
        st.sidebar.dataframe(score_history, use_container_width=True)

# Display leaderboard
st.sidebar.markdown("## 🏆 กระดานผู้นำ")
leaderboard_df = get_leaderboard()
if not leaderboard_df.empty:
    leaderboard_df["อันดับ"] = leaderboard_df["อันดับ"] + 1
    leaderboard_df["เงิน (USDT)"] = leaderboard_df["เงิน (USDT)"].apply(lambda x: f"{x:,.2f}")
    leaderboard_df["เครดิต"] = leaderboard_df["เครดิต"].apply(lambda x: f"{x:,.2f}")
    st.sidebar.dataframe(leaderboard_df[["อันดับ", "ชื่อผู้ใช้", "เงิน (USDT)", "เครดิต"]], use_container_width=True)
else:
    st.sidebar.markdown("<div class='alert-box'>ไม่มีข้อมูลในกระดานผู้นำ</div>", unsafe_allow_html=True)

st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")

# ---------- Game Session Logic ----------
if st.session_state.game_active and datetime.utcnow().replace(tzinfo=pytz.utc) >= st.session_state.game_end_time and not st.session_state.bankrupt:
    st.session_state.btc_final_price = get_price("BTCUSDT") if st.session_state.btc_prediction is not None else None
    st.session_state.eth_final_price = get_price("ETHUSDT") if st.session_state.eth_prediction is not None else None

    initial_money = st.session_state.money
    btc_money_change = 0
    eth_money_change = 0

    if st.session_state.btc_prediction is not None and st.session_state.btc_pred_price is not None and st.session_state.btc_final_price is not None:
        btc_result, btc_price_diff = check_prediction("BTC", st.session_state.btc_prediction, st.session_state.btc_pred_price, st.session_state.btc_final_price)
        st.session_state.money, btc_money_change = update_money(st.session_state.money, btc_result, btc_price_diff)
        st.session_state.prediction_history.append({
            "เซสชัน": st.session_state.session_number,
            "เหรียญ": "BTC",
            "สถานะ": st.session_state.btc_prediction,
            "ราคาที่ทาย": format_big_price(st.session_state.btc_pred_price),
            "ราคาจริง": format_big_price(st.session_state.btc_final_price),
            "เปลี่ยนแปลงเงิน": btc_money_change,
            "เวลา": datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
        })

    if st.session_state.eth_prediction is not None and st.session_state.eth_pred_price is not None and st.session_state.eth_final_price is not None:
        eth_result, eth_price_diff = check_prediction("ETH", st.session_state.eth_prediction, st.session_state.eth_pred_price, st.session_state.eth_final_price)
        st.session_state.money, eth_money_change = update_money(st.session_state.money, eth_result, eth_price_diff)
        st.session_state.prediction_history.append({
            "เซสชัน": st.session_state.session_number,
            "เหรียญ": "ETH",
            "สถานะ": st.session_state.eth_prediction,
            "ราคาที่ทาย": format_big_price(st.session_state.eth_pred_price),
            "ราคาจริง": format_big_price(st.session_state.eth_final_price),
            "เปลี่ยนแปลงเงิน": eth_money_change,
            "เวลา": datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
        })

    st.session_state.score_history.append({
        "เซสชัน": st.session_state.session_number,
        "BTC Money Change": btc_money_change,
        "ETH Money Change": eth_money_change,
        "เวลา": datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
    })

    save_score(
        user_id,
        st.session_state.session_number,
        st.session_state.money,
        st.session_state.credits,
        st.session_state.btc_prediction,
        st.session_state.eth_prediction,
        st.session_state.btc_pred_price,
        st.session_state.eth_pred_price,
        st.session_state.btc_final_price,
        st.session_state.eth_final_price
    )

    if st.session_state.money < 0:
        st.session_state.bankrupt = True
        st.markdown(
            "<div class='bankrupt-box'>💥 พอร์ตคุณแตกแล้ว! กรุณาเริ่มเกมใหม่</div>",
            unsafe_allow_html=True
        )
        st.session_state.money = 10000000
        st.session_state.credits = 100
        st.session_state.session_number += 1
        save_score(
            user_id,
            st.session_state.session_number,
            st.session_state.money,
            st.session_state.credits,
            None, None, None, None, None, None
        )
        logging.info(f"User {st.session_state.username} went bankrupt in session {st.session_state.session_number}")

    st.markdown(
        f"<div class='game-result'>🎮 เกมจบแล้ว! เงินคงเหลือ: {st.session_state.money:,.2f} USDT</div>",
        unsafe_allow_html=True
    )
    st.session_state.game_active = False
    st.session_state.btc_prediction = None
    st.session_state.eth_prediction = None
    st.session_state.btc_pred_price = None
    st.session_state.eth_pred_price = None
    st.session_state.btc_final_price = None
    st.session_state.eth_final_price = None
    st.session_state.game_end_time = None
    logging.info(f"User {st.session_state.username} ended game session {st.session_state.session_number}")

# ---------- Fetch Data ----------
btc = get_price("BTCUSDT")
eth = get_price("ETHUSDT")
btc_kline = get_kline_data("BTCUSDT", "1d", 2)
eth_kline = get_kline_data("ETHUSDT", "1d", 2)
btc_kline_tf = get_kline_data("BTCUSDT", timeframe_map.get(selected_tf, {"binance_interval": "4h"})["binance_interval"], history_len)
eth_kline_tf = get_kline_data("ETHUSDT", timeframe_map.get(selected_tf, {"binance_interval": "4h"})["binance_interval"], history_len)

# ---------- Store prices during game ----------
if st.session_state.game_active and not st.session_state.bankrupt:
    if btc is not None and st.session_state.btc_prediction is not None:
        st.session_state.btc_final_price = btc
    if eth is not None and st.session_state.eth_prediction is not None:
        st.session_state.eth_final_price = eth

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