import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime

# ==========================================
# 0. 基本設定與日期解封
# ==========================================
st.set_page_config(page_title="量化回測實驗室", layout="wide")
st.title("🔬 終極量化回測實驗室 (邏輯校準版)")

# 🌟 解決 2026 日期選取限制
today = datetime.now().date()
min_date = datetime(2000, 1, 1).date()

# ==========================================
# 1. 側邊欄控制面板
# ==========================================
st.sidebar.title("⚙️ 控制面板")
engine = st.sidebar.selectbox("🧠 模擬引擎", ["1. 歷史區塊抽樣 (Block)", "2. 數學模型 (GBM)"])

initial_capital = st.sidebar.number_input("初始資金 (元)", min_value=100000, value=2000000, step=100000)
sim_years = st.sidebar.slider("⏳ 模擬未來幾年？", min_value=1, max_value=30, value=10)
N = st.sidebar.slider("模擬次數", min_value=1000, max_value=10000, value=5000)

st.sidebar.header("📅 買入策略設定")
dca_parts = st.sidebar.slider("分批次數", min_value=2, max_value=48, value=12)
dca_interval = st.sidebar.slider("買入頻率 (交易日)", min_value=5, max_value=60, value=21)

if "歷史" in engine:
    ticker = st.sidebar.text_input("代碼", value="0050.TW")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("開始", value=datetime(2008, 1, 1).date(), min_value=min_date, max_value=today)
    end_date = col2.date_input("結束", value=today, min_value=min_date, max_value=today)
    block_size = st.sidebar.slider("區塊大小", 5, 60, 21)
else:
    mu_base = st.sidebar.number_input("年報酬 (%)", value=9.0) / 100
    sig_base = st.sidebar.number_input("波動率 (%)", value=16.0) / 100
    df_t = st.sidebar.slider("肥尾強度", 2, 30, 3)

lev_mult = st.sidebar.number_input("槓桿倍數", 1.0, 5.0, 2.0, 0.5)
drag_annual = st.sidebar.slider("槓桿耗損 (%)", 0.0, 10.0, 1.5) / 100
drop_threshold = st.sidebar.slider("抄底觸發 (%)", 5, 50, 20) / 100
transfer_pct = st.sidebar.slider("轉入比例 (%)", 10, 100, 20) / 100

# ==========================================
# 2. 核心運算
# ==========================================
@st.cache_data(show_spinner=False, ttl=600)
def get_hist_data(tkr, start, end):
    try:
        data = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty: return None
        return data['Close'].pct_change().dropna().values.flatten()
    except: return None

if st.sidebar.button("🚀 開始運算", type="primary", use_container_width=True):
    with st.spinner('運算中...'):
        days = sim_years * 252
        dt = 1/252
        cash_growth = np.exp(0.01 * dt)
        
        if "歷史" in engine:
            rets = get_hist_data(ticker, start_date, end_date)
            if rets is None: st.error("抓不到資料"); st.stop()
            indices = np.random.randint(0, len(rets)-block_size, (int(np.ceil(days/block_size)), N))
            sim_ret_base = np.zeros((days, N))
            for b in range(indices.shape[0]):
                for i in range(block_size):
                    if b*block_size+i < days: sim_ret_base[b*block_size+i, :] = rets[indices[b,:]+i]
        else:
            Z = np.clip(np.random.standard_t(df_t, (days, N)) * np.sqrt(1/3), -15, 15)
            sim_ret_base = np.exp((mu_base - 0.5*sig_base**2)*dt + sig_base*np.sqrt(dt)*Z) - 1
            
        sim_ret_lev = (sim_ret_base * lev_mult) - (drag_annual/252)
        
        # 🌟 破產防線：乘數不可小於 0
        m_B, m_L = np.maximum(0, 1+sim_ret_base), np.maximum(0, 1+sim_ret_lev)

        # 策略初始化 (全部歸一化到 initial_capital)
        v1, v2_c, v2_L, v3_c, v3_L, v4, v5_B, v5_L, v6_c, v6_s = [np.ones(N)*initial_capital for _ in range(10)]
        v2_c, v2_L, v3_c, v3_L = v2_c*0.5, v2_L*0.5, v3_c*0.5, v3_L*0.5
        v5_L, v6_s = np.zeros(N), np.zeros(N)
        
        ath = np.ones(N)
        trig = np.zeros(N, dtype=bool)
        dca_amt = initial_capital / dca_parts

        for d in range(days):
            rb, rl = m_B[d], m_L[d]
            # 策略 1: 100% 槓桿
            v1 *= rl
            # 策略 4: 100% 基準
            v4 *= rb
            # 策略 2: 50/50 持有 (基準報酬)
            v2_c *= cash_growth; v2_L *= rb
            # 策略 3: 50/50 再平衡 (基準報酬)
            v3_c *= cash_growth; v3_L *= rb
            if (d+1)%252==0: v3_c, v3_L = (v3_c+v3_L)*0.5, (v3_c+v3_L)*0.5
            # 策略 5: 跌深抄底
            v5_B *= rb; v5_L *= rl
            ath = np.maximum(ath, v4/initial_capital)
            dd = (v4/initial_capital)/ath
            trig[dd == 1] = False
            cond = (dd <= 1-drop_threshold) & (~trig)
            if np.any(cond):
                move = v5_B[cond]*transfer_pct
                v5_B[cond]-=move; v5_L[cond]+=move; trig[cond]=True
            # 策略 6: 分批買入
            v6_c *= cash_growth; v6_s *= rb
            if d%dca_interval==0 and d//dca_interval < dca_parts:
                v6_c -= dca_amt; v6_s += dca_amt

        df_res = pd.DataFrame({
            '1. 100% 槓桿': v1,
            '2. 50/50 持有': v2_c + v2_L,
            '3. 50/50 再平衡': v3_c + v3_L,
            '4. 100% 基準 (一次)': v4,
            '5. 跌深抄底': v5_B + v5_L,
            '6. 100% 基準 (分批)': v6_c + v6_s
        })

    st.success("模擬完成！")
    stats = []
    for col in df_res.columns:
        d = df_res[col]
        stats.append({'策略': col, '獲勝率': f"{(d>initial_capital).mean()*100:.1f}%", '中位數': f"{d.median():,.0f}", '悲觀5%': f"{np.percentile(d, 5):,.0f}"})
    st.table(pd.DataFrame(stats))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in df_res.columns: sns.kdeplot(df_res[col], ax=ax, label=col, fill=True, alpha=0.1)
    ax.set_xlim(0, np.percentile(df_res.values, 95)*1.5); ax.legend(); st.pyplot(fig)
