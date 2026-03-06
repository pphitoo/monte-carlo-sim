import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import matplotlib.font_manager as fm
import os

# ==========================================
# 0. 字體設定與日期解封
# ==========================================
found_font = False
for f in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    try:
        if 'wqy' in f.lower() or 'noto' in f.lower() or 'jhenghei' in f.lower(): 
            font_prop = fm.FontProperties(fname=f)
            plt.rcParams['font.family'] = font_prop.get_name()
            found_font = True
            break
    except:
        pass

if not found_font and os.name == 'nt':
    plt.rcParams['font.family'] = ['Microsoft JhengHei']

today = datetime.now().date()
min_date = datetime(2000, 1, 1).date()

# ==========================================
# 1. 網頁標題與說明書面板 (Expander)
# ==========================================
st.set_page_config(page_title="蒙地卡羅回測實驗室", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 蒙地卡羅量化回測實驗室 (實戰財務規劃版)")
st.markdown("基於**蒙地卡羅模擬 (Monte Carlo Simulation)**，為真實投資人量身打造。可同時設定**「初期一筆資金」**與**「每月定期定額」**，分析不同資金佈局在長達數十年的平行宇宙中的真實勝率。")

with st.expander("📖 實驗室使用說明與 6 大實戰策略 (點擊展開)", expanded=False):
    st.markdown("""
    ### ⚙️ 雙引擎運算模型
    1. **歷史區塊抽樣 (Block Bootstrapping)**：將歷史真實的日報酬率切分為連續的「區塊」，隨機抽取拼湊出未來。完美保留市場的「波動聚集」與「連續崩盤」真實特性。
    2. **GBM 肥尾數學模型**：採用幾何布朗運動，並將傳統的常態分配替換為 Student's t-distribution，模擬金融市場的「黑天鵝」。

    ---

    ### 📈 6 大實戰策略說明 (皆同時處理初期與每月資金)
    系統將根據你的設定，執行以下 6 種資金佈局對決：
    * **1. 100% 基準 (初期全下 + 每月)**：最標準的作法。第一天將「初期資金」全買大盤，之後每月固定投入「分期資金」。
    * **2. 100% 槓桿 (初期全下 + 每月)**：激進版。初期與每月資金皆全數買入 2 倍槓桿標的。
    * **3. 100% 基準 (初期分12期攤平 + 每月)**：謹慎版。擔心目前是高點，將「初期資金」放活存 (享 1% 利率)，分 12 個月慢慢買進；「分期資金」則正常每月投入。
    * **4. 50/50 再平衡 (初期 + 每月)**：穩健版。初期資金與每月資金皆按 50% 活存 / 50% 大盤分配，每 252 個交易日 (約一年) 強制重新平衡回 50/50 比例。
    * **5. 跌深抄底 (半數現金預備 + 每月買股)**：初期資金 50% 買大盤、50% 放活存預備。每月分期資金全買大盤。遇大盤自高點回落達設定比例時，將現金部位轉入 2 倍槓桿。
    * **6. [對照組] 完美穿越 (總資金首日全下)**：理論上的神之視角。假設你將未來幾十年要存的錢全部「預支」到現在，第一天全數買入大盤。用以對比「提早曝險」的終極威力。
    """)

# ==========================================
# 2. 側邊欄：控制面板
# ==========================================
st.sidebar.title("⚙️ 控制面板")
engine = st.sidebar.selectbox("🧠 模擬引擎", ["1. 歷史區塊抽樣 (Block)", "2. 數學模型 (GBM)"])

st.sidebar.header("💰 資金與時間設定")

# 🌟 拆分初期資金與分期資金
initial_input_wan = st.sidebar.number_input("🏦 初期單筆資金 (萬)", min_value=0.0, value=100.0, step=10.0)
monthly_input_wan = st.sidebar.number_input("📅 每月分期資金 (萬)", min_value=0.0, value=1.0, step=0.1)
sim_years = st.sidebar.slider("⏳ 模擬未來幾年？", min_value=1, max_value=50, value=30)
N = st.sidebar.slider("模擬次數 (平行宇宙)", min_value=1000, max_value=10000, value=5000)

# 自動計算總成本
total_months = sim_years * 12
total_capital_wan = initial_input_wan + (monthly_input_wan * total_months)
initial_cap = initial_input_wan * 10000
monthly_inv = monthly_input_wan * 10000
total_cap = total_capital_wan * 10000

st.sidebar.info(f"💡 換算 30 年總投入成本：**{total_capital_wan:.1f} 萬**\n\n(勝率將以此金額作為是否回本的基準線)")

st.sidebar.header("📅 歷史區間與標的")
if "歷史" in engine:
    ticker = st.sidebar.text_input("輸入代碼 (Yahoo Finance)", value="0050.TW")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("開始", value=datetime(2008, 1, 1).date(), min_value=min_date, max_value=today)
    end_date = col2.date_input("結束", value=today, min_value=min_date, max_value=today)
    block_size = st.sidebar.slider("區塊大小 (歷史連續天數)", 5, 60, 21)
else:
    mu_base = st.sidebar.number_input("基準標的 預期年報酬 (%)", value=9.0) / 100
    sig_base = st.sidebar.number_input("基準標的 年化波動率 (%)", value=16.0) / 100
    df_t = st.sidebar.slider("肥尾效應強度 (t分配)", 2, 30, 3)

st.sidebar.header("🛠️ 槓桿與抄底微調")
lev_mult = st.sidebar.number_input("槓桿倍數", 1.0, 5.0, 2.0, 0.5)
drag_annual = st.sidebar.slider("槓桿標的 年化耗損 (%)", 0.0, 10.0, 1.5) / 100
drop_threshold = st.sidebar.slider("策略 5 抄底觸發 (%)", 5, 50, 20) / 100
transfer_pct = st.sidebar.slider("策略 5 轉入比例 (%)", 10, 100, 20) / 100

# ==========================================
# 3. 資料下載快取
# ==========================================
@st.cache_data(show_spinner=False, ttl=600)
def get_hist_data(tkr, start, end):
    try:
        data = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'].iloc[:, 0].pct_change().dropna().values.flatten()
        return data['Close'].pct_change().dropna().values.flatten()
    except Exception as e:
        return None

# ==========================================
# 4. 核心運算區塊
# ==========================================
if st.sidebar.button("🚀 開始實戰模擬", type="primary", use_container_width=True):
    with st.spinner(f'⚙️ 正在根據你的資金藍圖進行 {sim_years} 年的平行宇宙運算...'):
        days = sim_years * 252
        dt = 1/252
        cash_growth = np.exp(0.01 * dt)
        
        sim_ret_base = np.zeros((days, N))
        
        if "歷史" in engine:
            rets = get_hist_data(ticker, start_date, end_date)
            if rets is None or len(rets) < block_size:
                st.error("❌ 無法載入歷史資料。請檢查日期或代碼。")
                st.stop()

            indices = np.random.randint(0, len(rets)-block_size, (int(np.ceil(days/block_size)), N))
            for b in range(indices.shape[0]):
                starts = indices[b,:]
                for i in range(block_size):
                    d_idx = b * block_size + i
                    if d_idx < days: sim_ret_base[d_idx, :] = rets[starts + i]
        else:
            Z = np.clip(np.random.standard_t(df_t, (days, N)) * np.sqrt(1/3), -15, 15)
            log_ret_base = (mu_base - 0.5 * sig_base**2) * dt + sig_base * np.sqrt(dt) * Z
            sim_ret_base = np.exp(log_ret_base) - 1
            
        sim_ret_lev = (sim_ret_base * lev_mult) - (drag_annual/252)
        
        # 破產防線
        m_B, m_L = np.maximum(0, 1+sim_ret_base), np.maximum(0, 1+sim_ret_lev)

        # === 實戰化策略初始化 ===
        v1_base = np.ones(N) * initial_cap
        v2_lev = np.ones(N) * initial_cap
        
        # 策略3: 初期資金放現金，準備分 12 個月買進
        v3_cash = np.ones(N) * initial_cap
        v3_base = np.zeros(N)
        
        # 策略4: 50/50 再平衡
        v4_cash = np.ones(N) * initial_cap * 0.5
        v4_base = np.ones(N) * initial_cap * 0.5
        
        # 策略5: 跌深抄底
        v5_cash = np.ones(N) * initial_cap * 0.5
        v5_base = np.ones(N) * initial_cap * 0.5
        v5_lev = np.zeros(N)
        
        # 策略6: 理論對照組 (總成本首日全下)
        v6_theory = np.ones(N) * total_cap
        
        ath = np.ones(N) 
        trig = np.zeros(N, dtype=bool)

        for d in range(days):
            rb, rl = m_B[d], m_L[d]
            
            # 1. 結算當日資產變化
            v1_base *= rb
            v2_lev *= rl
            
            v3
