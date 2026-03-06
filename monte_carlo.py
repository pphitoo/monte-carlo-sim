import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta

# ==========================================
# 0. 網頁介面基本設定
# ==========================================
st.set_page_config(page_title="量化回測實驗室", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 終極量化回測實驗室 (雙引擎 + 買入策略分析)")
st.markdown("比較「一次買入」與「分批買入」在不同引擎下的績效差異。")

# 🌟 修正點：必須定義今日日期，否則下方 max_date 會報錯
today = datetime.now().date()

# ==========================================
# 1. 側邊欄：控制中心
# ==========================================
st.sidebar.title("⚙️ 控制面板")

# 模組 1: 核心引擎選擇
engine = st.sidebar.selectbox(
    "🧠 選擇模擬引擎", ["1. 歷史區塊抽樣 (Block Bootstrapping)", "2. 數學模型 (GBM 肥尾效應)"])
st.sidebar.divider()

# 模組 2: 基本參數
st.sidebar.header("💰 基本資金與時間")
initial_capital = st.sidebar.number_input("初始資金 (元)", min_value=100000, value=2000000, step=100000)
sim_years = st.sidebar.slider("⏳ 模擬未來幾年？", min_value=1, max_value=30, value=10)
N = st.sidebar.slider("模擬次數 (平行宇宙)", min_value=1000, max_value=10000, value=5000, step=1000)

# 🌟 買入策略參數
st.sidebar.header("📅 買入策略設定")
dca_parts = st.sidebar.slider("分批買入次數 (次)", min_value=2, max_value=48, value=12)
dca_interval = st.sidebar.slider("買入頻率 (隔多少交易日買一次)", min_value=5, max_value=60, value=21, help="21天約為一個月")
st.sidebar.divider()

# 模組 3: 市場環境
if "歷史" in engine:
    ticker = st.sidebar.text_input("輸入歷史標的代碼", value="0050.TW")
    col1, col2 = st.sidebar.columns(2)
    min_date = datetime(2000, 1, 1).date()
    max_date = today  # 現在定義好了

    start_date = col1.date_input("開始日期", value=datetime(2008, 1, 1).date(), min_value=min_date, max_value=max_date)
    end_date = col2.date_input("結束日期", value=max_date, min_value=min_date, max_value=max_date)
    block_size = st.sidebar.slider("區塊大小", min_value=5, max_value=60, value=21)
else:
    mu_base = st.sidebar.number_input("基準標的 預期年報酬 (%)", value=9.0) / 100
    sig_base = st.sidebar.number_input("基準標的 年化波動 (%)", value=16.0) / 100
    df_t = st.sidebar.slider("肥尾效應強度 (df)", min_value=2, max_value=30, value=3)
st.sidebar.divider()

# 模組 4: 槓桿與策略
st.sidebar.header("🛠️ 槓桿與策略微調")
lev_mult = st.sidebar.number_input("槓桿倍數", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
drag_annual = st.sidebar.slider("槓桿標的 額外耗損 (%)", min_value=0.0, max_value=10.0, value=1.5, step=0.1) / 100

st.sidebar.caption("策略 5：跌深抄底設定")
drop_threshold = st.sidebar.slider("回檔觸發比例 (%)", min_value=5, max_value=50, value=20) / 100
transfer_pct = st.sidebar.slider("轉入比例 (%)", min_value=10, max_value=100, value=20) / 100

# ==========================================
# 核心功能與資料處理
# ==========================================
@st.cache_data(show_spinner=False, ttl=3600)
def get_hist_data(tkr, start, end):
    try:
        data = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'].iloc[:, 0].pct_change().dropna().values.flatten()
        return data['Close'].pct_change().dropna().values.flatten()
    except: return None

if st.sidebar.button("🚀 開始模擬運算", type="primary", use_container_width=True):
    with st.spinner('⚙️ 運算中...'):
        days_per_year = 252
        total_days = sim_years * days_per_year
        dt = 1 / days_per_year
        cash_growth = np.exp(0.01 * dt)
        
        sim_ret_base = np.zeros((total_days, N))
        sim_ret_lev = np.zeros((total_days, N))
        daily_drag = drag_annual / days_per_year

        if "歷史" in engine:
            hist_ret = get_hist_data(ticker, start_date, end_date)
            if hist_ret is None or len(hist_ret) < block_size:
                st.error("❌ 無法載入歷史資料。"); st.stop()
            hist_ret_lev = (hist_ret * lev_mult) - daily_drag
            blocks_per_path = int(np.ceil(total_days / block_size))
            max_start_idx = len(hist_ret) - block_size
            random_starts = np.random.randint(0, max_start_idx, size=(blocks_per_path, N))
            for b in range(blocks_per_path):
                starts = random_starts[b, :]
                for i in range(block_size):
                    d_idx = b * block_size + i
                    if d_idx < total_days:
                        sim_ret_base[d_idx, :] = hist_ret[starts + i]
                        sim_ret_lev[d_idx, :] = hist_ret_lev[starts + i]
        else:
            Z = np.clip(np.random.standard_t(df=df_t, size=(total_days, N)) * np.sqrt(1/3), -25, 25)
            log_ret_base = (mu_base - 0.5 * sig_base**2) * dt + sig_base * np.sqrt(dt) * Z
            log_ret_lev = (mu_base*lev_mult - 0.5*(sig_base*lev_mult)**2) * dt + (sig_base*lev_mult)*np.sqrt(dt)*Z - daily_drag
            sim_ret_base, sim_ret_lev = np.exp(log_ret_base) - 1, np.exp(log_ret_lev) - 1

        # 🌟 破產防線
        sim_mult_base = np.maximum(0, 1 + sim_ret_base)
        sim_mult_lev = np.maximum(0, 1 + sim_ret_lev)
        price_base = np.vstack([np.ones(N), np.cumprod(sim_mult_base, axis=0)])

        # 初始化策略
        V1 = np.ones(N) * initial_capital
        V4_LS = np.ones(N) * initial_capital
        V6_DCA_cash = np.ones(N) * initial_capital
        V6_DCA_stock = np.zeros(N)
        V5_B, V5_L = np.ones(N) * initial_capital, np.zeros(N)
        
        price_ATH_base = np.ones(N)
        triggered = np.zeros(N, dtype=bool)
        dca_amount_per_step = initial_capital / dca_parts

        for d in range(1, total_days + 1):
            r_B, r_L = sim_mult_base[d-1], sim_mult_lev[d-1]
            V4_LS *= r_B
            V6_DCA_cash *= cash_growth; V6_DCA_stock *= r_B
            if (d - 1) % dca_interval == 0 and (d - 1) // dca_interval < dca_parts:
                V6_DCA_cash -= dca_amount_per_step
                V6_DCA_stock += dca_amount_per_step
            V1 *= r_L
            V5_B *= r_B; V5_L *= r_L
            price_ATH_base = np.maximum(price_ATH_base, price_base[d])
            with np.errstate(divide='ignore', invalid='ignore'):
                dd = np.where(price_ATH_base > 0, price_base[d] / price_ATH_base, 0)
            triggered[dd == 1.0] = False
            cond = (dd <= (1.0 - drop_threshold)) & (~triggered)
            if np.any(cond):
                trans = V5_B[cond] * transfer_pct
                V5_B[cond] -= trans; V5_L[cond] += trans; triggered[cond] = True

        df_res = pd.DataFrame({
            '1. 100% 槓桿 (一次)': V1,
            '4. 100% 基準 (一次)': V4_LS,
            '6. 100% 基準 (分批)': V6_DCA_cash + V6_DCA_stock,
            '5. 跌深抄底策略': V5_B + V5_L
        })

    # ==========================================
    # 產出報表
    # ==========================================
    st.success(f"✅ 模擬完成！分批買入設定：分為 {dca_parts} 次，每 {dca_interval} 天投入一次。")
    stats = []
    for col in df_res.columns:
        data = df_res[col]
        stats.append({'投資策略': col, '獲勝機率 (%)': f"{(data > initial_capital).mean()*100:.2f}%", 
                     '平均終值': f"{data.mean():,.0f}", '中位數': f"{data.median():,.0f}", '悲觀 5%': f"{np.percentile(data, 5):,.0f}"})
    st.subheader(f"📊 {sim_years} 年後績效統計")
    st.dataframe(pd.DataFrame(stats), use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in df_res.columns:
        sns.kdeplot(df_res[col], ax=ax, label=col, fill=True, alpha=0.1)
    ax.axvline(initial_capital, color='red', linestyle='--', label='Initial Capital')
    ax.set_title(f'Lump-sum vs Staggered Entry ({sim_years} Years)')
    ax.set_xlim(0, np.percentile(df_res.values, 95) * 1.5)
    ax.legend(); st.pyplot(fig)
else:
    st.info("👈 請在左側設定參數，點擊「開始模擬運算」。")
