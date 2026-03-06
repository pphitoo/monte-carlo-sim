import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime

# ==========================================
# 🌟 修正點 1：處理圖表亂碼問題 (設定中文支援)
# ==========================================
import matplotlib.font_manager as fm
import os

# 嘗試自動尋找支援中文的字體路徑
found_font = False
for f in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    try:
        # 雲端 Linux 常見的中文支援字體 (WenQuanYi 或 Noto)
        if 'wqy' in f.lower() or 'noto' in f.lower(): 
            font_prop = fm.FontProperties(fname=f)
            plt.rcParams['font.family'] = font_prop.get_name()
            found_font = True
            break
    except:
        pass

# 如果自動尋找失敗，在本地 Windows 通常需要這個
if not found_font and os.name == 'nt':
    plt.rcParams['font.family'] = ['Microsoft JhengHei'] # 微軟正黑體

# ==========================================
# 0. 基本設定與日期解封
# ==========================================
st.set_page_config(page_title="量化回測實驗室", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 終極量化回測實驗室 (專業數據呈現版)")
st.markdown("自由切換「歷史區塊抽樣」與「GBM 肥尾數學模型」，分析「一次投入」與「分批買入」在不同環境下的勝率。數據已自動換算為**「萬 (10k TWD)」**單位。")

# 🌟 解決 2026 日期選取限制
today = datetime.now().date()
min_date = datetime(2000, 1, 1).date()

# ==========================================
# 1. 側邊欄：控制面板
# ==========================================
st.sidebar.title("⚙️ 控制面板")
engine = st.sidebar.selectbox("🧠 模擬引擎", ["1. 歷史區塊抽樣 (Block)", "2. 數學模型 (GBM)"])

col1_cap, col2_cap = st.sidebar.columns(2)
# 這裡維持原生單位輸入，內部計算會處理
initial_capital = col1_cap.number_input("初始資金 (元)", min_value=100000, value=2000000, step=100000)
# ⚠️ 修正上限為 10000，保護雲端伺服器記憶體不崩潰
N = col2_cap.slider("模擬次數", min_value=1000, max_value=10000, value=5000)

sim_years = st.sidebar.slider("⏳ 模擬未來幾年？", min_value=1, max_value=30, value=10)

st.sidebar.header("📅 買入策略設定")
dca_parts = st.sidebar.slider("分批次數", min_value=2, max_value=48, value=12)
dca_interval = st.sidebar.slider("買入頻率 (交易日)", min_value=5, max_value=60, value=21)

if "歷史" in engine:
    ticker = st.sidebar.text_input("輸入代碼 (Yahoo Finance)", value="0050.TW")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("開始", value=datetime(2008, 1, 1).date(), min_value=min_date, max_value=today)
    end_date = col2.date_input("結束", value=today, min_value=min_date, max_value=today)
    block_size = st.sidebar.slider("區塊大小 (歷史連續天數)", 5, 60, 21)
else:
    mu_base = st.sidebar.number_input("基準標的 預期年化報酬率 (%)", value=9.0) / 100
    sig_base = st.sidebar.number_input("基準標的 年化波動率 (%)", value=16.0) / 100
    df_t = st.sidebar.slider("肥尾效應強度 (t分配自由度)", 2, 30, 3)

lev_mult = st.sidebar.number_input("槓桿倍數", 1.0, 5.0, 2.0, 0.5)
drag_annual = st.sidebar.slider("槓桿標的 額外年化耗損 (%)", 0.0, 10.0, 1.5) / 100
drop_threshold = st.sidebar.slider("抄底觸發比例 (%)", 5, 50, 20) / 100
transfer_pct = st.sidebar.slider("轉入槓桿比例 (%)", 10, 100, 20) / 100

# ==========================================
# 資料下載快取
# ==========================================
@st.cache_data(show_spinner=False, ttl=600)
def get_hist_data(tkr, start, end):
    try:
        # 🌟 auto_adjust=True 讓 Yahoo 直接給我們還原權息的資料，徹底排除除權息或拆分造成的假暴跌
        data = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'].iloc[:, 0].pct_change().dropna().values.flatten()
        return data['Close'].pct_change().dropna().values.flatten()
    except Exception as e:
        return None

# ==========================================
# 核心運算區塊
# ==========================================
if st.sidebar.button("🚀 開始模擬運算", type="primary", use_container_width=True):
    with st.spinner('⚙️ 正在建構平行宇宙與運算策略...'):
        days = sim_years * 252
        dt = 1/252
        cash_growth = np.exp(0.01 * dt)
        
        # 準備回報率矩陣
        sim_ret_base = np.zeros((days, N))
        
        # 根據引擎產生走勢
        if "歷史" in engine:
            rets = get_hist_data(ticker, start_date, end_date)
            if rets is None or len(rets) < block_size:
                st.error("❌ 無法載入歷史資料。可能是日期選取錯誤或 Yahoo 連線中斷。")
                st.stop()

            # 歷史抽樣
            indices = np.random.randint(0, len(rets)-block_size, (int(np.ceil(days/block_size)), N))
            for b in range(indices.shape[0]):
                starts = indices[b,:]
                for i in range(block_size):
                    d_idx = b * block_size + i
                    if d_idx < days: sim_ret_base[d_idx, :] = rets[starts + i]
        else:
            # GBM 引擎 (使用肥尾 t 分配，並限制極端黑天鵝)
            Z = np.clip(np.random.standard_t(df_t, (days, N)) * np.sqrt(1/3), -15, 15)
            # 使用對數報酬近似
            log_ret_base = (mu_base - 0.5 * sig_base**2) * dt + sig_base * np.sqrt(dt) * Z
            # 轉換回真實每日報酬率
            sim_ret_base = np.exp(log_ret_base) - 1
            
        # 計算槓桿標的回報 (考慮額外內扣費用)
        sim_ret_lev = (sim_ret_base * lev_mult) - (drag_annual/252)
        
        # 🌟 修正點 2：加入「破產防線」，資產乘數不可小於 0 (資產歸零即停止下跌，不產生負債)
        m_B, m_L = np.maximum(0, 1+sim_ret_base), np.maximum(0, 1+sim_ret_lev)

        # 策略初始資金初始化 (全部歸一化到 initial_capital)
        v1, v2_c, v2_L, v3_c, v3_L, v4, v5_B, v5_L, v6_c, v6_s = [np.ones(N)*initial_capital for _ in range(10)]
        v2_c, v2_L, v3_c, v3_L = v2_c*0.5, v2_L*0.5, v3_c*0.5, v3_L*0.5
        v5_L, v6_s = np.zeros(N), np.zeros(N)
        
        ath = np.ones(N) # 基準標點歷史高點自 normalized 值開始
        trig = np.zeros(N, dtype=bool)
        dca_amt = initial_capital / dca_parts

        # 逐日結算迴圈
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
            
            # 用 V4 的歸一化價格計算抄底
            ath = np.maximum(ath, v4/initial_capital)
            dd = (v4/initial_capital)/ath
            trig[dd == 1] = False # 回到高點時重置觸發器
            
            cond = (dd <= 1-drop_threshold) & (~trig) # 下跌超過門檻且未曾觸發過
            if np.any(cond):
                move = v5_B[cond]*transfer_pct
                v5_B[cond]-=move; v5_L[cond]+=move; trig[cond]=True # 轉入槓桿，並標記為已觸發
            # 策略 6: 分批買入基準標的
            v6_c *= cash_growth; v6_s *= rb
            if d%dca_interval==0 and d//dca_interval < dca_parts:
                v6_c -= dca_amt; v6_s += dca_amt

        # 彙整所有 6 種策略
        df_res = pd.DataFrame({
            '1. 100% 槓桿': v1,
            '2. 50/50 持有': v2_c + v2_L,
            '3. 50/50 再平衡': v3_c + v3_L,
            '4. 100% 基準 (一次投入)': v4,
            '5. 跌深抄底策略': v5_B + v5_L,
            '6. 100% 基準 (分批買入)': v6_c + v6_s
        })

    # ==========================================
    # 🌟 修正點 2：處理單位 (除以 10000 變成 萬)
    # ==========================================
    # 我們直接把整個 DataFrame 的數據除以 10000，讓統計數據的中位數、悲觀/樂觀 5% 就會是萬為單位
    df_res_van = df_res / 10000
    # 把 initial_capital 也換算成萬，用於圖表 vline
    initial_capital_van = initial_capital / 10000

    # ==========================================
    # 產出報表
    # ==========================================
    st.success("成功完成 5,000 次平行宇宙模擬！數據已換算為**「萬」**為單位。")
    
    # 統計表
    stats = []
    for col in df_res_van.columns:
        d = df_res_van[col]
        
        # 判斷勝率時，也要用換算後的萬單位判斷是否大於初始資金
        win_rate = (d > initial_capital_van).mean() * 100
        
        # ==========================================
        # 🌟 修正點 3：統計表格內容 (格式化為萬，並加入樂觀 5%)
        # ==========================================
        stats.append({
            '策略': col,
            '獲勝率 (%)': f"{win_rate:.1f}%",
            '中位數 (萬)': f"{d.median():,.1f}", 
            '悲觀 5% (萬)': f"{np.percentile(d, 5):,.1f}", # 悲觀 (左尾)
            '樂觀 5% (萬)': f"{np.percentile(d, 95):,.1f}" # 加入這行：樂觀 (右尾)
        })
    st.dataframe(pd.DataFrame(stats).set_index('策略'), use_container_width=True)
    
    # 圖表
    st.subheader(f"📈 終值分佈密度圖 ({sim_years} 年後 / 萬為單位)")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 🌟 用換算成萬單位的 DataFrame 繪圖
    for col in df_res_van.columns:
        sns.kdeplot(df_res_van[col], ax=ax, label=col, fill=True, alpha=0.15, linewidth=2)
    
    ax.axvline(initial_capital_van, color='red', linestyle='--', label='Initial Capital', zorder=10)
    
    # ==========================================
    # 🌟 修正點 4：圖表標籤設定 (萬為單位)
    # ==========================================
    title_prefix = "Historical Block" if "歷史" in engine else "GBM Fat-Tail"
    ax.set_title(f'{sim_years}-Year Final Asset Distribution ({title_prefix} / 10k TWD)', fontsize=14)
    ax.set_xlabel('Final Asset Value (萬 TWD)', fontsize=12) # 🌟 修正 X 軸標籤
    ax.set_ylabel('Density', fontsize=12)

    # 動態調整 X 軸範圍 (也要用換算後的萬單位)
    x_max = np.percentile(df_res_van.values, 95) * 1.5
    ax.set_xlim(0, max(x_max, initial_capital_van * 2))

    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
else:
    st.info("👈 請在左側設定參數，日期 (YYYY/MM/DD) 必須輸入完整，調整完畢點擊「開始模擬運算」。")
