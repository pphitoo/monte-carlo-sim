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
st.set_page_config(page_title="量化回測實驗室", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 終極量化回測實驗室 (雙引擎架構)")
st.markdown("自由切換「歷史區塊抽樣」與「GBM 肥尾數學模型」，分析「一次投入」與「分批買入」在不同環境下的勝率。數據已自動換算為**「萬 (10k TWD)」**單位。")

with st.expander("📖 實驗室使用說明與核心運算公式 (點擊展開)", expanded=False):
    st.markdown("""
    ### ⚙️ 雙引擎運算模型
    本系統提供兩種平行宇宙生成方式，幫助你進行投資策略的壓力測試：
    
    1. **歷史區塊抽樣 (Block Bootstrapping)**
       * **邏輯**：將歷史真實的日報酬率切分為連續的「區塊」，隨機抽取拼湊出未來。
       * **優勢**：完美保留市場的「波動聚集」與「連續崩盤」真實特性。
    2. **GBM 肥尾數學模型**
       * **邏輯**：採用幾何布朗運動 (Geometric Brownian Motion)，並將傳統的常態分配替換為 Student's t-distribution，以更真實地模擬金融市場的「黑天鵝」極端事件。
       * **公式**：
         $$R_{base, t} = \exp\\left(\\left(\\mu - \\frac{\\sigma^2}{2}\\right)\\Delta t + \\sigma \\sqrt{\\Delta t} Z\\right) - 1$$
         *(其中 $Z \\sim t(df)$，並限制極端值以防止數學溢位)*

    ---

    ### 🛡️ 槓桿耗損與破產防線
    在模擬槓桿 ETF 時，系統考量了真實的耗損與物理極限：
    
    * **槓桿報酬與內扣耗損 (Volatility Drag & Fees)**：
      每日槓桿報酬會根據設定倍數放大，並精準扣除年化內扣與轉倉成本。
      $$R_{lev, t} = R_{base, t} \\times L - \\frac{Drag}{252}$$
    * **破產防線 (Zero-Bound Protection)**：
      單日最大跌幅極限為 $100\\%$，確保資產歸零後不會產生負債複利。
      $$M_{t} = \\max(0, 1 + R_{t})$$

    ---

    ### 📈 投資組合策略說明
    系統逐日結算以下 6 種策略，並統整於最終報表：
    * **策略 1 / 4**：期初將資金 $100\\%$ 投入槓桿標的或基準標的。
    * **策略 2 (50/50 持有)**：$50\\%$ 現金 (享 $1\\%$ 活存利率) + $50\\%$ 基準標的，**買入後不動作**。
    * **策略 3 (50/50 再平衡)**：同上，但**每 252 個交易日 (約一年) 強制重新平衡**回 50/50 比例。
    * **策略 5 (跌深抄底)**：每日追蹤基準標的之歷史最高點 (ATH)，當回落比例大於設定的 $Drop_{threshold}$ 時，觸發單次資金轉移，將現金池比例轉入槓桿標的。
    * **策略 6 (分批買入/DCA)**：將資金等分為 $N$ 份，每隔固定的交易日，將一份資金從現金池轉入市場，以驗證分批攤平的防禦力。
    """)

# ==========================================
# 2. 側邊欄：控制面板
# ==========================================
st.sidebar.title("⚙️ 控制面板")
engine = st.sidebar.selectbox("🧠 模擬引擎", ["1. 歷史區塊抽樣 (Block)", "2. 數學模型 (GBM)"])

st.sidebar.header("💰 基本資金與時間")
# 介面顯示百萬，背景還原為真實金額
input_mb = st.sidebar.number_input("初始資金 (百萬)", min_value=0.1, value=2.0, step=0.1)
initial_capital = input_mb * 1000000

sim_years = st.sidebar.slider("⏳ 模擬未來幾年？", min_value=1, max_value=30, value=10)
N = st.sidebar.slider("模擬次數", min_value=1000, max_value=10000, value=5000)

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
if st.sidebar.button("🚀 開始模擬運算", type="primary", use_container_width=True):
    with st.spinner('⚙️ 正在建構平行宇宙與運算策略...'):
        days = sim_years * 252
        dt = 1/252
        cash_growth = np.exp(0.01 * dt)
        
        sim_ret_base = np.zeros((days, N))
        
        if "歷史" in engine:
            rets = get_hist_data(ticker, start_date, end_date)
            if rets is None or len(rets) < block_size:
                st.error("❌ 無法載入歷史資料。可能是日期選取錯誤或 Yahoo 連線中斷。")
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

        v1, v2_c, v2_L, v3_c, v3_L, v4, v5_B, v5_L, v6_c, v6_s = [np.ones(N)*initial_capital for _ in range(10)]
        v2_c, v2_L, v3_c, v3_L = v2_c*0.5, v2_L*0.5, v3_c*0.5, v3_L*0.5
        v5_L, v6_s = np.zeros(N), np.zeros(N)
        
        ath = np.ones(N) 
        trig = np.zeros(N, dtype=bool)
        dca_amt = initial_capital / dca_parts

        for d in range(days):
            rb, rl = m_B[d], m_L[d]
            v1 *= rl
            v4 *= rb
            v2_c *= cash_growth; v2_L *= rb
            v3_c *= cash_growth; v3_L *= rb
            if (d+1)%252==0: v3_c, v3_L = (v3_c+v3_L)*0.5, (v3_c+v3_L)*0.5
            
            v5_B *= rb; v5_L *= rl
            ath = np.maximum(ath, v4/initial_capital)
            dd = (v4/initial_capital)/ath
            trig[dd == 1] = False 
            
            cond = (dd <= 1-drop_threshold) & (~trig) 
            if np.any(cond):
                move = v5_B[cond]*transfer_pct
                v5_B[cond]-=move; v5_L[cond]+=move; trig[cond]=True 
                
            v6_c *= cash_growth; v6_s *= rb
            if d%dca_interval==0 and d//dca_interval < dca_parts:
                v6_c -= dca_amt; v6_s += dca_amt

        df_res = pd.DataFrame({
            '1. 100% 槓桿': v1,
            '2. 50/50 持有': v2_c + v2_L,
            '3. 50/50 再平衡': v3_c + v3_L,
            '4. 100% 基準 (一次投入)': v4,
            '5. 跌深抄底策略': v5_B + v5_L,
            '6. 100% 基準 (分批買入)': v6_c + v6_s
        })

    # 處理單位變為「萬」
    df_res_van = df_res / 10000
    initial_capital_van = initial_capital / 10000

    # ==========================================
    # 5. 產出報表
    # ==========================================
    st.success("成功完成 5,000 次平行宇宙模擬！數據已換算為**「萬」**為單位。")
    
    stats = []
    for col in df_res_van.columns:
        d = df_res_van[col]
        win_rate = (d > initial_capital_van).mean() * 100
        stats.append({
            '策略': col,
            '獲勝率 (%)': f"{win_rate:.1f}%",
            '中位數 (萬)': f"{d.median():,.1f}", 
            '悲觀 5% (萬)': f"{np.percentile(d, 5):,.1f}",
            '樂觀 5% (萬)': f"{np.percentile(d, 95):,.1f}" 
        })
    st.dataframe(pd.DataFrame(stats).set_index('策略'), use_container_width=True)
    
    st.subheader(f"📈 終值分佈密度圖 ({sim_years} 年後 / 萬為單位)")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in df_res_van.columns:
        sns.kdeplot(df_res_van[col], ax=ax, label=col, fill=True, alpha=0.15, linewidth=2)
    
    ax.axvline(initial_capital_van, color='red', linestyle='--', label='Initial Capital', zorder=10)
    
    title_prefix = "Historical Block" if "歷史" in engine else "GBM Fat-Tail"
    ax.set_title(f'{sim_years}-Year Final Asset Distribution ({title_prefix} / 10k TWD)', fontsize=14)
    ax.set_xlabel('Final Asset Value (萬 TWD)', fontsize=12) 
    ax.set_ylabel('Density', fontsize=12)

    x_max = np.percentile(df_res_van.values, 95) * 1.5
    ax.set_xlim(0, max(x_max, initial_capital_van * 2))

    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
else:
    st.info("👈 請在左側設定參數，調整完畢點擊「開始模擬運算」。")
