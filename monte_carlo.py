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
st.title("🔬 蒙地卡羅量化回測實驗室 (長線終極對決版)")
st.markdown("基於**蒙地卡羅模擬 (Monte Carlo Simulation)**，深入探討長達 30~50 年的投資週期中，**「第一天單筆全下 (Lump-sum)」**與**「真實每月定期定額 (DCA)」**在不同平行宇宙下的勝率與極端風險。")

with st.expander("📖 蒙地卡羅實驗室使用說明與核心公式 (點擊展開)", expanded=False):
    st.markdown("""
    ### 🎲 關於蒙地卡羅模擬 (Monte Carlo Simulation)
    傳統的回測只能告訴你「過去發生的那唯一一次結果」。而**蒙地卡羅模擬**透過引入隨機性，創造出數千種「未來可能發生的平行宇宙」，幫助我們找出策略在最極端（悲觀 5%）與最理想（樂觀 5%）狀態下的真實表現。

    ### ⚙️ 雙引擎運算模型
    1. **歷史區塊抽樣 (Block Bootstrapping)**
       * 將歷史真實的日報酬率切分為連續的「區塊」，隨機抽取拼湊出未來。完美保留市場的「波動聚集」與「連續崩盤」真實特性。
    2. **GBM 肥尾數學模型**
       * 採用幾何布朗運動，並將傳統的常態分配替換為 Student's t-distribution，更真實地模擬金融市場的「黑天鵝」。
       * 公式：
         $$R_{base, t} = \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z\right) - 1$$

    ---

    ### 🛡️ 槓桿耗損與破產防線
    * **槓桿報酬與內扣耗損**：每日槓桿報酬會根據設定倍數放大，並精準扣除年化內扣與轉倉成本。
      $$R_{lev, t} = R_{base, t} \times L - \frac{Drag}{252}$$
    * **破產防線**：單日最大跌幅極限為 100%，確保資產歸零後不會產生負債複利。
      $$M_{t} = \max(0, 1 + R_{t})$$

    ---

    ### 📈 投資組合策略對決說明 (總成本皆相同)
    系統將針對你設定的「總投入本金」，進行以下 6 種策略的逐日結算對決：
    * **策略 1. 100% 基準 (單筆全下)**：在回測第一天，直接把總本金全部買進基準標的，死抱不放。
    * **策略 2. 100% 基準 (每月定額)**：真實小資族模式。一開始資產為 0，每 21 個交易日 (約一個月) 投入一筆資金，持續到期末。
    * **策略 3. 100% 槓桿 (單筆全下)**：第一天將總本金全部買滿槓桿標的。
    * **策略 4. 100% 槓桿 (每月定額)**：一開始為 0，每個月固定買入槓桿標的。
    * **策略 5. 50/50 再平衡 (單筆)**：第一天投入總本金，50% 留作現金 (1% 利息)，50% 買入基準標的。每年強制重新平衡一次。
    * **策略 6. 跌深抄底 (單筆)**：第一天投入總本金，部分放現金。當基準標的從高點回落達設定比例時，將現金轉入槓桿標的。
    """)

# ==========================================
# 2. 側邊欄：控制面板
# ==========================================
st.sidebar.title("⚙️ 控制面板")
engine = st.sidebar.selectbox("🧠 模擬引擎", ["1. 歷史區塊抽樣 (Block)", "2. 數學模型 (GBM)"])

st.sidebar.header("💰 資金與時間設定")

monthly_input_wan = st.sidebar.number_input("每月定期定額投入 (萬)", min_value=0.1, value=1.0, step=0.1)
sim_years = st.sidebar.slider("⏳ 模擬未來幾年？", min_value=1, max_value=50, value=30)
N = st.sidebar.slider("模擬次數 (平行宇宙)", min_value=1000, max_value=10000, value=5000)

# 自動計算總成本
total_months = sim_years * 12
total_capital_wan = monthly_input_wan * total_months
total_capital = total_capital_wan * 10000

st.sidebar.info(f"💡 換算總投入本金：**{total_capital_wan:.1f} 萬**\n\n(單筆全下策略將在第一天直接投入此金額)")

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
drop_threshold = st.sidebar.slider("策略 6 抄底觸發 (%)", 5, 50, 20) / 100
transfer_pct = st.sidebar.slider("策略 6 轉入比例 (%)", 10, 100, 20) / 100

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
if st.sidebar.button("🚀 開始模擬對決", type="primary", use_container_width=True):
    with st.spinner(f'⚙️ 正在進行 {sim_years} 年的長線平行宇宙運算...'):
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

        # === 策略初始化 ===
        # 單筆全下組 (起手就有所有資金)
        v1_base_lump = np.ones(N) * total_capital
        v3_lev_lump = np.ones(N) * total_capital
        v5_c, v5_b = np.ones(N) * total_capital * 0.5, np.ones(N) * total_capital * 0.5
        v6_b, v6_l = np.ones(N) * total_capital, np.zeros(N)
        
        # 定期定額組 (起手 0 元，每月發薪水投入)
        v2_base_dca = np.zeros(N)
        v4_lev_dca = np.zeros(N)
        monthly_inv = monthly_input_wan * 10000
        
        ath = np.ones(N) 
        trig = np.zeros(N, dtype=bool)

        for d in range(days):
            rb, rl = m_B[d], m_L[d]
            
            # 1. 結算當日資產變化
            v1_base_lump *= rb
            v3_lev_lump *= rl
            v2_base_dca *= rb
            v4_lev_dca *= rl
            
            # 策略 5: 50/50 再平衡
            v5_c *= cash_growth; v5_b *= rb
            if (d+1)%252==0: v5_c, v5_b = (v5_c+v5_b)*0.5, (v5_c+v5_b)*0.5
            
            # 策略 6: 跌深抄底
            v6_b *= rb; v6_l *= rl
            ath = np.maximum(ath, v1_base_lump/total_capital)
            dd = (v1_base_lump/total_capital)/ath
            trig[dd == 1] = False 
            
            cond = (dd <= 1-drop_threshold) & (~trig) 
            if np.any(cond):
                move = v6_b[cond]*transfer_pct
                v6_b[cond]-=move; v6_l[cond]+=move; trig[cond]=True 

            # 2. 注入每月薪水 (每 21 個交易日 = 1 個月)
            if d % 21 == 0 and d // 21 < total_months:
                v2_base_dca += monthly_inv
                v4_lev_dca += monthly_inv

        df_res = pd.DataFrame({
            '1. 100% 基準 (單筆全下)': v1_base_lump,
            '2. 100% 基準 (每月定額)': v2_base_dca,
            '3. 100% 槓桿 (單筆全下)': v3_lev_lump,
            '4. 100% 槓桿 (每月定額)': v4_lev_dca,
            '5. 50/50 再平衡 (單筆)': v5_c + v5_b,
            '6. 跌深抄底 (單筆)': v6_b + v6_l
        })

    # 處理單位變為「萬」
    df_res_van = df_res / 10000

    # ==========================================
    # 5. 產出報表
    # ==========================================
    st.success(f"✅ 成功完成 {sim_years} 年蒙地卡羅模擬！對決總成本皆為：{total_capital_wan:.1f} 萬。")
    
    stats = []
    for col in df_res_van.columns:
        d = df_res_van[col]
        # 勝率的基準線為：期末資產大於「總投入成本」
        win_rate = (d > total_capital_wan).mean() * 100
        stats.append({
            '策略': col,
            '獲勝率 (%)': f"{win_rate:.1f}%",
            '中位數 (萬)': f"{d.median():,.1f}", 
            '悲觀 5% (萬)': f"{np.percentile(d, 5):,.1f}",
            '樂觀 5% (萬)': f"{np.percentile(d, 95):,.1f}" 
        })
    st.dataframe(pd.DataFrame(stats).set_index('策略'), use_container_width=True)
    
    st.subheader(f"📈 {sim_years} 年期終值分佈密度圖 (萬為單位)")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in df_res_van.columns:
        sns.kdeplot(df_res_van[col], ax=ax, label=col, fill=True, alpha=0.15, linewidth=2)
    
    # 標示總成本線
    ax.axvline(total_capital_wan, color='red', linestyle='--', label=f'Total Cost ({total_capital_wan:.0f} 萬)', zorder=10)
    
    title_prefix = "Historical Block Bootstrapping" if "歷史" in engine else "GBM Fat-Tail"
    ax.set_title(f'Monte Carlo Simulation: {sim_years}-Year Asset Distribution ({title_prefix})', fontsize=14)
    ax.set_xlabel('Final Asset Value (萬 TWD)', fontsize=12) 
    ax.set_ylabel('Density', fontsize=12)

    x_max = np.percentile(df_res_van.values, 95) * 1.5
    ax.set_xlim(0, max(x_max, total_capital_wan * 2))

    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
else:
    st.info("👈 請在左側輸入「每月定期定額投入」金額與「模擬年份」，準備見證長線策略對決！")
