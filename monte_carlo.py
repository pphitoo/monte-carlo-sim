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
st.title("🔬 蒙地卡羅量化回測實驗室 (彈性分期實戰版)")
st.markdown("完美結合**「初期單筆資金」**與**「自訂次數與頻率的分期資金」**。透過蒙地卡羅模擬，真實還原你的資金流在各種平行宇宙中的勝率與極端風險。")

with st.expander("📖 實驗室使用說明與 6 大策略邏輯 (點擊展開)", expanded=False):
    st.markdown("""
    ### ⚙️ 資金流運作邏輯
    本系統採用「真實現金流」模式：
    * **初期資金**：在回測的第 1 天就會依據策略投入市場（或放入活存現金池）。
    * **分期資金**：從外部（例如薪水）定期注入。系統會依照你設定的「買入頻率」，每次加碼你設定的金額，直到達到「分批次數」為止。這能最大化資金利用率，避免閒置拖累。

    ---

    ### 📈 6 大實戰策略對決說明
    * **策略 4. 100% 基準 (提早曝險對照組)**：理論神視角。假設你將「初期 + 所有分期」的總成本預支到現在，第一天全數買滿基準大盤。用以對比提早曝險的威力。
    * **策略 6. 100% 基準 (初期 + 分批買入)**：初期資金第一天買大盤，後續分期資金按設定頻率買入。
    * **策略 1. 100% 槓桿 (初期 + 分批買入)**：初期與後續分期資金，全數買入 2 倍槓桿標的。
    * **策略 2. 50/50 持有 (初期 + 分批)**：初期與分期資金皆按 50% 活存 / 50% 大盤分配，買入後不動作。
    * **策略 3. 50/50 再平衡 (初期 + 分批)**：同上，但每 252 個交易日 (約一年) 強制重新平衡回 50/50 比例。
    * **策略 5. 跌深抄底 (半現金預備 + 分批買)**：初期資金 50% 買大盤、50% 放活存預備。分期資金全買大盤。遇大盤自高點回落達設定比例時，將現金部位轉入 2 倍槓桿。
    """)

# ==========================================
# 2. 側邊欄：控制面板
# ==========================================
st.sidebar.title("⚙️ 控制面板")
engine = st.sidebar.selectbox("🧠 模擬引擎", ["1. 歷史區塊抽樣 (Block)", "2. 數學模型 (GBM)"])

st.sidebar.header("💰 彈性資金設定")

# 🌟 獨立的初期與分期資金，並保留次數與頻率
initial_input_wan = st.sidebar.number_input("🏦 初期單筆資金 (萬)", min_value=0.0, value=100.0, step=10.0)
periodic_input_wan = st.sidebar.number_input("📥 每次分期投入資金 (萬)", min_value=0.0, value=10.0, step=1.0)
dca_parts = st.sidebar.slider("分批次數", min_value=1, max_value=240, value=12)
dca_interval = st.sidebar.slider("買入頻率 (交易日)", min_value=5, max_value=60, value=21)
sim_years = st.sidebar.slider("⏳ 模擬未來幾年？", min_value=1, max_value=30, value=10)
N = st.sidebar.slider("模擬次數 (平行宇宙)", min_value=1000, max_value=10000, value=5000)

# 自動計算總成本 = 初期 + (每次分期 * 次數)
total_capital_wan = initial_input_wan + (periodic_input_wan * dca_parts)
initial_cap = initial_input_wan * 10000
periodic_cap = periodic_input_wan * 10000
total_cap = total_capital_wan * 10000

st.sidebar.info(f"💡 總投入成本：**{total_capital_wan:.1f} 萬**\n\n(勝率將以此金額作為是否回本的基準線，對照組策略將在第一天投入此總額)")

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
    with st.spinner(f'⚙️ 正在根據你的彈性資金流進行平行宇宙運算...'):
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

        # === 彈性策略初始化 (第一天的初期資金) ===
        v1_lev = np.ones(N) * initial_cap
        v2_c = np.ones(N) * initial_cap * 0.5; v2_b = np.ones(N) * initial_cap * 0.5
        v3_c = np.ones(N) * initial_cap * 0.5; v3_b = np.ones(N) * initial_cap * 0.5
        
        # 對照組：第一天就將「初期 + 所有分期」全下
        v4_lumpsum = np.ones(N) * total_cap 
        
        v5_c = np.ones(N) * initial_cap * 0.5; v5_b = np.ones(N) * initial_cap * 0.5; v5_l = np.zeros(N)
        v6_base = np.ones(N) * initial_cap
        
        ath = np.ones(N) 
        trig = np.zeros(N, dtype=bool)

        for d in range(days):
            rb, rl = m_B[d], m_L[d]
            
            # 1. 結算當日資產變化
            v1_lev *= rl
            
            v2_c *= cash_growth; v2_b *= rb
            v3_c *= cash_growth; v3_b *= rb
            if (d+1)%252==0: v3_c, v3_b = (v3_c+v3_b)*0.5, (v3_c+v3_b)*0.5
            
            v4_lumpsum *= rb
            
            v5_c *= cash_growth; v5_b *= rb; v5_l *= rl
            ath = np.maximum(ath, v4_lumpsum/total_cap) # 使用純大盤走勢當作高點指標
            dd = (v4_lumpsum/total_cap)/ath
            trig[dd == 1] = False 
            
            cond = (dd <= 1-drop_threshold) & (~trig) 
            if np.any(cond):
                move = v5_c[cond]*transfer_pct
                v5_c[cond]-=move; v5_l[cond]+=move; trig[cond]=True 

            v6_base *= rb

            # 2. 注入外部的分期資金
            # 當滿足「買入頻率」且尚未超過「分批次數」時，執行加碼
            if d % dca_interval == 0 and (d // dca_interval) < dca_parts:
                v1_lev += periodic_cap
                v2_c += periodic_cap * 0.5; v2_b += periodic_cap * 0.5
                v3_c += periodic_cap * 0.5; v3_b += periodic_cap * 0.5
                v5_b += periodic_cap # 抄底策略的分期資金穩健買入大盤
                v6_base += periodic_cap
                # 備註：v4_lumpsum 不加碼，因為它在第一天已經把這些錢全部買進去了

        df_res = pd.DataFrame({
            '1. 100% 槓桿 (初期 + 分批)': v1_lev,
            '2. 50/50 持有 (初期 + 分批)': v2_c + v2_b,
            '3. 50/50 再平衡 (初期 + 分批)': v3_c + v3_b,
            '4. 100% 基準 (總成本首日全下)': v4_lumpsum,
            '5. 跌深抄底 (半現金預備 + 分批)': v5_c + v5_b + v5_l,
            '6. 100% 基準 (初期 + 分批)': v6_base
        })

    # 處理單位變為「萬」
    df_res_van = df_res / 10000

    # ==========================================
    # 5. 產出報表
    # ==========================================
    st.success(f"✅ 成功完成 {sim_years} 年蒙地卡羅模擬！總投入成本為：{total_capital_wan:.1f} 萬。")
    
    stats = []
    for col in df_res_van.columns:
        d = df_res_van[col]
        # 勝率基準線為：總投入成本
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
    st.info("👈 資金設定更靈活了！請在左側輸入「初期資金」與「分期資金/次數」，準備開始運算。")
