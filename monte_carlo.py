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
st.title("🔬 蒙地卡羅量化回測實驗室 (實戰對決版)")
st.markdown("完美結合**「初期單筆資金」**與**「自訂次數的分期資金」**。透過蒙地卡羅模擬，真實還原你的資金流在各種平行宇宙中的勝率與極端風險。")

with st.expander("📖 實驗室說明與 6 大情境策略 (點擊展開)", expanded=False):
    st.markdown("""
    ### ⚙️ 資金流運作邏輯
    * **初期資金**：在回測的第 1 天就會依據策略投入市場（或放入活存現金池）。
    * **分期資金**：從外部（例如發薪水）定期注入。系統會依照你設定的「買入頻率 (月)」與「次數」慢慢加碼，直到扣完為止。

    ---

    ### 📈 6 大策略人設與作法 (超白話文)
    * **策略 1. 一般散戶 (100% 基準)**：手邊的錢第一天全買大盤，之後的閒錢也按時買大盤。最標準的投資法。
    * **策略 2. 激進賭徒 (100% 槓桿)**：手邊的錢跟每個月的閒錢，全部拿去買 2 倍槓桿。追求極致報酬，但也承擔極致風險。
    * **策略 3. 保守定存 (50/50 持有)**：手邊的錢跟每個月的閒錢，都只拿一半買大盤，另一半放銀行定存 (1% 利率) 絕對不碰。
    * **策略 4. 紀律經理 (50大盤/50槓桿 再平衡)**：手邊的錢跟閒錢，一半買 1 倍大盤，一半買 2 倍槓桿。每年底會強制「重新平衡」，把賺比較多的賣掉，補給另一個，維持 1:1 的完美比例。
    * **策略 5. 老謀深算 (跌深抄底)**：手邊的錢留一半放銀行等崩盤，等大盤跌到設定的滿足點，就拿存款去抄底「2 倍槓桿」；分期閒錢則安分買大盤。
    * **策略 6. 時空旅人 (神明對照組)**：向神明借未來所有的錢，第一天直接「歐印 (All-in)」大盤。用來測試「資金越早進場越好」的終極對照組。
    """)

# ==========================================
# 2. 側邊欄：控制面板
# ==========================================
st.sidebar.title("⚙️ 控制面板")
engine = st.sidebar.selectbox("🧠 模擬引擎", ["1. 歷史區塊抽樣 (Block)", "2. 數學模型 (GBM)"])

st.sidebar.header("💰 彈性資金設定")

initial_input_wan = st.sidebar.number_input("🏦 初期單筆資金 (萬)", min_value=0.0, value=100.0, step=10.0)
periodic_input_wan = st.sidebar.number_input("📥 每次分期投入資金 (萬)", min_value=0.0, value=10.0, step=1.0)
dca_parts = st.sidebar.slider("分批次數", min_value=1, max_value=360, value=12)

# 🌟 這裡改成以「月」為單位的滑桿
dca_interval_months = st.sidebar.slider("買入頻率 (月)", min_value=1, max_value=12, value=1)
# 🌟 背景自動將月份換算為交易日 (1個月約 = 21個交易日)
dca_interval = dca_interval_months * 21 

sim_years = st.sidebar.slider("⏳ 模擬未來幾年？", min_value=1, max_value=50, value=10)
N = st.sidebar.slider("模擬次數 (平行宇宙)", min_value=1000, max_value=10000, value=5000)

# 自動計算總成本
total_capital_wan = initial_input_wan + (periodic_input_wan * dca_parts)
initial_cap = initial_input_wan * 10000
periodic_cap = periodic_input_wan * 10000
total_cap = total_capital_wan * 10000

st.sidebar.info(f"💡 總投入成本：**{total_capital_wan:.1f} 萬**\n\n(勝率將以此金額作為基準線，時空旅人策略將在首日投入此總額)")

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
        
        m_B, m_L = np.maximum(0, 1+sim_ret_base), np.maximum(0, 1+sim_ret_lev)

        # === 策略變數初始化 (對齊 1 到 6 的人設順序) ===
        v1_base = np.ones(N) * initial_cap
        v2_lev = np.ones(N) * initial_cap
        v3_c = np.ones(N) * initial_cap * 0.5; v3_b = np.ones(N) * initial_cap * 0.5
        v4_b = np.ones(N) * initial_cap * 0.5; v4_l = np.ones(N) * initial_cap * 0.5
        v5_c = np.ones(N) * initial_cap * 0.5; v5_b = np.ones(N) * initial_cap * 0.5; v5_l = np.zeros(N)
        v6_lumpsum = np.ones(N) * total_cap 
        
        ath = np.ones(N) 
        trig = np.zeros(N, dtype=bool)

        for d in range(days):
            rb, rl = m_B[d], m_L[d]
            
            # 1. 結算當日變化
            v1_base *= rb
            v2_lev *= rl
            
            v3_c *= cash_growth; v3_b *= rb
            
            v4_b *= rb; v4_l *= rl
            if (d+1)%252==0: v4_b, v4_l = (v4_b+v4_l)*0.5, (v4_b+v4_l)*0.5
            
            v5_c *= cash_growth; v5_b *= rb; v5_l *= rl
            ath = np.maximum(ath, v6_lumpsum/total_cap) 
            dd = (v6_lumpsum/total_cap)/ath
            trig[dd == 1] = False 
            
            cond = (dd <= 1-drop_threshold) & (~trig) 
            if np.any(cond):
                move = v5_c[cond]*transfer_pct
                v5_c[cond]-=move; v5_l[cond]+=move; trig[cond]=True 

            v6_lumpsum *= rb

            # 2. 注入外部的分期資金 (使用換算過後的 dca_interval)
            if d % dca_interval == 0 and (d // dca_interval) < dca_parts:
                v1_base += periodic_cap
                v2_lev += periodic_cap
                v3_c += periodic_cap * 0.5; v3_b += periodic_cap * 0.5
                v4_b += periodic_cap * 0.5; v4_l += periodic_cap * 0.5
                v5_b += periodic_cap 
                # v6_lumpsum 不加碼

        # 整理成表格
        df_res = pd.DataFrame({
            '1. 一般散戶 (100% 基準)': v1_base,
            '2. 激進賭徒 (100% 槓桿)': v2_lev,
            '3. 保守定存 (50/50 持有)': v3_c + v3_b,
            '4. 紀律經理 (50大盤/50槓桿 再平衡)': v4_b + v4_l,
            '5. 老謀深算 (跌深抄底)': v5_c + v5_b + v5_l,
            '6. 時空旅人 (總成本首日全下)': v6_lumpsum
        })

    df_res_van = df_res / 10000

    # ==========================================
    # 5. 產出報表
    # ==========================================
    st.success(f"✅ 成功完成 {sim_years} 年蒙地卡羅模擬！總投入成本為：{total_capital_wan:.1f} 萬。")
    
    stats = []
    for col in df_res_van.columns:
        d = df_res_van[col]
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
    st.info("👈 資金設定更靈活了！請在左側輸入「初期資金」與「分期資金/頻率」，準備開始運算。")
