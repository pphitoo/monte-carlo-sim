import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import matplotlib.font_manager as fm
import os
import platform

# ==========================================
# 0. 字體設定與日期解封 (Streamlit Cloud 專用版)
# ==========================================
import matplotlib.pyplot as plt
from datetime import datetime

# 直接呼叫透過 packages.txt 安裝的系統級開源中文字體
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'Microsoft JhengHei', 'PingFang TC']
plt.rcParams['axes.unicode_minus'] = False  # 確保負號正常顯示

today = datetime.now().date()
min_date = datetime(2000, 1, 1).date()

# ==========================================
# 1. 網頁標題與說明書面板
# ==========================================
st.set_page_config(page_title="蒙地卡羅回測實驗室", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 蒙地卡羅量化回測實驗室 (實戰對決版)")
st.markdown("完美結合**「初期單筆資金」**與**「自訂次數的分期資金」**。加入**定存機會成本**概念，真實還原你的資金流在各種平行宇宙中，是否值得承擔股市風險！")

with st.expander("📖 實驗室說明與 6 大情境策略 (點擊展開)", expanded=False):
    st.markdown("""
    ### ⚙️ 資金流運作邏輯與勝率定義
    * **實際投入防呆**：若設定的分批次數超過模擬年限的極限，系統會自動截斷，只計算「實際有扣款」的真實總成本。
    * **勝率 (擊敗定存)**：系統會在背景同步模擬一個「無風險定存帳戶」。你的策略期末資產，必須大於「相同現金流放在銀行滾出來的本利和」，才會被判定為獲勝！

    ### 📈 6 大策略人設與作法
    * **1. 一般散戶 (100% 基準)**：全買大盤。
    * **2. 激進賭徒 (100% 槓桿)**：全買 2 倍槓桿。
    * **3. 保守定存 (50/50 持有)**：一半買大盤，一半放銀行定存 (1% 利率) 不動。
    * **4. 紀律經理 (50大盤/50槓桿 再平衡)**：一半 1 倍大盤，一半 2 倍槓桿。每年底強制重新平衡回 1:1 比例。
    * **5. 危機入市 (滿倉階梯換槓桿)**：100% 買大盤。當大盤每跌破設定級距 (例如 20%)，就賣掉設定比例的大盤換成 2 倍槓桿；創歷史新高時重置觸發器。
    * **6. 時空旅人 (神明對照組)**：向神明借未來所有的錢 (實際總成本)，第一天直接歐印大盤。
    """)

# ==========================================
# 2. 側邊欄：控制面板
# ==========================================
st.sidebar.title("⚙️ 控制面板")
engine = st.sidebar.selectbox("🧠 模擬引擎", ["1. 歷史區塊抽樣 (Block)", "2. 數學模型 (GBM)"])

st.sidebar.header("💰 彈性資金與機會成本")
initial_input_wan = st.sidebar.number_input("🏦 初期單筆資金 (萬)", min_value=0.0, value=100.0, step=10.0)
periodic_input_wan = st.sidebar.number_input("📥 每次分期投入 (萬)", min_value=0.0, value=10.0, step=1.0)
dca_parts = st.sidebar.slider("分批次數上限", min_value=1, max_value=360, value=12)
dca_interval_months = st.sidebar.slider("買入頻率 (月)", min_value=1, max_value=12, value=1)
# 🌟 C 計劃：加入無風險定存利率
risk_free_rate = st.sidebar.number_input("🏦 無風險定存利率 (%)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

sim_years = st.sidebar.slider("⏳ 模擬未來幾年？", min_value=1, max_value=50, value=10)
N = st.sidebar.slider("模擬次數 (平行宇宙)", min_value=1000, max_value=10000, value=5000)

# 🌟 C 計劃：防呆計算真實投入成本與定存帳戶本利和
days = sim_years * 252
dca_interval = dca_interval_months * 21 
# 計算這段時間最多只能扣款幾次
possible_injections = (days - 1) // dca_interval + 1
actual_dca_parts = min(dca_parts, possible_injections)

# 真實總成本
actual_total_capital_wan = initial_input_wan + (periodic_input_wan * actual_dca_parts)
initial_cap = initial_input_wan * 10000
periodic_cap = periodic_input_wan * 10000
total_cap = actual_total_capital_wan * 10000 # 時空旅人的本金

# 計算虛擬定存帳戶的本利和 (連續複利)
bank_value_wan = initial_input_wan
rf_growth = np.exp((risk_free_rate / 100) / 252)
for d in range(days):
    bank_value_wan *= rf_growth
    if d % dca_interval == 0 and (d // dca_interval) < actual_dca_parts:
        bank_value_wan += periodic_input_wan

st.sidebar.info(f"💡 **真實投入分析**\n\n"
                f"實際扣款次數：**{actual_dca_parts} 次**\n"
                f"實際投入總本金：**{actual_total_capital_wan:.1f} 萬**\n\n"
                f"🎯 **定存機會成本 (勝率基準)**\n"
                f"若全放定存，期末為：**{bank_value_wan:.1f} 萬**")

st.sidebar.header("📅 歷史區間與標的")
if "歷史" in engine:
    ticker = st.sidebar.text_input("輸入代碼 (Yahoo Finance)", value="0050.TW")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("開始", value=datetime(2008, 1, 1).date(), min_value=min_date, max_value=today)
    end_date = col2.date_input("結束", value=today, min_value=min_date, max_value=today)
    block_size = st.sidebar.slider("區塊大小 (歷史連續天數)", 5, 60, 21)
else:
    mu_base = st.sidebar.number_input("基準標的 預期年報酬 (%)", value=10.0) / 100
    sig_base = st.sidebar.number_input("基準標的 年化波動率 (%)", value=0.0) / 100
    df_t = st.sidebar.slider("肥尾效應強度 (t分配)", 2, 30, 3)

st.sidebar.header("🛠️ 槓桿與抄底微調")
lev_mult = st.sidebar.number_input("槓桿倍數", 1.0, 5.0, 2.0, 0.5)
drag_annual = st.sidebar.slider("槓桿標的 年化耗損 (%)", 0.0, 10.0, 1.5) / 100
drop_threshold = st.sidebar.slider("策略 5 抄底觸發級距 (%)", 5, 50, 20) / 100
transfer_pct = st.sidebar.slider("策略 5 賣大盤換槓桿比例 (%)", 10, 100, 20) / 100

# ==========================================
# 3. 資料下載快取
# ==========================================
@st.cache_data(show_spinner=False, ttl=600)
def get_hist_data(tkr, start, end):
    try:
        data = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'].iloc[:, 0].pct_change().dropna()
        return data['Close'].pct_change().dropna()
    except Exception as e:
        return None

# ==========================================
# 4. 核心運算區塊
# ==========================================
if st.sidebar.button("🚀 開始實戰模擬", type="primary", use_container_width=True):
    with st.spinner(f'⚙️ 正在進行第一階段平行宇宙運算...'):
        dt = 1/252
        cash_growth = np.exp(0.01 * dt)
        
        sim_ret_base = np.zeros((days, N))
        raw_hist_series = None
        raw_dates = None
        indices = None
        
        if "歷史" in engine:
            raw_hist_series = get_hist_data(ticker, start_date, end_date)
            if raw_hist_series is None or len(raw_hist_series) < block_size:
                st.error("❌ 無法載入歷史資料。請檢查日期或代碼。")
                st.stop()
            
            rets = raw_hist_series.values.flatten()
            raw_dates = raw_hist_series.index.strftime('%Y-%m-%d').values
            indices = np.random.randint(0, len(rets)-block_size, (int(np.ceil(days/block_size)), N))
            
            for b in range(indices.shape[0]):
                starts = indices[b,:]
                for i in range(block_size):
                    d_idx = b * block_size + i
                    if d_idx < days: 
                        sim_ret_base[d_idx, :] = rets[starts + i]
        else:
            Z = np.clip(np.random.standard_t(df_t, (days, N)) * np.sqrt(1/3), -15, 15)
            log_ret_base = (mu_base - 0.5 * sig_base**2) * dt + sig_base * np.sqrt(dt) * Z
            sim_ret_base = np.exp(log_ret_base) - 1
            
        sim_ret_lev = (sim_ret_base * lev_mult) - (drag_annual/252)
        m_B, m_L = np.maximum(0, 1+sim_ret_base), np.maximum(0, 1+sim_ret_lev)

        v1_base = np.ones(N) * initial_cap
        v2_lev = np.ones(N) * initial_cap
        v3_c = np.ones(N) * initial_cap * 0.5; v3_b = np.ones(N) * initial_cap * 0.5
        v4_b = np.ones(N) * initial_cap * 0.5; v4_l = np.ones(N) * initial_cap * 0.5
        v5_b = np.ones(N) * initial_cap; v5_l = np.zeros(N)
        trig_level = np.zeros(N) 
        v6_lumpsum = np.ones(N) * total_cap 
        ath = np.ones(N) 

        for d in range(days):
            rb, rl = m_B[d], m_L[d]
            
            v1_base *= rb
            v2_lev *= rl
            v3_c *= cash_growth; v3_b *= rb
            v4_b *= rb; v4_l *= rl
            if (d+1)%252==0: v4_b, v4_l = (v4_b+v4_l)*0.5, (v4_b+v4_l)*0.5
            v5_b *= rb; v5_l *= rl
            
            ath = np.maximum(ath, v6_lumpsum) 
            dd = v6_lumpsum / ath
            current_level = np.floor((1 - dd) / drop_threshold)
            trig_level[dd == 1] = 0 
            cond = current_level > trig_level 
            if np.any(cond):
                move = v5_b[cond] * transfer_pct
                v5_b[cond] -= move
                v5_l[cond] += move
                trig_level[cond] = current_level[cond] 

            v6_lumpsum *= rb

            if d % dca_interval == 0 and (d // dca_interval) < actual_dca_parts:
                v1_base += periodic_cap
                v2_lev += periodic_cap
                v3_c += periodic_cap * 0.5; v3_b += periodic_cap * 0.5
                v4_b += periodic_cap * 0.5; v4_l += periodic_cap * 0.5
                v5_b += periodic_cap 

        df_res = pd.DataFrame({
            '1. 一般散戶 (100% 基準)': v1_base,
            '2. 激賭徒徒 (100% 槓桿)': v2_lev,
            '3. 保守定存 (50/50 持有)': v3_c + v3_b,
            '4. 紀律經理 (50大盤/50槓桿)': v4_b + v4_l,
            '5. 危機入市 (階梯換槓桿)': v5_b + v5_l,
            '6. 時空旅人 (總成本首日全下)': v6_lumpsum
        })
        df_res_van = df_res / 10000

    # ==========================================
    # 🌟 4.5 階段：捕捉五大代表性宇宙的逐日軌跡
    # ==========================================
    with st.spinner(f'🕵️ 正在回放並記錄 5 大代表性宇宙的逐日軌跡...'):
        final_vals = v1_base
        sorted_args = np.argsort(final_vals)
        
        target_indices = [
            sorted_args[0],                   
            sorted_args[int(N * 0.25)],       
            sorted_args[int(N * 0.50)],       
            sorted_args[int(N * 0.75)],       
            sorted_args[-1]                   
        ]
        target_labels = ["Worst (最糟)", "Q1 (較差)", "Median (中位數)", "Q3 (較佳)", "Best (最佳)"]

        m_B_sub = m_B[:, target_indices]
        m_L_sub = m_L[:, target_indices]
        
        v1_s = np.ones(5) * initial_cap
        v2_s = np.ones(5) * initial_cap
        v3_c_s = np.ones(5) * initial_cap * 0.5; v3_b_s = np.ones(5) * initial_cap * 0.5
        v4_b_s = np.ones(5) * initial_cap * 0.5; v4_l_s = np.ones(5) * initial_cap * 0.5
        v5_b_s = np.ones(5) * initial_cap; v5_l_s = np.zeros(5)
        trig_lvl_s = np.zeros(5)
        v6_s = np.ones(5) * total_cap
        ath_s = np.ones(5)
        
        hist_v1 = np.zeros((days, 5))
        hist_v2 = np.zeros((days, 5))
        hist_v3 = np.zeros((days, 5))
        hist_v4 = np.zeros((days, 5))
        hist_v5 = np.zeros((days, 5))
        hist_v6 = np.zeros((days, 5))

        for d in range(days):
            rb, rl = m_B_sub[d], m_L_sub[d]
            
            v1_s *= rb
            v2_s *= rl
            v3_c_s *= cash_growth; v3_b_s *= rb
            v4_b_s *= rb; v4_l_s *= rl
            if (d+1)%252==0: v4_b_s, v4_l_s = (v4_b_s+v4_l_s)*0.5, (v4_b_s+v4_l_s)*0.5
            v5_b_s *= rb; v5_l_s *= rl
            
            ath_s = np.maximum(ath_s, v6_s) 
            dd = v6_s / ath_s
            current_level = np.floor((1 - dd) / drop_threshold)
            trig_lvl_s[dd == 1] = 0 
            cond = current_level > trig_lvl_s 
            if np.any(cond):
                move = v5_b_s[cond] * transfer_pct
                v5_b_s[cond] -= move
                v5_l_s[cond] += move
                trig_lvl_s[cond] = current_level[cond] 

            v6_s *= rb

            if d % dca_interval == 0 and (d // dca_interval) < actual_dca_parts:
                v1_s += periodic_cap
                v2_s += periodic_cap
                v3_c_s += periodic_cap * 0.5; v3_b_s += periodic_cap * 0.5
                v4_b_s += periodic_cap * 0.5; v4_l_s += periodic_cap * 0.5
                v5_b_s += periodic_cap 

            hist_v1[d] = v1_s / 10000
            hist_v2[d] = v2_s / 10000
            hist_v3[d] = (v3_c_s + v3_b_s) / 10000
            hist_v4[d] = (v4_b_s + v4_l_s) / 10000
            hist_v5[d] = (v5_b_s + v5_l_s) / 10000
            hist_v6[d] = v6_s / 10000

        sub_dates = np.empty((days, 5), dtype=object)
        sub_blocks = np.empty((days, 5), dtype=object)
        if "歷史" in engine:
            for col, og_idx in enumerate(target_indices):
                for b in range(indices.shape[0]):
                    start = indices[b, og_idx]
                    for i in range(block_size):
                        d_idx = b * block_size + i
                        if d_idx < days:
                            sub_dates[d_idx, col] = raw_dates[start + i]
                            sub_blocks[d_idx, col] = f"Block #{b+1}"
        else:
            sub_dates[:] = "N/A"
            sub_blocks[:] = "N/A"

    # ==========================================
    # 5. 產出報表與參數看板
    # ==========================================
    st.success(f"✅ 成功完成 {sim_years} 年蒙地卡羅模擬！勝率是以打敗定存 ({risk_free_rate}%) 為基準。")
    
    st.markdown("### 📋 本次模擬參數設定")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown("**💰 資金佈局**")
        st.write(f"- 初期單筆：**{initial_input_wan} 萬**")
        st.write(f"- 分期投入：**{periodic_input_wan} 萬** (實際扣 {actual_dca_parts} 次)")
        st.write(f"- 實際總成本：**{actual_total_capital_wan:.1f} 萬**")
    with p_col2:
        st.markdown("**⏳ 時間與基準**")
        st.write(f"- 模擬年限：**{sim_years} 年**")
        st.write(f"- 買入頻率：**每 {dca_interval_months} 個月**")
        st.write(f"- 基準定存：**{bank_value_wan:.1f} 萬** ({risk_free_rate}%)")
    with p_col3:
        st.markdown("**🛠️ 進階策略設定**")
        if "歷史" in engine:
            st.write(f"- 歷史區間：**{start_date} ~ {end_date}**")
        else:
            st.write(f"- 預期報酬/波動：**{mu_base*100:.1f}% / {sig_base*100:.1f}%**")
        st.write(f"- 槓桿設定：**{lev_mult}x** (耗損 {drag_annual*100:.1f}%)")
        st.write(f"- 危機入市：**每跌 {drop_threshold*100:.0f}% 換 {transfer_pct*100:.0f}%**")
    
    st.divider() 
    
    stats = []
    for col in df_res_van.columns:
        d = df_res_van[col]
        # 🌟 C 計劃：勝率不再是看總成本，而是看有沒有贏過銀行定存本利和
        win_rate = (d > bank_value_wan).mean() * 100
        stats.append({
            '策略': col,
            '勝率 (擊敗定存 %)': f"{win_rate:.1f}%",
            '中位數 (萬)': f"{d.median():,.1f}", 
            '悲觀 5% (萬)': f"{np.percentile(d, 5):,.1f}",
            '樂觀 5% (萬)': f"{np.percentile(d, 95):,.1f}" 
        })
    st.dataframe(pd.DataFrame(stats).set_index('策略'), use_container_width=True)
    
    st.subheader(f"📈 {sim_years} 年期終值分佈密度圖 (萬為單位)")
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in df_res_van.columns:
        sns.kdeplot(df_res_van[col], ax=ax, label=col, fill=True, alpha=0.15, linewidth=2)
    
    # 🌟 C 計劃：畫上真實成本線與定存機會成本線
    ax.axvline(actual_total_capital_wan, color='gray', linestyle=':', label=f'實際總成本 ({actual_total_capital_wan:.0f} 萬)', zorder=10)
    ax.axvline(bank_value_wan, color='red', linestyle='--', label=f'定存基準線 ({bank_value_wan:.0f} 萬)', zorder=10)
    
    title_prefix = "Historical Block Bootstrapping" if "歷史" in engine else "GBM Fat-Tail"
    ax.set_title(f'Monte Carlo Simulation: {sim_years}-Year Asset Distribution ({title_prefix})', fontsize=14)
    ax.set_xlabel('Final Asset Value (萬 TWD)', fontsize=12) 
    ax.set_ylabel('Density', fontsize=12)
    x_max = np.percentile(df_res_van.values, 95) * 1.5
    ax.set_xlim(0, max(x_max, actual_total_capital_wan * 2.5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ==========================================
    # 🌟 C 計劃：資料與邏輯驗證專區 
    # ==========================================
    st.divider()
    with st.expander("🕵️ 開發者專屬：資料與運算邏輯驗證專區", expanded=False):
        st.markdown("#### 1. 檢驗原始歷史資料")
        if "歷史" in engine and raw_hist_series is not None:
            df_raw = raw_hist_series.reset_index()
            df_raw.columns = ['Date', 'Daily Return']
            csv_raw = df_raw.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下載 Yahoo Finance 原始資料 (CSV)", csv_raw, "raw_history.csv", "text/csv")
        else:
            st.info("目前使用 GBM 數學模型，無歷史真實報價資料可供下載。")

        st.divider()

        st.markdown("#### 2. 下載 5 大代表性宇宙的逐日明細 (以大盤終值排名)")
        st.write("系統已精準捕捉在 5000 次平行宇宙中，表現達到 **Worst, Q1, Median, Q3, Best** 的五條時間線。")
        st.write("你可以點擊下方按鈕，下載該宇宙這幾千天以來的每一天淨值變化、大盤跌幅與資金軌跡！")
        
        cols = st.columns(5)
        for i, label in enumerate(target_labels):
            df_export = pd.DataFrame({
                'Day': np.arange(1, days + 1),
                '抽樣區塊編號': sub_blocks[:, i],
                '歷史對應日期': sub_dates[:, i],
                '大盤單日報酬': m_B_sub[:, i] - 1,
                '槓桿單日報酬': m_L_sub[:, i] - 1,
                '1. 一般散戶': hist_v1[:, i],
                '2. 激進賭徒': hist_v2[:, i],
                '3. 保守定存': hist_v3[:, i],
                '4. 紀律經理': hist_v4[:, i],
                '5. 危機入市': hist_v5[:, i],
                '6. 時空旅人': hist_v6[:, i],
            })
            csv_export = df_export.to_csv(index=False).encode('utf-8-sig')
            cols[i].download_button(f"📥 下載 {label}", csv_export, f"Universe_{label.split(' ')[0]}.csv", "text/csv")

else:
    st.info("👈 防呆機制與定存機會成本已上線！再也不會出現幽靈本金了。")


