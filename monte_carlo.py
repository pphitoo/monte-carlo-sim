import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
from datetime import datetime
import matplotlib.font_manager as fm
import os
import platform

# ==========================================
# 0. 字體設定與日期解封 (Streamlit Cloud 專用版)
# ==========================================
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'Microsoft JhengHei', 'PingFang TC']
plt.rcParams['axes.unicode_minus'] = False  

today = datetime.now().date()
min_date = datetime(2000, 1, 1).date()

# ==========================================
# 1. 網頁標題與說明書面板
# ==========================================
st.set_page_config(page_title="蒙地卡羅回測實驗室", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 蒙地卡羅量化回測實驗室 (實戰對決版)")
st.markdown("完美結合**「初期單筆資金」**與**「自訂次數的分期資金」**。加入**定存機會成本**與**雙源共識除錯引擎**，真實還原你的資金流在各種平行宇宙中，是否值得承擔股市風險！")

with st.expander("📖 實驗室說明與 6 大情境策略 (點擊展開)", expanded=False):
    st.markdown("""
    ### ⚙️ 資金流運作邏輯與勝率定義
    * **實際投入防呆**：若設定的分批次數超過模擬年限的極限，系統會自動截斷，只計算「實際有扣款」的真實總成本。
    * **勝率 (擊敗定存)**：系統會在背景同步模擬一個「無風險定存帳戶」。你的策略期末資產，必須大於「相同現金流放在銀行滾出來的本利和」，才會被判定為獲勝！

    ---

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
risk_free_rate = st.sidebar.number_input("🏦 無風險定存利率 (%)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

sim_years = st.sidebar.slider("⏳ 模擬未來幾年？", min_value=1, max_value=50, value=10)
N = st.sidebar.slider("模擬次數 (平行宇宙)", min_value=1000, max_value=10000, value=5000)

days = sim_years * 252
dca_interval = dca_interval_months * 21 
possible_injections = (days - 1) // dca_interval + 1
actual_dca_parts = min(dca_parts, possible_injections)

actual_total_capital_wan = initial_input_wan + (periodic_input_wan * actual_dca_parts)
initial_cap = initial_input_wan * 10000
periodic_cap = periodic_input_wan * 10000
total_cap = actual_total_capital_wan * 10000 

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
    # 🌟 升級為雙源共識融合引擎
    api_source = st.sidebar.radio("📡 歷史資料庫引擎", [
        "🔥 雙源共識融合 (FinMind 主體 + Yahoo 智能除錯)"
    ])
    ticker = st.sidebar.text_input("輸入代碼", value="0050.TW")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("開始", value=datetime(2003, 6, 30).date(), min_value=min_date, max_value=today)
    end_date = col2.date_input("結束", value=today, min_value=min_date, max_value=today)
    block_size = st.sidebar.slider("區塊大小 (歷史連續天數)", 5, 60, 21)
else:
    mu_base = st.sidebar.number_input("基準標的 預期年報酬 (%)", value=10.0) / 100
    sig_base = st.sidebar.number_input("基準標的 年化波動率 (%)", value=16.0) / 100
    df_t = st.sidebar.slider("肥尾效應強度 (t分配)", 2, 30, 3)

st.sidebar.header("🛠️ 槓桿與抄底微調")
lev_mult = st.sidebar.number_input("槓桿倍數", 1.0, 5.0, 2.0, 0.5)
drag_annual = st.sidebar.slider("槓桿標的 年化耗損 (%)", 0.0, 10.0, 1.5) / 100
drop_threshold = st.sidebar.slider("策略 5 抄底觸發級距 (%)", 5, 50, 20) / 100
transfer_pct = st.sidebar.slider("策略 5 賣大盤換槓桿比例 (%)", 10, 100, 20) / 100

# ==========================================
# 3. 雙源共識融合下載模組 (完美除蟲版)
# ==========================================
@st.cache_data(show_spinner=False, ttl=600)
def get_hist_data_consensus(tkr, start, end):
    try:
        # --- 步驟 1：下載 Yahoo 資料 (當作糾察隊) ---
        df_y = pd.Series(dtype=float)
        try:
            data_y = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
            if not data_y.empty:
                if isinstance(data_y.columns, pd.MultiIndex):
                    df_y = data_y['Close'].iloc[:, 0].pct_change().dropna()
                else:
                    df_y = data_y['Close'].pct_change().dropna()
        except:
            pass
            
        # --- 步驟 2：下載 FinMind 資料 (當作主體，因為歷史夠長) ---
        df_f = pd.Series(dtype=float)
        try:
            clean_tkr = tkr.replace(".TW", "").replace(".TWO", "")
            
            # 抓股價
            url = "https://api.finmindtrade.com/api/v4/data"
            params_price = {"dataset": "TaiwanStockPrice", "data_id": clean_tkr, "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d")}
            res_price = requests.get(url, params=params_price).json()
            
            if res_price.get("msg") == "success" and len(res_price.get("data", [])) > 0:
                df_price = pd.DataFrame(res_price["data"])
                df_price['date'] = pd.to_datetime(df_price['date'])
                df_price.set_index('date', inplace=True)
                
                # 抓股息
                params_div = {"dataset": "TaiwanStockDividendResult", "data_id": clean_tkr, "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d")}
                res_div = requests.get(url, params=params_div).json()
                df_price['dividend'] = 0.0 
                
                if res_div.get("msg") == "success" and len(res_div.get("data", [])) > 0:
                    df_div = pd.DataFrame(res_div["data"])
                    df_div['date'] = pd.to_datetime(df_div['date'])
                    df_div.set_index('date', inplace=True)
                    div_col = 'stock_and_cache_dividend' if 'stock_and_cache_dividend' in df_div.columns else df_div.columns[-1]
                    df_price = df_price.join(df_div[[div_col]], how='left')
                    df_price['dividend'] = df_price[div_col].fillna(0.0)
                
                # 算總報酬
                df_price['prev_close'] = df_price['close'].shift(1)
                df_price['total_return'] = (df_price['close'] + df_price['dividend']) / df_price['prev_close'] - 1
                df_f = df_price['total_return'].dropna()
        except:
            pass

        # --- 步驟 3：神級融合邏輯 (Consensus Engine) ---
        if df_f.empty and df_y.empty:
            return None
        
        # 建立一個合併用的 DataFrame
        df_merged = pd.DataFrame({'finmind': df_f, 'yahoo': df_y})
        
        # 以 FinMind 為主體，如果有缺值，用 Yahoo 補上
        df_merged['final_return'] = df_merged['finmind'].fillna(df_merged['yahoo'])
        
        # 🚨 啟動極端值濾網：找出單日漲跌大於 15% 的異常值 (胖手指)
        anomaly_mask = df_merged['final_return'].abs() > 0.15
        
        # 如果發生異常，優先拿 Yahoo 的正常數據來替換；如果 Yahoo 也異常或沒有資料，就強制歸零 (當作沒開盤)
        for date in df_merged[anomaly_mask].index:
            y_val = df_merged.loc[date, 'yahoo']
            if pd.notna(y_val) and abs(y_val) <= 0.15:
                df_merged.loc[date, 'final_return'] = y_val # 用正常的 Yahoo 數據替換核彈
            else:
                df_merged.loc[date, 'final_return'] = 0.0   # 強制歸零防禦
                
        return df_merged['final_return'].dropna()
        
    except Exception as e:
        return None

# ==========================================
# 4. 核心運算區塊
# ==========================================
if st.sidebar.button("🚀 開始實戰模擬", type="primary", use_container_width=True):
    with st.spinner(f'⚙️ 正在啟動雙源共識融合引擎...'):
        dt = 1/252
        cash_growth = np.exp(0.01 * dt)
        
        sim_ret_base = np.zeros((days, N))
        raw_hist_series = None
        raw_dates = None
        indices = None
        
        if "歷史" in engine:
            raw_hist_series = get_hist_data_consensus(ticker, start_date, end_date)
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
            '2. 激進賭徒 (100% 槓桿)': v2_lev,
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
    api_label = "雙源共識引擎" if "歷史" in engine else "GBM"
    st.success(f"✅ 成功完成 {sim_years} 年蒙地卡羅模擬 ({api_label})！勝率是以打敗定存 ({risk_free_rate}%) 為基準。")
    
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
    
    def calc_cagr(fv, pv, years):
        if fv <= 0: return -100.0
        return ((fv / pv) ** (1 / years) - 1) * 100

    stats = []
    for col in df_res_van.columns:
        d = df_res_van[col]
        win_rate = (d > bank_value_wan).mean() * 100
        
        v_min = np.min(d)
        v_q1 = np.percentile(d, 25)
        v_med = np.median(d)
        v_q3 = np.percentile(d, 75)
        v_max = np.max(d)
        
        stats.append({
            '策略': col,
            '勝率 (贏過定存)': f"{win_rate:.1f}%",
            '最糟 Min': f"{v_min:,.1f} 萬 ({calc_cagr(v_min, actual_total_capital_wan, sim_years):.1f}%)",
            '較差 Q1': f"{v_q1:,.1f} 萬 ({calc_cagr(v_q1, actual_total_capital_wan, sim_years):.1f}%)",
            '中位 Median': f"{v_med:,.1f} 萬 ({calc_cagr(v_med, actual_total_capital_wan, sim_years):.1f}%)",
            '較佳 Q3': f"{v_q3:,.1f} 萬 ({calc_cagr(v_q3, actual_total_capital_wan, sim_years):.1f}%)",
            '最佳 Max': f"{v_max:,.1f} 萬 ({calc_cagr(v_max, actual_total_capital_wan, sim_years):.1f}%)"
        })
        
    st.markdown("#### 🏆 策略終值與年化報酬率對照表")
    st.info("💡 括號內為 **(換算總成本年化報酬率 CAGR)**。注意：除時空旅人外，其他策略為分期投入，此年化報酬率代表「將現金閒置的機會成本一併計入」的嚴格總體年化標準。")
    st.dataframe(pd.DataFrame(stats).set_index('策略'), use_container_width=True)
    
    st.subheader(f"📈 {sim_years} 年期終值分佈密度圖 (萬為單位)")
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in df_res_van.columns:
        sns.kdeplot(df_res_van[col], ax=ax, label=col, fill=True, alpha=0.15, linewidth=2)
    
    ax.axvline(actual_total_capital_wan, color='gray', linestyle=':', label=f'實際總成本 ({actual_total_capital_wan:.0f} 萬)', zorder=10)
    ax.axvline(bank_value_wan, color='red', linestyle='--', label=f'定存基準線 ({bank_value_wan:.0f} 萬)', zorder=10)
    
    title_prefix = f"Historical Block ({api_label})" if "歷史" in engine else "GBM Fat-Tail"
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
            st.download_button(f"📥 下載 融合清洗後原始資料 (CSV)", csv_raw, f"raw_history_consensus.csv", "text/csv")
        else:
            st.info("目前使用 GBM 數學模型，無歷史真實報價資料可供下載。")

        st.divider()

        st.markdown("#### 2. 下載 5 大代表性宇宙的逐日明細 (以大盤終值排名)")
        st.write("系統已精準捕捉在 5000 次平行宇宙中，表現達到 **Worst, Q1, Median, Q3, Best** 的五條時間線。")
        
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
    st.info("👈 雙源共識融合引擎已上線！準備好迎接最純淨的歷史回測了嗎？")
