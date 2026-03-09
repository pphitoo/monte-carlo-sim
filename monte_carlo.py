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
import tempfile

# ==========================================
# 0. 字體設定與日期解封 (Streamlit Cloud 專用版)
# ==========================================
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'Microsoft JhengHei', 'PingFang TC']
plt.rcParams['axes.unicode_minus'] = False  

today = datetime.now().date()
min_date = datetime(2000, 1, 1).date()

# 🌟 初始化 Session State 保險箱
if 'sim_done' not in st.session_state:
    st.session_state['sim_done'] = False

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

    ### 📈 6 大策略人設與作法
    * **1. 一般散戶**：全買大盤。
    * **2. 激進賭徒**：全買 2 倍槓桿。
    * **3. 保守定存**：一半買大盤，一半放銀行定存。
    * **4. 紀律經理**：一半大盤，一半槓桿，每年再平衡。
    * **5. 危機入市**：大盤下跌破級距換槓桿。
    * **6. 時空旅人**：總成本第一天全下大盤。
    """)

# ==========================================
# 2. 側邊欄：控制面板
# ==========================================
st.sidebar.title("⚙️ 控制面板")
engine = st.sidebar.selectbox("🧠 模擬引擎", ["1. 歷史區塊抽樣 (Block)", "2. 數學模型 (GBM)"])

st.sidebar.header("💰 彈性資金與機會成本")
initial_input_wan = st.sidebar.number_input("🏦 初期單筆資金 (萬)", min_value=0.0, value=100.0, step=10.0)
periodic_input_wan = st.sidebar.number_input("📥 每次分期投入 (萬)", min_value=0.0, value=10.0, step=1.0)
dca_parts = st.sidebar.number_input("分批次數上限", min_value=1, value=12, step=1)
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
                f"🎯 **定存基準線**：**{bank_value_wan:.1f} 萬**")

st.sidebar.header("📅 歷史區間與標的")
if "歷史" in engine:
    api_source = st.sidebar.radio("📡 歷史資料庫引擎", ["🔥 雙源共識融合 (FinMind 主體 + Yahoo 智能除錯)"])
    ticker = st.sidebar.text_input("輸入代碼", value="0050.TW")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("開始", value=datetime(2003, 6, 30).date(), min_value=min_date, max_value=today)
    end_date = col2.date_input("結束", value=today, min_value=min_date, max_value=today)
    block_size = st.sidebar.slider("區塊大小 (歷史連續天數)", 5, 60, 21)
else:
    ticker = "數學模型 (無特定標的)"
    mu_base = st.sidebar.number_input("基準標的 預期年報酬 (%)", value=10.0) / 100
    sig_base = st.sidebar.number_input("基準標的 年化波動率 (%)", value=16.0) / 100
    df_t = st.sidebar.slider("肥尾效應強度 (t分配)", min_value=2.0, max_value=30.0, value=3.0, step=0.5)

st.sidebar.header("🛠️ 槓桿與抄底微調")
lev_mult = st.sidebar.number_input("槓桿倍數", 1.0, 5.0, 2.0, 0.5)
drag_annual = st.sidebar.slider("槓桿標的 年化耗損 (%)", 0.0, 10.0, 1.5) / 100
drop_threshold = st.sidebar.slider("策略 5 抄底觸發級距 (%)", 5, 50, 20) / 100
transfer_pct = st.sidebar.slider("策略 5 賣大盤換槓桿比例 (%)", 10, 100, 20) / 100

# ==========================================
# 3. 雙源共識融合下載模組
# ==========================================
@st.cache_data(show_spinner=False, ttl=600)
def get_hist_data_consensus(tkr, start, end):
    try:
        df_y = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        try:
            data_y = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
            if not data_y.empty:
                if isinstance(data_y.columns, pd.MultiIndex):
                    df_y = data_y['Close'].iloc[:, 0].pct_change().dropna()
                else:
                    df_y = data_y['Close'].pct_change().dropna()
                if df_y.index.tz is not None:
                    df_y.index = df_y.index.tz_localize(None)
        except:
            pass
            
        df_f = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        try:
            clean_tkr = tkr.replace(".TW", "").replace(".TWO", "")
            url = "https://api.finmindtrade.com/api/v4/data"
            params_price = {"dataset": "TaiwanStockPrice", "data_id": clean_tkr, "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d")}
            res_price = requests.get(url, params=params_price).json()
            if res_price.get("msg") == "success" and len(res_price.get("data", [])) > 0:
                df_price = pd.DataFrame(res_price["data"])
                df_price['date'] = pd.to_datetime(df_price['date'])
                df_price.set_index('date', inplace=True)
                
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
                
                df_price['prev_close'] = df_price['close'].shift(1)
                df_price['total_return'] = (df_price['close'] + df_price['dividend']) / df_price['prev_close'] - 1
                df_f = df_price['total_return'].dropna()
        except:
            pass

        if df_f.empty and df_y.empty:
            return None
        
        df_merged = pd.DataFrame({'FinMind_Raw': df_f, 'Yahoo_Raw': df_y})
        df_merged['Final_Consensus'] = df_merged['FinMind_Raw'].fillna(df_merged['Yahoo_Raw'])
        anomaly_mask = df_merged['Final_Consensus'].abs() > 0.15
        for date in df_merged[anomaly_mask].index:
            y_val = df_merged.loc[date, 'Yahoo_Raw']
            if pd.notna(y_val) and abs(y_val) <= 0.15:
                df_merged.loc[date, 'Final_Consensus'] = y_val 
            else:
                df_merged.loc[date, 'Final_Consensus'] = 0.0   
        return df_merged.dropna(subset=['Final_Consensus'])
    except Exception as e:
        return None

# ==========================================
# 4. 核心運算區塊 
# ==========================================
if st.sidebar.button("🚀 開始實戰模擬", type="primary", use_container_width=True):
    with st.spinner(f'⚙️ 正在啟動運算引擎...'):
        dt = 1/252
        cash_growth = np.exp(0.01 * dt)
        sim_ret_base = np.zeros((days, N))
        raw_hist_df = None
        raw_dates = None
        indices = None
        
        if "歷史" in engine:
            raw_hist_df = get_hist_data_consensus(ticker, start_date, end_date)
            if raw_hist_df is None or len(raw_hist_df) < block_size:
                st.error("❌ 無法載入歷史資料。請檢查日期或代碼。")
                st.stop()
            rets = raw_hist_df['Final_Consensus'].values.flatten()
            raw_dates = raw_hist_df.index.strftime('%Y-%m-%d').values
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
            v1_base *= rb; v2_lev *= rl
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
                v1_base += periodic_cap; v2_lev += periodic_cap
                v3_c += periodic_cap * 0.5; v3_b += periodic_cap * 0.5
                v4_b += periodic_cap * 0.5; v4_l += periodic_cap * 0.5
                v5_b += periodic_cap 

        df_res = pd.DataFrame({
            '1. 一般散戶 (100% 基準)': v1_base, '2. 激進賭徒 (100% 槓桿)': v2_lev,
            '3. 保守定存 (50/50 持有)': v3_c + v3_b, '4. 紀律經理 (50大盤/50槓桿)': v4_b + v4_l,
            '5. 危機入市 (階梯換槓桿)': v5_b + v5_l, '6. 時空旅人 (總成本首日全下)': v6_lumpsum
        })
        df_res_van = df_res / 10000

        # 🌟 4.5 捕捉五大代表性宇宙
        final_vals = v1_base
        sorted_args = np.argsort(final_vals)
        target_indices = [sorted_args[0], sorted_args[int(N * 0.25)], sorted_args[int(N * 0.50)], sorted_args[int(N * 0.75)], sorted_args[-1]]
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
        
        hist_v1 = np.zeros((days, 5)); hist_v2 = np.zeros((days, 5)); hist_v3 = np.zeros((days, 5))
        hist_v4 = np.zeros((days, 5)); hist_v5 = np.zeros((days, 5)); hist_v6 = np.zeros((days, 5))

        for d in range(days):
            rb, rl = m_B_sub[d], m_L_sub[d]
            v1_s *= rb; v2_s *= rl
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
                v1_s += periodic_cap; v2_s += periodic_cap
                v3_c_s += periodic_cap * 0.5; v3_b_s += periodic_cap * 0.5
                v4_b_s += periodic_cap * 0.5; v4_l_s += periodic_cap * 0.5
                v5_b_s += periodic_cap 

            hist_v1[d] = v1_s / 10000; hist_v2[d] = v2_s / 10000; hist_v3[d] = (v3_c_s + v3_b_s) / 10000
            hist_v4[d] = (v4_b_s + v4_l_s) / 10000; hist_v5[d] = (v5_b_s + v5_l_s) / 10000; hist_v6[d] = v6_s / 10000

        sub_dates = np.empty((days, 5), dtype=object); sub_blocks = np.empty((days, 5), dtype=object)
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

        st.session_state['sim_data'] = {
            'df_res_van': df_res_van, 'api_label': "雙源共識引擎" if "歷史" in engine else "GBM",
            'sim_years': sim_years, 'risk_free_rate': risk_free_rate,
            'initial_input_wan': initial_input_wan, 'periodic_input_wan': periodic_input_wan,
            'actual_dca_parts': actual_dca_parts, 'actual_total_capital_wan': actual_total_capital_wan,
            'dca_interval_months': dca_interval_months, 'bank_value_wan': bank_value_wan,
            'ticker': ticker, 'start_date': start_date if "歷史" in engine else None,
            'end_date': end_date if "歷史" in engine else None,
            'mu_base': mu_base if "歷史" not in engine else None, 'sig_base': sig_base if "歷史" not in engine else None,
            'df_t': df_t if "歷史" not in engine else None, 'lev_mult': lev_mult, 'drag_annual': drag_annual,
            'drop_threshold': drop_threshold, 'transfer_pct': transfer_pct,
            'raw_hist_df': raw_hist_df, 'engine': engine, 'target_labels': target_labels,
            'sub_blocks': sub_blocks, 'sub_dates': sub_dates,
            'm_B_sub': m_B_sub, 'm_L_sub': m_L_sub,
            'hist_v1': hist_v1, 'hist_v2': hist_v2, 'hist_v3': hist_v3,
            'hist_v4': hist_v4, 'hist_v5': hist_v5, 'hist_v6': hist_v6, 'days': days
        }
        st.session_state['sim_done'] = True

# ==========================================
# 5. 畫面渲染區 & PDF 報告生成功能
# ==========================================
if st.session_state['sim_done']:
    data = st.session_state['sim_data']
    
    st.success(f"✅ 成功完成 {data['sim_years']} 年蒙地卡羅模擬 ({data['api_label']})！勝率是以打敗定存 ({data['risk_free_rate']}%) 為基準。")
    
    st.markdown("### 📋 本次模擬參數設定")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown("**💰 資金佈局**")
        st.write(f"- 初期單筆：**{data['initial_input_wan']} 萬**")
        st.write(f"- 分期投入：**{data['periodic_input_wan']} 萬** (實際扣 {data['actual_dca_parts']} 次)")
        st.write(f"- 實際總成本：**{data['actual_total_capital_wan']:.1f} 萬**")
    with p_col2:
        st.markdown("**⏳ 時間與基準**")
        st.write(f"- 模擬年限：**{data['sim_years']} 年**")
        st.write(f"- 買入頻率：**每 {data['dca_interval_months']} 個月**")
        st.write(f"- 基準定存：**{data['bank_value_wan']:.1f} 萬** ({data['risk_free_rate']}%)")
    with p_col3:
        st.markdown("**🛠️ 進階策略設定**")
        st.write(f"- 模擬標的：**{data['ticker']}**")
        if "歷史" in data['engine']:
            st.write(f"- 歷史區間：**{data['start_date']} ~ {data['end_date']}**")
        else:
            st.write(f"- 預期報酬/波動：**{data['mu_base']*100:.1f}% / {data['sig_base']*100:.1f}%**")
            st.write(f"- 肥尾效應強度：**{data['df_t']}** (數值越小，極端股災機率越高)")
        st.write(f"- 槓桿與危機入市：**{data['lev_mult']}x** (跌 **{data['drop_threshold']*100:.0f}%** 換 **{data['transfer_pct']*100:.0f}%**)")
    
    st.divider() 
    
    def calc_cagr(fv, pv, years):
        if fv <= 0: return -100.0
        return ((fv / pv) ** (1 / years) - 1) * 100

    # 🌟 表格數據生成區 (加入「年化報酬率」明確備註)
    stats = []
    df_res_van = data['df_res_van']
    for col in df_res_van.columns:
        d = df_res_van[col]
        win_rate = (d > data['bank_value_wan']).mean() * 100
        v_min, v_q1, v_med, v_q3, v_max = np.min(d), np.percentile(d, 25), np.median(d), np.percentile(d, 75), np.max(d)
        
        stats.append({
            '策略': col,
            '勝率(贏定存)': f"{win_rate:.1f}%",
            '最糟 Min': f"{v_min:,.1f}萬\n(年化報酬率 {calc_cagr(v_min, data['actual_total_capital_wan'], data['sim_years']):.1f}%)",
            '較差 Q1': f"{v_q1:,.1f}萬\n(年化報酬率 {calc_cagr(v_q1, data['actual_total_capital_wan'], data['sim_years']):.1f}%)",
            '中位 Median': f"{v_med:,.1f}萬\n(年化報酬率 {calc_cagr(v_med, data['actual_total_capital_wan'], data['sim_years']):.1f}%)",
            '較佳 Q3': f"{v_q3:,.1f}萬\n(年化報酬率 {calc_cagr(v_q3, data['actual_total_capital_wan'], data['sim_years']):.1f}%)",
            '最佳 Max': f"{v_max:,.1f}萬\n(年化報酬率 {calc_cagr(v_max, data['actual_total_capital_wan'], data['sim_years']):.1f}%)"
        })
        
    st.markdown("#### 🏆 策略終值與年化報酬率對照表")
    df_stats = pd.DataFrame(stats).set_index('策略')
    st.dataframe(df_stats, use_container_width=True)
    
    # 畫圖
    st.subheader(f"📈 {data['sim_years']} 年期：終值分佈 vs 年化報酬率(CAGR) 盒鬚圖")
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        for col in df_res_van.columns:
            sns.kdeplot(df_res_van[col], ax=ax1, label=col, fill=True, alpha=0.15, linewidth=2)
        ax1.axvline(data['actual_total_capital_wan'], color='gray', linestyle=':', label=f"實際總成本 ({data['actual_total_capital_wan']:.0f} 萬)", zorder=10)
        ax1.axvline(data['bank_value_wan'], color='red', linestyle='--', label=f"定存基準 ({data['bank_value_wan']:.0f} 萬)", zorder=10)
        ax1.set_title(f"Final Asset Value Distribution Density", fontsize=14)
        ax1.set_xlabel('Final Asset Value (萬 TWD)', fontsize=12); ax1.set_ylabel('Density', fontsize=12)
        ax1.set_xlim(0, max(np.percentile(df_res_van.values, 95) * 1.5, data['actual_total_capital_wan'] * 2.5))
        ax1.legend(); ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with fig_col2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        safe_res = np.maximum(df_res_van, 0.0001)
        df_cagr = ((safe_res / data['actual_total_capital_wan']) ** (1 / data['sim_years']) - 1) * 100
        sns.boxplot(data=df_cagr, ax=ax2, orient='h', palette='Set2', showfliers=True, flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))
        ax2.axvline(data['risk_free_rate'], color='red', linestyle='--', label=f"定存基準 ({data['risk_free_rate']}%)", zorder=10)
        ax2.set_title("CAGR Distribution (Box Plot)", fontsize=14)
        ax2.set_xlabel('CAGR (%)', fontsize=12); ax2.set_ylabel('')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # ==========================================
    # 🌟 匯出 PDF 報告按鈕區 (圖表+精美表格完全體)
    # ==========================================
    st.divider()
    st.markdown("### 📄 實驗室報告輸出")
    
    def generate_pdf():
        try:
            from fpdf import FPDF
        except ImportError:
            st.error("❌ 找不到 FPDF 套件，請確認已安裝 fpdf2。")
            return None
            
        font_path = "NotoSansTC-Regular.ttf"
        if not os.path.exists(font_path):
            st.error("❌ 找不到實體中文字型檔！請確保您已經將 `NotoSansTC-Regular.ttf` 上傳至 GitHub 與 `monte_carlo.py` 同一個資料夾中。")
            return None
            
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        pdf.add_font('NotoSans', '', font_path, uni=True)
        pdf.set_font('NotoSans', '', 16)
            
        title = "蒙地卡羅量化回測實驗室 - 分析報告"
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.set_font('NotoSans', '', 12)
        pdf.ln(5)
        
        params_txt = f"""
        [模擬參數]
        - 模擬標的: {data['ticker']}
        - 模擬年限: {data['sim_years']} 年 ({data['api_label']})
        - 總投入成本: {data['actual_total_capital_wan']:.1f} 萬
        - 槓桿設定: {data['lev_mult']}x (耗損 {data['drag_annual']*100}%)
        """
        for line in params_txt.strip().split('\n'):
            pdf.cell(0, 8, line.strip(), ln=True)
        
        pdf.ln(5)
        pdf.set_font('NotoSans', '', 14)
        pdf.cell(0, 10, "🏆 策略終值與年化報酬率對照表", ln=True, align="L")
        
        # 🌟 將數據 DataFrame 轉換為 Matplotlib 精美表格圖片
        fig_tbl, ax_tbl = plt.subplots(figsize=(16, 5))
        ax_tbl.axis('off')
        
        df_render = df_stats.reset_index()
        tbl = ax_tbl.table(cellText=df_render.values,
                           colLabels=df_render.columns,
                           loc='center',
                           cellLoc='center')
        
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 3.5) # 拉高儲存格，讓多行文字完美顯示
        
        # 幫表格塗上專業的配色
        for (i, j), cell in tbl.get_celld().items():
            if i == 0:
                cell.set_facecolor('#4C72B0')
                cell.set_text_props(color='white', weight='bold')
            elif j == 0:
                cell.set_facecolor('#EAEAEA')
                cell.set_text_props(weight='bold')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_tbl:
            fig_tbl.savefig(tmp_tbl.name, format="png", bbox_inches="tight", dpi=200)
            pdf.image(tmp_tbl.name, x=10, w=190)
        plt.close(fig_tbl) # 釋放記憶體
            
        pdf.ln(5)
        pdf.cell(0, 10, "📈 終值分佈與年化報酬率盒鬚圖", ln=True, align="L")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile1:
            fig1.savefig(tmpfile1.name, format="png", bbox_inches="tight", dpi=150)
            pdf.image(tmpfile1.name, x=10, w=190)
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile2:
            fig2.savefig(tmpfile2.name, format="png", bbox_inches="tight", dpi=150)
            pdf.ln(5)
            pdf.image(tmpfile2.name, x=10, w=190)
            
        return bytes(pdf.output())

    with st.spinner("準備 PDF 報告中..."):
        pdf_bytes = generate_pdf()
        if pdf_bytes:
            st.download_button(
                label="📥 下載 PDF 分析報告",
                data=pdf_bytes,
                file_name=f"MonteCarlo_Report_{data['ticker']}.pdf",
                mime="application/pdf",
                type="primary"
            )

    # ==========================================
    # 🌟 開發者專區
    # ==========================================
    st.divider()
    with st.expander("🕵️ 開發者專屬：資料與運算邏輯驗證專區", expanded=False):
        if "歷史" in data['engine'] and data['raw_hist_df'] is not None:
            df_export_check = data['raw_hist_df'].reset_index()
            if 'index' in df_export_check.columns: df_export_check.rename(columns={'index': 'Date'}, inplace=True)
            elif 'date' in df_export_check.columns: df_export_check.rename(columns={'date': 'Date'}, inplace=True)
            csv_check = df_export_check.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下載 雙引擎除錯對照表 (CSV)", csv_check, "consensus_history_check.csv", "text/csv")

        cols = st.columns(5)
        for i, label in enumerate(data['target_labels']):
            df_export = pd.DataFrame({
                'Day': np.arange(1, data['days'] + 1), '抽樣區塊編號': data['sub_blocks'][:, i], '歷史對應日期': data['sub_dates'][:, i],
                '大盤單日報酬': data['m_B_sub'][:, i] - 1, '槓桿單日報酬': data['m_L_sub'][:, i] - 1,
                '1. 一般散戶': data['hist_v1'][:, i], '2. 激進賭徒': data['hist_v2'][:, i], '3. 保守定存': data['hist_v3'][:, i],
                '4. 紀律經理': data['hist_v4'][:, i], '5. 危機入市': data['hist_v5'][:, i], '6. 時空旅人': data['hist_v6'][:, i],
            })
            cols[i].download_button(f"📥 下載 {label}", df_export.to_csv(index=False).encode('utf-8-sig'), f"Universe_{label.split(' ')[0]}.csv", "text/csv")

else:
    st.info("👈 防閃退記憶體保險箱已就緒！請在左側設定參數並點擊「開始實戰模擬」。")
