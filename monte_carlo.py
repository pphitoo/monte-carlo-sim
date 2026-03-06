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
st.set_page_config(page_title="量化回測實驗室", layout="wide",
                   initial_sidebar_state="expanded")
st.title("🔬 終極量化回測實驗室 (雙引擎架構)")
st.markdown("自由切換「歷史區塊抽樣」與「GBM 肥尾數學模型」，自訂標的、參數與抄底策略，尋找最佳投資組合。")

# ==========================================
# 1. 側邊欄：強大的互動參數控制中心
# ==========================================
st.sidebar.title("⚙️ 控制面板")

# 模組 1: 核心引擎選擇
engine = st.sidebar.selectbox(
    "🧠 選擇模擬引擎", ["1. 歷史區塊抽樣 (Block Bootstrapping)", "2. 數學模型 (GBM 肥尾效應)"])
st.sidebar.divider()

# 模組 2: 基本參數設定
st.sidebar.header("💰 基本資金與時間")
initial_capital = st.sidebar.number_input(
    "初始資金 (元)", min_value=100000, value=2000000, step=100000)
sim_years = st.sidebar.slider("⏳ 獨立設定：要模擬未來幾年？", min_value=1, max_value=30,
                              value=10, help="這決定了平行宇宙的長度。你可以拿短短幾年的歷史碎片，重複抽樣拼湊出 10 年或 20 年的極端未來。")
# ⚠️ 修正上限為 10000，保護雲端伺服器記憶體不崩潰
N = st.sidebar.slider("模擬次數 (平行宇宙)", min_value=1000,
                      max_value=10000, value=5000, step=1000)
st.sidebar.divider()

# 模組 3: 引擎專屬參數
st.sidebar.header("📈 市場環境設定")
if "歷史" in engine:
    ticker = st.sidebar.text_input("輸入歷史標的代碼 (Yahoo Finance)", value="0050.TW")

    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("開始日期", value=datetime(2008, 1, 1))
    end_date = col2.date_input("結束日期", value=datetime(2013, 12, 31))

    block_size = st.sidebar.slider(
        "區塊大小 (歷史連續天數)", min_value=5, max_value=60, value=21, help="每次抽樣保留幾天的連續歷史波動")

else:
    mu_base = st.sidebar.number_input("基準標的 預期年化報酬率 (%)", value=9.0) / 100
    sig_base = st.sidebar.number_input("基準標的 年化波動率 (%)", value=16.0) / 100
    df_t = st.sidebar.slider(
        "肥尾效應強度 (t分配自由度)", min_value=2, max_value=30, value=3, help="數值越小，極端黑天鵝事件越多")
st.sidebar.divider()

# 模組 4: 策略細節微調
st.sidebar.header("🛠️ 槓桿與策略微調")
lev_mult = st.sidebar.number_input(
    "槓桿倍數", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
drag_annual = st.sidebar.slider(
    "槓桿標的 額外年化耗損 (%)", min_value=0.0, max_value=10.0, value=1.5, step=0.1) / 100

st.sidebar.caption("策略 5：跌深抄底設定")
drop_threshold = st.sidebar.slider(
    "自高點回檔觸發比例 (%)", min_value=5, max_value=50, value=20) / 100
transfer_pct = st.sidebar.slider(
    "觸發時轉入槓桿標的資金比例 (%)", min_value=10, max_value=100, value=20) / 100
st.sidebar.divider()

# 模組 5: 圖表顯示勾選
st.sidebar.header("👁️ 投資組合顯示設定")
show_s1 = st.sidebar.checkbox("1. 100% 槓桿標的", value=True)
show_s2 = st.sidebar.checkbox("2. 50% 現金 + 50% 槓桿 (買入持有)", value=True)
show_s3 = st.sidebar.checkbox("3. 50% 現金 + 50% 槓桿 (每年再平衡)", value=True)
show_s4 = st.sidebar.checkbox("4. 100% 基準標的", value=True)
show_s5 = st.sidebar.checkbox(
    f"5. 基準標的跌 {drop_threshold*100:.0f}% -> 轉 {transfer_pct*100:.0f}% 至槓桿", value=True)

# ==========================================
# 資料下載快取 (僅針對歷史引擎)
# ==========================================


@st.cache_data(show_spinner=False, ttl=3600)  # 快取 1 小時，避免頻繁請求被鎖
def get_hist_data(tkr, start, end):
    import time
    # 🌟 嘗試 3 次重試機制
    for i in range(3):
        try:
            # 強制設定為 auto_adjust=True 讓 Yahoo 直接給我們還原權息的資料
            data = yf.download(tkr, start=start, end=end,
                               progress=False, auto_adjust=True)

            if not data.empty and len(data) > 5:
                # 偵測欄位：auto_adjust=True 會讓欄位直接變成 'Close'
                # 這裡要小心 Multi-index 的問題
                if isinstance(data.columns, pd.MultiIndex):
                    # 如果是多層標題，取第一層是 Close 的那一欄
                    returns = data['Close'].iloc[:, 0].pct_change().dropna()
                else:
                    returns = data['Close'].pct_change().dropna()

                return returns.values.flatten()
        except Exception as e:
            st.sidebar.warning(f"第 {i+1} 次連線嘗試失敗... 正在重試")
            time.sleep(1)  # 停一秒再試

    return None


# ==========================================
# 核心運算區塊
# ==========================================
if st.sidebar.button("🚀 開始模擬運算", type="primary", use_container_width=True):
    with st.spinner('⚙️ 正在建構平行宇宙與運算策略...'):

        days_per_year = 252
        total_days = sim_years * days_per_year
        dt = 1 / days_per_year
        cash_growth = np.exp(0.01 * dt)

        # 準備回報率矩陣
        sim_ret_base = np.zeros((total_days, N))
        sim_ret_lev = np.zeros((total_days, N))

        daily_drag = drag_annual / days_per_year

        # 根據引擎產生走勢
        if "歷史" in engine:
            hist_ret = get_hist_data(ticker, start_date, end_date)
            if hist_ret is None or len(hist_ret) < block_size:
                st.error("❌ 無法下載歷史資料，或所選日期區間太短（需大於區塊大小）。請檢查代碼或日期。")
                st.stop()

            hist_ret_lev = (hist_ret * lev_mult) - daily_drag

            blocks_per_path = int(np.ceil(total_days / block_size))
            max_start_idx = len(hist_ret) - block_size

            random_starts = np.random.randint(
                0, max_start_idx, size=(blocks_per_path, N))

            for b in range(blocks_per_path):
                starts = random_starts[b, :]
                for i in range(block_size):
                    day_idx = b * block_size + i
                    if day_idx < total_days:
                        sim_ret_base[day_idx, :] = hist_ret[starts + i]
                        sim_ret_lev[day_idx, :] = hist_ret_lev[starts + i]
        else:
            # GBM 引擎
            Z = np.random.standard_t(
                df=df_t, size=(total_days, N)) * np.sqrt(1/3)

            # 美股專用防爆機制：容許最大 +/- 25 個標準差的黑天鵝
            Z = np.clip(Z, -25, 25)

            mu_lev = mu_base * lev_mult
            sig_lev = sig_base * lev_mult

            # 使用對數報酬近似
            log_ret_base = (mu_base - 0.5 * sig_base**2) * \
                dt + sig_base * np.sqrt(dt) * Z
            log_ret_lev = (mu_lev - 0.5 * sig_lev**2) * dt + \
                sig_lev * np.sqrt(dt) * Z - daily_drag

            # 轉換回真實每日報酬率
            sim_ret_base = np.exp(log_ret_base) - 1
            sim_ret_lev = np.exp(log_ret_lev) - 1

        # 🌟 修正點 2：加入「破產防線」，強制資產乘數不可小於 0 (最多歸零)
        sim_mult_base = np.maximum(0, 1 + sim_ret_base)
        sim_mult_lev = np.maximum(0, 1 + sim_ret_lev)

        price_base = np.cumprod(sim_mult_base, axis=0)
        price_base = np.vstack([np.ones(N), price_base])

        # 初始化策略資金
        V1 = np.ones(N) * initial_capital
        V2_c, V2_L = np.ones(N) * initial_capital * \
            0.5, np.ones(N) * initial_capital * 0.5
        V3_c, V3_L = np.ones(N) * initial_capital * \
            0.5, np.ones(N) * initial_capital * 0.5
        V4 = np.ones(N) * initial_capital
        V5_B, V5_L = np.ones(N) * initial_capital, np.zeros(N)

        price_ATH_base = np.ones(N)
        triggered = np.zeros(N, dtype=bool)

        # 逐日結算
        for d in range(1, total_days + 1):
            r_B = sim_mult_base[d-1]
            r_L = sim_mult_lev[d-1]

            V1 *= r_L
            V4 *= r_B
            V2_c *= cash_growth
            V2_L *= r_L
            V3_c *= cash_growth
            V3_L *= r_L

            if d % days_per_year == 0:
                tot_V3 = V3_c + V3_L
                V3_c, V3_L = tot_V3 * 0.5, tot_V3 * 0.5

            V5_B *= r_B
            V5_L *= r_L
            price_ATH_base = np.maximum(price_ATH_base, price_base[d])

            # 🌟 修正點 3：加入 np.errstate 避免資產歸零時產生「除以 0」的運算錯誤
            with np.errstate(divide='ignore', invalid='ignore'):
                drawdown = np.where(price_ATH_base > 0,
                                    price_base[d] / price_ATH_base, 0)

            triggered[drawdown == 1.0] = False
            cond = (drawdown <= (1.0 - drop_threshold)) & (~triggered)

            if np.any(cond):
                transfer = V5_B[cond] * transfer_pct
                V5_B[cond] -= transfer
                V5_L[cond] += transfer
                triggered[cond] = True

        # 篩選要顯示的結果
        all_results = {}
        if show_s1:
            all_results['1. 100% 槓桿'] = V1
        if show_s2:
            all_results['2. 50/50 (Hold)'] = V2_c + V2_L
        if show_s3:
            all_results['3. 50/50 (Rebalance)'] = V3_c + V3_L
        if show_s4:
            all_results['4. 100% 基準標的'] = V4
        if show_s5:
            all_results['5. 跌深抄底策略'] = V5_B + V5_L

        if not all_results:
            st.warning("⚠️ 請至少在左側勾選一種投資組合來顯示結果。")
            st.stop()

        df_res = pd.DataFrame(all_results)

    # ==========================================
    # 產出報表與動態圖表
    # ==========================================
    st.success(f"✅ 成功完成 {N:,} 次平行宇宙模擬！")

    # 統計表
    stats = []
    for col in df_res.columns:
        data = df_res[col]
        win_rate = (data > initial_capital).mean() * 100
        stats.append({
            '投資策略': col,
            '獲勝機率 (%)': f"{win_rate:.2f}%",
            '平均終值': f"{data.mean():,.0f}",
            '中位數': f"{data.median():,.0f}",
            '悲觀 5% (左尾)': f"{np.percentile(data, 5):,.0f}",
            '樂觀 5% (右尾)': f"{np.percentile(data, 95):,.0f}"
        })

    st.subheader(f"📊 {sim_years} 年後資產績效統計")
    st.dataframe(pd.DataFrame(stats), use_container_width=True)

    # KDE 圖表
    st.subheader("📈 終值分佈密度圖 (KDE Plot)")
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in df_res.columns:
        sns.kdeplot(df_res[col], ax=ax, label=col,
                    fill=True, alpha=0.15, linewidth=2)

    ax.axvline(initial_capital, color='red', linestyle='--',
               label='Initial Capital', zorder=10)

    title_prefix = "Historical Block Bootstrapping" if "歷史" in engine else "GBM Fat-Tail Model"
    ax.set_title(
        f'{sim_years}-Year Final Asset Distribution ({title_prefix})', fontsize=14)
    ax.set_xlabel('Final Asset Value (TWD)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)

    # 動態調整 X 軸範圍以避免被極端樂觀值拉扁
    x_max = np.percentile(df_res.values, 95) * 1.5
    ax.set_xlim(0, max(x_max, initial_capital * 2))

    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

else:
    st.info("👈 請在左側設定各項參數，調整完畢後點擊「開始模擬運算」按鈕。")
