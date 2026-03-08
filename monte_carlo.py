# ==========================================
    # 5. 產出報表與參數看板 (五大情境 + 年化報酬升級版)
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
    
    # 🌟 核心升級：計算 5 大分佈終值與年化報酬率 (CAGR)
    def calc_cagr(fv, pv, years):
        if fv <= 0: return -100.0
        return ((fv / pv) ** (1 / years) - 1) * 100

    stats = []
    for col in df_res_van.columns:
        d = df_res_van[col]
        win_rate = (d > bank_value_wan).mean() * 100
        
        # 取得該策略自己的 5 個關鍵百分位數
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
    
    title_prefix = "Historical Block Bootstrapping" if "歷史" in engine else "GBM Fat-Tail"
    ax.set_title(f'Monte Carlo Simulation: {sim_years}-Year Asset Distribution ({title_prefix})', fontsize=14)
    ax.set_xlabel('Final Asset Value (萬 TWD)', fontsize=12) 
    ax.set_ylabel('Density', fontsize=12)
    x_max = np.percentile(df_res_van.values, 95) * 1.5
    ax.set_xlim(0, max(x_max, actual_total_capital_wan * 2.5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
