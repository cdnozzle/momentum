# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 06:26:13 2025

@author: cdnoz
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# --- Configuration ---
st.set_page_config(page_title="Top 2 ETF Rotation Backtest", layout="wide")
st.title("Quarterly Volatility-Adjusted Rotation Strategy (Top 2)")

# Sidebar for API Key
st.sidebar.header("Data Settings")
api_key = st.sidebar.text_input("Enter EODHD API Token", type="password", help="Get your key from https://eodhd.com/")

TICKERS = ['XLF', 'QQQ', 'XLU', 'XLE', 'GLD', 'TLT', 'FXI', 'INDA', 'EWZ','XLU','EUFN','KRE','EWY']
BENCHMARK_SYMBOL = 'GSPC' # EODHD typically uses GSPC.INDX
BENCHMARK_LABEL = 'S&P 500'

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def fetch_data_eod(api_token, tickers, benchmark_sym, period_years=25):
    """
    Fetches historical data from EOD Historical Data API.
    Fetched 25 years to ensure 20 years of clean backtest data.
    """
    if not api_token:
        return pd.DataFrame()

    start_date = (datetime.now() - timedelta(days=period_years*365)).strftime('%Y-%m-%d')
    data_dict = {}
    
    progress_bar = st.progress(0)
    total_tickers = len(tickers) + 1
    
    # 1. Fetch ETFs
    for i, t in enumerate(tickers):
        # Construct EODHD URL for US ETFs
        url = f'https://eodhd.com/api/eod/{t}.US'
        params = {
            'api_token': api_token,
            'fmt': 'json',
            'from': start_date,
            'period': 'd'
        }
        
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                # Use adjusted_close for total return calculations
                data_dict[t] = df['adjusted_close'].astype(float)
        except Exception as e:
            st.error(f"Error fetching {t}: {e}")
            
        progress_bar.progress((i + 1) / total_tickers)

    # 2. Fetch Benchmark (GSPC.INDX)
    url_bench = f'https://eodhd.com/api/eod/{benchmark_sym}.INDX'
    params_bench = {
        'api_token': api_token,
        'fmt': 'json',
        'from': start_date,
        'period': 'd'
    }
    
    try:
        r_b = requests.get(url_bench, params=params_bench)
        r_b.raise_for_status()
        data_b = r_b.json()
        
        if isinstance(data_b, list) and len(data_b) > 0:
            df_b = pd.DataFrame(data_b)
            df_b['date'] = pd.to_datetime(df_b['date'])
            df_b.set_index('date', inplace=True)
            data_dict[BENCHMARK_LABEL] = df_b['adjusted_close'].astype(float)
    except Exception as e:
        st.warning(f"Could not fetch benchmark {benchmark_sym}: {e}")

    progress_bar.progress(1.0)
    
    # Combine into a single DataFrame
    combined_df = pd.DataFrame(data_dict)
    combined_df.sort_index(inplace=True)
    combined_df.ffill(inplace=True) # Forward fill missing data
    return combined_df

def calculate_metrics(prices):
    """
    Calculates annualized return, volatility, and 'Sharpe' (Return/Vol)
    based on the last 252 trading days.
    """
    if len(prices) < 2:
        return 0
        
    daily_rets = prices.pct_change().dropna()
    
    # Annualized Return (CAGR approximation for the window)
    total_ret = (prices.iloc[-1] / prices.iloc[0]) - 1
    # Simple annualized return for ranking
    ann_ret = total_ret 
    
    # Annualized Volatility
    ann_vol = daily_rets.std() * np.sqrt(252)
    
    # Volatility Adjusted Return
    if ann_vol == 0:
        return 0
    return ann_ret / ann_vol

def run_backtest(data):
    """
    Simulates the strategy selecting TOP 2 ETFs.
    """
    # Resample to business quarter end
    quarterly_dates = data.resample('Q').last().index
    
    history = []
    
    # Start 1 year after the first data point
    start_index = 0
    for i, date in enumerate(quarterly_dates):
        if date > data.index[0] + timedelta(days=365):
            start_index = i
            break
            
    portfolio_value = 100.0
    benchmark_value = 100.0
    daily_curve = []
    
    # Track holdings string for display
    current_holdings_str = 'Cash'
    
    # Iterate through quarters
    for i in range(start_index, len(quarterly_dates) - 1):
        curr_date = quarterly_dates[i]
        next_date = quarterly_dates[i+1]
        
        # 1. Lookback Window: 1 Year ending on current rebalance date
        lookback_start = curr_date - timedelta(days=365)
        
        # Get slice of data for ranking
        window_data = data.loc[lookback_start:curr_date, TICKERS]
        
        scores = {}
        for ticker in TICKERS:
            if ticker not in window_data.columns:
                scores[ticker] = -np.inf
                continue
                
            ticker_prices = window_data[ticker].dropna()
            if len(ticker_prices) > 200: 
                scores[ticker] = calculate_metrics(ticker_prices)
            else:
                scores[ticker] = -np.inf
        
        # Select Top 2 Winners
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Filter out invalid scores (-inf)
        valid_winners = [x[0] for x in sorted_scores if x[1] > -1000]
        
        selected_tickers = valid_winners[:2]
        
        # Record selection
        history.append({
            'Date': curr_date,
            'Selected': ", ".join(selected_tickers),
            'Top1_Score': scores[selected_tickers[0]] if len(selected_tickers) > 0 else 0,
            'Previous_Holdings': current_holdings_str
        })
        
        # 2. Simulate Performance for the NEXT Quarter
        # Setup data for simulation
        cols_needed = selected_tickers[:]
        if BENCHMARK_LABEL in data.columns:
            cols_needed.append(BENCHMARK_LABEL)
            
        period_data = data.loc[curr_date:next_date, cols_needed]
        
        if len(period_data) > 0 and len(selected_tickers) > 0:
            # Prepare returns dataframe
            # We calculate value of each "pocket" of the portfolio separately to allow drift
            
            # Init weights (Equal Weight)
            num_assets = len(selected_tickers)
            pocket_value = portfolio_value / num_assets
            pockets = {t: pocket_value for t in selected_tickers}
            
            period_returns = period_data.pct_change().fillna(0)
            
            for day, row in period_returns.iterrows():
                # Update Strategy Pockets
                daily_total = 0
                for t in selected_tickers:
                    ret = row[t]
                    pockets[t] *= (1 + ret)
                    daily_total += pockets[t]
                
                portfolio_value = daily_total
                
                # Update Benchmark
                b_ret = row[BENCHMARK_LABEL] if BENCHMARK_LABEL in row else 0
                benchmark_value *= (1 + b_ret)
                
                daily_curve.append({
                    'Date': day,
                    'Strategy': portfolio_value,
                    'Benchmark': benchmark_value,
                    'Holdings': ", ".join(selected_tickers)
                })
        elif len(period_data) > 0:
            # Fallback if no tickers selected (should be rare/impossible with this universe)
            # Stay in Cash
            period_returns = period_data.pct_change().fillna(0)
            for day, row in period_returns.iterrows():
                b_ret = row[BENCHMARK_LABEL] if BENCHMARK_LABEL in row else 0
                benchmark_value *= (1 + b_ret)
                daily_curve.append({
                    'Date': day,
                    'Strategy': portfolio_value,
                    'Benchmark': benchmark_value,
                    'Holdings': "Cash"
                })

        current_holdings_str = ", ".join(selected_tickers)

    return pd.DataFrame(history), pd.DataFrame(daily_curve)

# --- Main App Logic ---

if not api_key:
    st.warning("Please enter your EODHD API Token in the sidebar to load data.")
else:
    with st.spinner('Fetching 20+ years of data from EODHD...'):
        data = fetch_data_eod(api_key, TICKERS, BENCHMARK_SYMBOL)

    if not data.empty:
        # 1. Current Allocation Analysis (Live)
        st.header("📊 Current Allocation Signal (Top 2)")
        
        last_date = data.index[-1]
        one_year_ago = last_date - timedelta(days=365)
        current_window = data.loc[one_year_ago:last_date, TICKERS]
        
        current_metrics = []
        for t in TICKERS:
            if t in current_window.columns:
                prices = current_window[t].dropna()
                if len(prices) > 200:
                    ret = (prices.iloc[-1] / prices.iloc[0]) - 1
                    vol = prices.pct_change().std() * np.sqrt(252)
                    ratio = ret / vol if vol > 0 else 0
                    current_metrics.append({
                        'Ticker': t,
                        '1Y Return': ret,
                        '1Y Volatility': vol,
                        'Vol-Adj Return': ratio
                    })
        
        if current_metrics:
            df_curr = pd.DataFrame(current_metrics).sort_values('Vol-Adj Return', ascending=False)
            
            # Get Top 2
            top_2 = df_curr.head(2)
            
            # Display Top 2 Metrics side-by-side
            c1, c2 = st.columns(2)
            
            if len(top_2) >= 1:
                w1 = top_2.iloc[0]
                with c1:
                    st.info(f"🥇 First Allocation: {w1['Ticker']}")
                    st.metric("1Y Return", f"{w1['1Y Return']:.1%}", f"Score: {w1['Vol-Adj Return']:.2f}")

            if len(top_2) >= 2:
                w2 = top_2.iloc[1]
                with c2:
                    st.success(f"🥈 Second Allocation: {w2['Ticker']}")
                    st.metric("1Y Return", f"{w2['1Y Return']:.1%}", f"Score: {w2['Vol-Adj Return']:.2f}")

            with st.expander("See Full Rankings"):
                st.dataframe(df_curr.style.format({
                    '1Y Return': '{:.1%}',
                    '1Y Volatility': '{:.1%}',
                    'Vol-Adj Return': '{:.2f}'
                }))
        else:
            st.warning("Not enough data to calculate current metrics.")

        # 2. Backtest Execution
        st.divider()
        st.header(f"📈 20-Year Backtest vs {BENCHMARK_LABEL}")
        
        rebalance_log, equity_curve = run_backtest(data)
        
        if not equity_curve.empty:
            start_val = equity_curve.iloc[0]['Strategy']
            end_val = equity_curve.iloc[-1]['Strategy']
            total_ret = (end_val / start_val) - 1
            days = (equity_curve.iloc[-1]['Date'] - equity_curve.iloc[0]['Date']).days
            cagr = (end_val / start_val) ** (365/days) - 1
            
            # Max Drawdown
            equity_curve['Peak'] = equity_curve['Strategy'].cummax()
            equity_curve['Drawdown'] = (equity_curve['Strategy'] - equity_curve['Peak']) / equity_curve['Peak']
            max_dd = equity_curve['Drawdown'].min()
            
            # Benchmark Stats
            b_start = equity_curve.iloc[0]['Benchmark']
            b_end = equity_curve.iloc[-1]['Benchmark']
            b_total_ret = (b_end / b_start) - 1
            b_cagr = (b_end / b_start) ** (365/days) - 1
            
            # Stats Display
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Strategy Total Return", f"{total_ret:.1%}")
            kpi2.metric(f"{BENCHMARK_LABEL} Total Return", f"{b_total_ret:.1%}")
            kpi3.metric("Strategy CAGR", f"{cagr:.1%}")
            kpi4.metric("Max Drawdown", f"{max_dd:.1%}")
            
            # Plotting
            fig = px.line(equity_curve, x='Date', y=['Strategy', 'Benchmark'], 
                          title=f"Equity Curve (20 Years): Top 2 Rotation vs {BENCHMARK_LABEL}",
                          color_discrete_map={'Strategy': '#00CC96', 'Benchmark': '#636EFA'})
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. Holding History
            st.subheader("📜 Quarterly Rebalance Log")
            
            if not rebalance_log.empty:
                log_display = rebalance_log[['Date', 'Selected']].copy()
                log_display['Date'] = log_display['Date'].dt.date
                st.dataframe(log_display.sort_values('Date', ascending=False), height=400, use_container_width=True)
    else:
        st.error("No data returned. Please check your API key and try again.")