import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ETF Momentum Rebalancer", layout="wide")

DEFAULT_ETFS = "SPY, QQQ, GLD, TLT, VNQ, EEM, IWM"

# -----------------------------------------------------------------------------
# DATA FETCHING FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_data_eodhd(api_key, tickers, start_date, end_date):
    """
    Fetches adjusted close data from EOD Historical Data API.
    """
    data = {}
    
    # EODHD expects specific format usually, but we'll try standard Ticker.Exchange 
    # If user provides just 'SPY', we default to 'US'.
    
    with st.spinner('Fetching data from EOD Historical Data API...'):
        for ticker in tickers:
            clean_ticker = ticker.strip().upper()
            if "." not in clean_ticker:
                clean_ticker = f"{clean_ticker}.US"
                
            url = f"https://eodhistoricaldata.com/api/eod/{clean_ticker}"
            params = {
                'api_token': api_key,
                'from': start_date,
                'to': end_date,
                'fmt': 'json',
                'period': 'd'
            }
            
            try:
                r = requests.get(url, params=params)
                r.raise_for_status()
                json_data = r.json()
                
                if isinstance(json_data, list) and len(json_data) > 0:
                    df = pd.DataFrame(json_data)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    # Use adjusted_close if available, else close
                    price_col = 'adjusted_close' if 'adjusted_close' in df.columns else 'close'
                    data[ticker.strip().upper()] = df[price_col]
                else:
                    st.warning(f"No data found for {clean_ticker}")
            except Exception as e:
                st.error(f"Error fetching {clean_ticker}: {e}")
                
    if not data:
        return pd.DataFrame()
        
    prices = pd.DataFrame(data)
    prices.sort_index(inplace=True)
    return prices.dropna()

@st.cache_data(ttl=3600)
def get_data_yfinance(tickers, start_date, end_date):
    """
    Fetches adjusted close data from Yahoo Finance (Fallback).
    """
    with st.spinner('Fetching data from Yahoo Finance...'):
        try:
            df = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
            df.sort_index(inplace=True)
            return df.dropna()
        except Exception as e:
            st.error(f"Error fetching YFinance data: {e}")
            return pd.DataFrame()

# -----------------------------------------------------------------------------
# STRATEGY LOGIC
# -----------------------------------------------------------------------------

def calculate_metrics(prices, lookback_days=252):
    """
    Calculates the Return/Volatility ratio for the last `lookback_days`.
    Metric = (Price_t / Price_{t-lookback} - 1) / (StdDev(daily_rets) * sqrt(252))
    """
    # 1. Total Return over lookback
    total_return = prices.pct_change(lookback_days)
    
    # 2. Volatility (Annualized Standard Deviation)
    daily_returns = prices.pct_change()
    # Rolling standard deviation over the lookback window
    rolling_std = daily_returns.rolling(window=lookback_days).std()
    annualized_vol = rolling_std * np.sqrt(252)
    
    # 3. Ratio
    # Avoid division by zero
    ratio = total_return / annualized_vol.replace(0, np.nan)
    
    return ratio

def run_backtest(prices, top_n=2, rebalance_freq='QE'):
    """
    Runs the backtest.
    rebalance_freq: 'QE' for Quarter End (Pandas alias).
    """
    # Calculate daily returns for the underlying assets
    asset_returns = prices.pct_change().dropna()
    
    # Calculate the ranking metric (Return / Volatility)
    # We use a 12-month (approx 252 trading days) lookback
    metrics = calculate_metrics(prices, lookback_days=252)
    
    # Create a schedule of rebalance dates
    # We can only rebalance if we have metric data (so after the first year)
    valid_dates = metrics.dropna(how='all').index
    if len(valid_dates) == 0:
        return None, None
    
    start_date = valid_dates[0]
    
    # Resample to rebalancing frequency (e.g., Quarterly)
    # We take the last available business day of the quarter
    rebalance_dates = prices.loc[start_date:].resample(rebalance_freq).last().index
    
    # To store portfolio daily returns
    portfolio_returns = pd.Series(0.0, index=asset_returns.index)
    
    # To store historical allocations for visualization
    allocations_history = []
    
    # Iterate through rebalance periods
    # We decide allocations at 'date', apply them from 'date' + 1 day to 'next_date'
    current_weights = pd.Series(0.0, index=prices.columns)
    
    # Initial setup: Find the first valid rebalance date
    # We need to ensure the rebalance date exists in our trading calendar
    # If a quarter ends on a Saturday, we need the Friday before.
    # The 'resample().last()' usually handles this if using data index, 
    # but let's align strictly with available data points.
    
    trading_days = prices.index
    
    for i in range(len(rebalance_dates) - 1):
        curr_date = rebalance_dates[i]
        next_date = rebalance_dates[i+1]
        
        # Find the closest trading day <= curr_date (Decision Day)
        loc_idx = trading_days.searchsorted(curr_date)
        if loc_idx >= len(trading_days) or trading_days[loc_idx] > curr_date:
            loc_idx -= 1
        decision_day = trading_days[loc_idx]
        
        # Get metrics for decision day
        if decision_day not in metrics.index:
            continue
            
        day_metrics = metrics.loc[decision_day]
        
        # Select Top N
        top_assets = day_metrics.nlargest(top_n).index.tolist()
        
        # Determine Weights (Equal Weight)
        weight = 1.0 / len(top_assets) if len(top_assets) > 0 else 0
        
        # Record allocation
        alloc_record = {'Date': decision_day}
        for col in prices.columns:
            alloc_record[col] = weight if col in top_assets else 0.0
        allocations_history.append(alloc_record)
        
        # Apply Returns for the holding period (Next Day -> Next Rebalance Day)
        # Find indices for the holding period
        mask = (asset_returns.index > decision_day) & (asset_returns.index <= next_date)
        
        if len(top_assets) > 0:
            # Calculate mean return of selected assets for the period
            period_returns = asset_returns.loc[mask, top_assets].mean(axis=1)
            portfolio_returns.loc[mask] = period_returns
            
    # Handle the final period (from last rebalance to today)
    last_rb_date = rebalance_dates[-1]
    
    # Logic repeats for the final segment
    loc_idx = trading_days.searchsorted(last_rb_date)
    if loc_idx >= len(trading_days) or trading_days[loc_idx] > last_rb_date:
        loc_idx -= 1
    last_decision_day = trading_days[loc_idx]
    
    if last_decision_day in metrics.index:
        day_metrics = metrics.loc[last_decision_day]
        top_assets = day_metrics.nlargest(top_n).index.tolist()
        
        # Record final allocation
        weight = 1.0 / len(top_assets) if len(top_assets) > 0 else 0
        alloc_record = {'Date': last_decision_day}
        for col in prices.columns:
            alloc_record[col] = weight if col in top_assets else 0.0
        allocations_history.append(alloc_record)
        
        mask = (asset_returns.index > last_decision_day)
        if len(top_assets) > 0:
            period_returns = asset_returns.loc[mask, top_assets].mean(axis=1)
            portfolio_returns.loc[mask] = period_returns

    # Construct Equity Curve
    portfolio_equity = (1 + portfolio_returns).cumprod()
    portfolio_equity = portfolio_equity / portfolio_equity.iloc[0] * 100 # Start at 100
    
    # Allocations DataFrame
    alloc_df = pd.DataFrame(allocations_history)
    if not alloc_df.empty:
        alloc_df.set_index('Date', inplace=True)
    
    return portfolio_equity, alloc_df

# -----------------------------------------------------------------------------
# MAIN APP UI
# -----------------------------------------------------------------------------

st.title("📊 Adaptive ETF Rebalancing Strategy")
st.markdown("""
This strategy rebalances every **3 months**. It looks back at the last **12 months** and selects the top **2 ETFs** with the highest **Return / Volatility** ratio.
""")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

data_source = st.sidebar.selectbox("Data Source", ["EOD Historical Data API", "Yahoo Finance (Free/Mock)"])

api_key = ""
if data_source == "EOD Historical Data API":
    api_key = st.sidebar.text_input("EODHD API Key", type="password")
    st.sidebar.info("Don't have a key? Switch to Yahoo Finance to test.")

ticker_input = st.sidebar.text_area("ETF Universe (Comma Separated)", DEFAULT_ETFS)
tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]

years_back = st.sidebar.slider("Backtest Duration (Years)", 5, 25, 20)
start_date_str = (datetime.now() - timedelta(days=years_back*365)).strftime('%Y-%m-%d')
end_date_str = datetime.now().strftime('%Y-%m-%d')

run_btn = st.sidebar.button("Run Strategy", type="primary")

# --- Main Execution ---

if run_btn:
    if not tickers:
        st.error("Please enter at least one ETF ticker.")
    elif data_source == "EOD Historical Data API" and not api_key:
        st.error("Please enter your EODHD API Key.")
    else:
        # 1. Fetch Data
        if data_source == "EOD Historical Data API":
            prices = get_data_eodhd(api_key, tickers, start_date_str, end_date_str)
        else:
            prices = get_data_yfinance(tickers, start_date_str, end_date_str)
            
        if prices.empty:
            st.error("No data returned. Please check tickers or API limits.")
        else:
            st.success(f"Data loaded for {len(prices.columns)} assets from {prices.index[0].date()} to {prices.index[-1].date()}")
            
            # 2. Run Backtest
            equity_curve, allocations = run_backtest(prices, top_n=2, rebalance_freq='QE')
            
            if equity_curve is None:
                st.warning("Not enough data to calculate initial metrics (Need > 12 months history).")
            else:
                # 3. Calculate Stats
                total_return = (equity_curve.iloc[-1] - 100)
                cagr = (equity_curve.iloc[-1] / 100) ** (1 / years_back) - 1
                
                # Drawdown
                rolling_max = equity_curve.cummax()
                drawdown = (equity_curve - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100

                # 4. Display KPIs
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Return", f"{total_return:.2f}%")
                col2.metric("CAGR", f"{cagr:.2%}")
                col3.metric("Max Drawdown", f"{max_drawdown:.2f}%")

                # 5. Charts
                st.subheader("Strategy Performance (Base 100)")
                
                # Compare vs Equal Weight Buy & Hold of the Universe
                benchmark_ret = prices.pct_change().mean(axis=1)
                benchmark_equity = (1 + benchmark_ret).cumprod()
                # Align benchmark start to strategy start
                strat_start = equity_curve.index[0]
                benchmark_equity = benchmark_equity.loc[strat_start:] 
                benchmark_equity = benchmark_equity / benchmark_equity.iloc[0] * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, name="Strategy (Top 2)", line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=benchmark_equity.index, y=benchmark_equity, name="Equal Weight Universe (Benchmark)", line=dict(color='gray', dash='dot')))
                
                fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Drawdown Analysis")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", fill='tozeroy', line=dict(color='red')))
                fig_dd.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_dd, use_container_width=True)

                # 6. Current Signals
                st.divider()
                st.subheader("📢 Current Signals & Allocation")
                
                if not allocations.empty:
                    last_alloc = allocations.iloc[-1]
                    last_rebalance_date = allocations.index[-1]
                    
                    st.write(f"**Last Rebalance Date:** {last_rebalance_date.date()}")
                    st.write("Hold the following positions until the next quarter:")
                    
                    # Filter for non-zero weights
                    current_holdings = last_alloc[last_alloc > 0]
                    
                    # Create nice dataframe for display
                    display_df = pd.DataFrame({
                        "Ticker": current_holdings.index,
                        "Weight": [f"{w*100:.0f}%" for w in current_holdings.values]
                    })
                    
                    st.table(display_df)
                    
                    # Show latest metrics for context
                    st.write("### Latest Metrics (Last 12 Months)")
                    latest_metrics = calculate_metrics(prices).iloc[-1].sort_values(ascending=False)
                    st.dataframe(latest_metrics.rename("Return/Vol Ratio").head(10))
                else:
                    st.write("No allocations generated yet.")
