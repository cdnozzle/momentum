import streamlit as st
import pandas as pd
import requests
import datetime
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="ETF Momentum Rebalancer", layout="wide")

st.title("ETF Momentum Rebalancing Strategy")
st.markdown("""
This tool backtests a momentum strategy that rebalances every **6 months** based on the **last 6 months' returns**.
It selects the **Top 2** performing ETFs from your list and holds them for the next period.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

# API Key Input
api_token = st.sidebar.text_input("EODHD API Token", value="demo", help="Enter your EOD Historical Data API Token. Use 'demo' for testing (limited tickers only).")

# ETF List Input
default_etfs = "SPY,QQQ,IWM,EEM,TLT,GLD,VNQ,LQD"
etf_input = st.sidebar.text_area("ETF Tickers (comma separated)", value=default_etfs, height=100)
tickers = [t.strip().upper() for t in etf_input.split(",") if t.strip()]

# Date Range
end_date = datetime.date.today()
start_date = end_date - relativedelta(years=20)

run_backtest = st.sidebar.button("Run Backtest", type="primary")

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_historical_data(symbol, api_token, start, end):
    """Fetches historical adjusted close data from EODHD."""
    url = f"https://eodhd.com/api/eod/{symbol}"
    params = {
        'api_token': api_token,
        'fmt': 'json',
        'from': start.strftime('%Y-%m-%d'),
        'to': end.strftime('%Y-%m-%d'),
        'period': 'd'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data or not isinstance(data, list):
            return None
            
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Use adjusted_close if available, else close
        if 'adjusted_close' in df.columns:
            return df['adjusted_close']
        else:
            return df['close']
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return None

def calculate_momentum_strategy(prices_df, top_n=2):
    """
    Executes the rebalancing strategy.
    
    Logic:
    1. Resample to Monthly to find month-ends.
    2. Filter for 6-month intervals (June/Dec or similar depending on start).
    3. Calculate 6-month lookback returns.
    4. Select Top N.
    5. Construct daily portfolio returns.
    """
    
    # Resample to business month end to get clean rebalance dates
    monthly_prices = prices_df.resample('ME').last()
    
    # Calculate 6-month returns for ranking
    lookback_returns = monthly_prices.pct_change(6)
    
    # Initialize portfolio tracking
    portfolio_daily_returns = pd.Series(0.0, index=prices_df.index)
    allocations = pd.DataFrame(index=lookback_returns.index, columns=prices_df.columns)
    
    # Track rebalance events for display
    rebalance_log = []
    
    # Determine rebalance dates: Every 6 months starting from the first valid lookback
    # We skip the first 6 months as we need data to calculate momentum
    valid_dates = lookback_returns.dropna(how='all').index
    
    # We will rebalance on these dates
    # Filter to ensure 6 month gaps roughly
    rebalance_dates = []
    if len(valid_dates) > 0:
        current = valid_dates[0]
        while current <= valid_dates[-1]:
            rebalance_dates.append(current)
            current += relativedelta(months=6)
            
    # Iterate through time
    # For each period between rebalance_date[i] and rebalance_date[i+1]
    
    current_holdings = []
    
    for i in range(len(rebalance_dates)):
        date = rebalance_dates[i]
        
        # 1. Selection Step: Get returns up to this date
        if date not in lookback_returns.index:
            # Fallback if specific date missing, find nearest previous
            try:
                idx = lookback_returns.index.get_indexer([date], method='ffill')[0]
                date = lookback_returns.index[idx]
            except:
                continue
                
        # Get momentum scores (returns) for this date
        scores = lookback_returns.loc[date]
        
        # Rank and select Top N, ignoring NaNs (assets not yet listed)
        valid_scores = scores.dropna()
        if valid_scores.empty:
            current_holdings = []
        else:
            top_performers = valid_scores.nlargest(top_n).index.tolist()
            current_holdings = top_performers
        
        # Log decision
        rebalance_log.append({
            'Date': date.date(),
            'Selected': ", ".join(current_holdings),
            'Returns_Last_6m': [f"{scores[t]:.1%}" for t in current_holdings]
        })
        
        # 2. Performance Step: Apply holdings to the *next* period
        # Period start: day after rebalance date
        # Period end: next rebalance date
        
        start_period = date
        if i < len(rebalance_dates) - 1:
            end_period = rebalance_dates[i+1]
        else:
            end_period = prices_df.index[-1]
            
        # Get daily returns for this specific period
        period_mask = (prices_df.index > start_period) & (prices_df.index <= end_period)
        
        if not current_holdings:
            # Cash position if no holdings
            portfolio_daily_returns.loc[period_mask] = 0.0
        else:
            # Equal weight among selected
            # Calculate mean return of selected assets
            asset_returns = prices_df.loc[period_mask, current_holdings].pct_change()
            
            # Simple average return (rebalanced daily or buy-and-hold logic?)
            # Standard backtest approximation: Average of daily returns (rebalanced daily assumption)
            # OR Buy-and-hold over the period. 
            # For simplicity and robustness in pandas, mean of daily returns is standard for "Equal Weight Portfolio"
            period_returns = asset_returns.mean(axis=1).fillna(0)
            portfolio_daily_returns.loc[period_mask] = period_returns

    return portfolio_daily_returns, pd.DataFrame(rebalance_log)

# --- Main Application Logic ---

if run_backtest:
    if not api_token:
        st.error("Please enter an API Token.")
    else:
        with st.spinner("Fetching data and crunching numbers..."):
            # 1. Fetch Data
            price_data = {}
            progress_bar = st.progress(0)
            
            for idx, ticker in enumerate(tickers):
                df_col = fetch_historical_data(ticker, api_token, start_date, end_date)
                if df_col is not None:
                    price_data[ticker] = df_col
                progress_bar.progress((idx + 1) / len(tickers))
            
            if not price_data:
                st.error("No data fetched. Check your API token and ticker symbols.")
            else:
                # Combine into one DataFrame
                prices_df = pd.DataFrame(price_data)
                prices_df.sort_index(inplace=True)
                prices_df.ffill(inplace=True) # Fill missing days (holidays etc)
                
                # 2. Run Strategy
                st.subheader("Backtest Results")
                
                # Strategy Returns
                strat_daily_rets, rebalance_log = calculate_momentum_strategy(prices_df, top_n=2)
                
                # Calculate Cumulative Returns
                strat_cum_rets = (1 + strat_daily_rets).cumprod()
                
                # Create a Benchmark (Equal Weight of all provided tickers)
                benchmark_daily = prices_df.pct_change().mean(axis=1).fillna(0)
                benchmark_cum = (1 + benchmark_daily).cumprod()
                
                # --- Metrics ---
                total_return = strat_cum_rets.iloc[-1] - 1
                years = (prices_df.index[-1] - prices_df.index[0]).days / 365.25
                cagr = (strat_cum_rets.iloc[-1])**(1/years) - 1
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Return", f"{total_return:.2%}")
                col2.metric("CAGR", f"{cagr:.2%}")
                
                # Calculate Max Drawdown
                rolling_max = strat_cum_rets.cummax()
                drawdown = strat_cum_rets / rolling_max - 1
                max_dd = drawdown.min()
                col3.metric("Max Drawdown", f"{max_dd:.2%}")

                # --- Charts ---
                chart_df = pd.DataFrame({
                    "Strategy": strat_cum_rets,
                    "Equal Weight Benchmark": benchmark_cum
                })
                
                fig = px.line(chart_df, title="Strategy vs Benchmark (Growth of $1)")
                fig.update_layout(hovermode="x unified", legend_title_text='Portfolio')
                st.plotly_chart(fig, use_container_width=True)

                # --- Current Signals ---
                st.divider()
                st.subheader("Current Signals (Based on latest data)")
                
                # Calculate momentum right now to see what we should be holding
                latest_prices = prices_df.resample('ME').last()
                if len(latest_prices) >= 7: # Need at least 6 months lookback
                    last_date = latest_prices.index[-1]
                    current_mom = latest_prices.pct_change(6).iloc[-1]
                    
                    # Create ranking table
                    mom_df = pd.DataFrame(current_mom).rename(columns={last_date: '6m Return'})
                    mom_df = mom_df.sort_values('6m Return', ascending=False)
                    
                    top_picks = mom_df.head(2)
                    
                    st.success(f"**Current Recommendation:** Buy **{', '.join(top_picks.index)}**")
                    
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.write("Top 2 Selection:")
                        st.table(top_picks.style.format("{:.2%}"))
                    with col_b:
                        st.write("Full Ranking:")
                        st.dataframe(mom_df.style.format("{:.2%}"), height=200)
                else:
                    st.warning("Not enough recent data to generate current signal.")

                # --- Historical Log ---
                with st.expander("View Historical Rebalance Log"):
                    st.dataframe(rebalance_log)

else:
    st.info("Enter your API Token and click 'Run Backtest' to start.")
    
    # Simple instructions for non-technical users
    st.markdown("""
    ### How to use:
    1.  **Get an API Key**: Sign up at [EOD Historical Data](https://eodhd.com/).
    2.  **Enter Tickers**: Input the ETF symbols you want to screen (e.g., SPY, QQQ, GLD).
    3.  **Run**: The system will download 20 years of daily data.
    4.  **Analyze**: See how the strategy of buying winners would have performed.
    """)
