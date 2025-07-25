import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # For Supertrend calculation
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Streamlit App UI ---
st.title("📈 Dual Supertrend Trading Strategy Backtester (15-min Primary)")
st.markdown("Backtests a strategy using Supertrend on 15-minute and 60-minute timeframes for Gold Futures (GC=F).")
st.markdown(
    "Buy/Sell conditions: Current candle closes above/below 15m Supertrend AND current price is above/below 60m Supertrend.")

# --- Sidebar Configuration ---
st.sidebar.header("Strategy Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="GC=F")
period = st.sidebar.selectbox("Data Period", ["30d", "60d", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                              index=1)  # Default to 60d for enough data

st.sidebar.subheader("Supertrend Settings (7, 3.0)")
st_length = st.sidebar.number_input("Supertrend Length", value=7, min_value=1)
st_multiplier = st.sidebar.number_input("Supertrend Multiplier", value=3.0, min_value=0.1, format="%.1f")

st.sidebar.subheader("Risk Management")
sl_points = st.sidebar.number_input("Stop Loss (points)", value=5.0, min_value=0.1, format="%.1f")
tp_points = st.sidebar.number_input("Take Profit (points)", value=15.0, min_value=0.1, format="%.1f")
lookahead_candles = st.sidebar.number_input("Lookahead Candles (15m)", value=10, min_value=1, max_value=100,
                                            help="Number of 15-min candles to look for SL/TP hit after entry.")  # Adjusted default for 15m

# --- Step 1: Download Historical Data for both timeframes ---
st.subheader("Data Download")
st.info(f"Downloading {period} of 15-minute and 60-minute data for {ticker}...")

df_15m = pd.DataFrame()  # Primary dataframe (15-minute)
df_60m = pd.DataFrame()  # Secondary dataframe (60-minute)

try:
    # Download 15-minute data (Primary)
    df_15m = yf.download(ticker, interval="15m", period=period, auto_adjust=False)
    if df_15m.empty:
        st.error(f"Error: No 15-minute data downloaded for {ticker}. Please check ticker/period.")
        st.stop()

    if isinstance(df_15m.columns, pd.MultiIndex):
        df_15m.columns = df_15m.columns.droplevel(1)
    df_15m.reset_index(inplace=True)
    df_15m.rename(columns={'index': 'Datetime', 'Date': 'Datetime'}, inplace=True, errors='ignore')
    df_15m['Datetime'] = pd.to_datetime(df_15m['Datetime'])
    df_15m.set_index('Datetime', inplace=True)
    df_15m.dropna(inplace=True)
    st.success(f"Successfully downloaded {len(df_15m)} rows of 15-minute data.")

    # Download 60-minute data (Secondary)
    df_60m = yf.download(ticker, interval="60m", period=period, auto_adjust=False)
    if df_60m.empty:
        st.error(f"Error: No 60-minute data downloaded for {ticker}. Please check ticker/period.")
        st.stop()

    if isinstance(df_60m.columns, pd.MultiIndex):
        df_60m.columns = df_60m.columns.droplevel(1)
    df_60m.reset_index(inplace=True)
    df_60m.rename(columns={'index': 'Datetime', 'Date': 'Datetime'}, inplace=True, errors='ignore')
    df_60m['Datetime'] = pd.to_datetime(df_60m['Datetime'])
    df_60m.set_index('Datetime', inplace=True)
    df_60m.dropna(inplace=True)
    st.success(f"Successfully downloaded {len(df_60m)} rows of 60-minute data.")

except Exception as e:
    st.error(
        f"Failed to download data: {e}. Please check ticker, internet connection, or try a different period/interval.")
    st.stop()

# Ensure OHLC columns are numeric
for col in ['Open', 'High', 'Low', 'Close']:
    df_15m[col] = pd.to_numeric(df_15m[col], errors='coerce')
    df_60m[col] = pd.to_numeric(df_60m[col], errors='coerce')
df_15m.dropna(inplace=True)
df_60m.dropna(inplace=True)

if df_15m.empty or df_60m.empty:
    st.error("Error: DataFrames are empty after numeric conversion and NaN removal. Exiting.")
    st.stop()

# --- Step 2: Compute Supertrend Indicator for both timeframes ---
st.subheader("Supertrend Calculation")
with st.spinner(f"Calculating Supertrend ({st_length},{st_multiplier}) for both timeframes..."):
    # 15-minute Supertrend
    st_15m_data = ta.supertrend(df_15m['High'], df_15m['Low'], df_15m['Close'], length=st_length,
                                multiplier=st_multiplier)
    if st_15m_data is None or st_15m_data.empty:
        st.error("Error: 15-minute Supertrend calculation failed. Exiting.")
        st.stop()
    df_15m[f'SUPERT_{st_length}_{st_multiplier}'] = st_15m_data[f'SUPERT_{st_length}_{st_multiplier}']
    df_15m.dropna(subset=[f'SUPERT_{st_length}_{st_multiplier}'], inplace=True)

    # 60-minute Supertrend
    st_60m_data = ta.supertrend(df_60m['High'], df_60m['Low'], df_60m['Close'], length=st_length,
                                multiplier=st_multiplier)
    if st_60m_data is None or st_60m_data.empty:
        st.error("Error: 60-minute Supertrend calculation failed. Exiting.")
        st.stop()
    df_60m[f'SUPERT_{st_length}_{st_multiplier}'] = st_60m_data[f'SUPERT_{st_length}_{st_multiplier}']
    df_60m.dropna(subset=[f'SUPERT_{st_length}_{st_multiplier}'], inplace=True)
st.success("Supertrend calculations complete.")

# --- Step 3: Align 60-minute Supertrend to 15-minute timeframe ---
st.subheader("Data Alignment")
with st.spinner("Aligning 60-minute Supertrend to 15-minute timeframe..."):
    st_60m_resampled = df_60m[f'SUPERT_{st_length}_{st_multiplier}'].resample('15min').ffill()
    df_15m = df_15m.merge(st_60m_resampled.rename(f'SUPERT_60m_{st_length}_{st_multiplier}'),
                          left_index=True, right_index=True, how='inner')
    df_15m.dropna(inplace=True)

    if df_15m.empty:
        st.error("Error: DataFrame became empty after aligning Supertrend data. Adjust period or check data integrity.")
        st.stop()
st.success("Supertrend data aligned and prepared.")

# --- Step 4: Generate Buy/Sell Signals and Backtest ---
st.subheader("Backtesting Strategy")
trades = []
position = None  # 'Buy', 'Sell', or None
entry_price = 0.0
entry_time = None
entry_direction = None

with st.spinner("Running backtest simulation..."):
    for i in range(len(df_15m)):  # Iterate over the 15-minute DataFrame
        current_candle = df_15m.iloc[i]

        if pd.isna(current_candle['Close']) or \
                pd.isna(current_candle[f'SUPERT_{st_length}_{st_multiplier}']) or \
                pd.isna(current_candle[f'SUPERT_60m_{st_length}_{st_multiplier}']):  # Changed to 60m Supertrend
            continue

        close_15m = current_candle['Close']
        supertrend_15m = current_candle[f'SUPERT_{st_length}_{st_multiplier}']
        supertrend_60m = current_candle[f'SUPERT_60m_{st_length}_{st_multiplier}']  # Changed to 60m Supertrend

        # Buy Condition
        buy_condition = (close_15m > supertrend_15m) and (close_15m > supertrend_60m)

        # Sell Condition
        sell_condition = (close_15m < supertrend_15m) and (close_15m < supertrend_60m)

        if position is None:  # No open position, look for entry
            if buy_condition:
                position = 'Buy'
                entry_price = close_15m
                entry_time = current_candle.name  # Datetime index
                entry_direction = 'Buy'
            elif sell_condition:
                position = 'Sell'
                entry_price = close_15m
                entry_time = current_candle.name  # Datetime index
                entry_direction = 'Sell'

        # If a position is open, check for SL/TP or counter-signal
        # Skip exit check on the entry candle itself
        if position is not None and current_candle.name == entry_time:
            continue

        if position is not None:
            trade_closed = False
            exit_price = 0.0
            exit_time = None
            pnl = 0

            # Look ahead for SL/TP hit in the next LOOKAHEAD_CANDLES
            # Ensure there are enough candles for the lookahead slice
            if (i + 1 + lookahead_candles) > len(df_15m):  # Changed to df_15m
                lookahead_for_exit_df = df_15m.iloc[i + 1:].copy()  # Slice till end if not enough candles
            else:
                lookahead_for_exit_df = df_15m.iloc[i + 1: i + 1 + lookahead_candles].copy()  # Changed to df_15m

            # If no lookahead candles available, handle as a timeout at the very end of data
            if lookahead_for_exit_df.empty:
                # This specific case will be handled by the final open position check outside the loop
                continue

            for exit_idx, exit_row in lookahead_for_exit_df.iterrows():
                current_lookahead_low = float(exit_row['Low'])
                current_lookahead_high = float(exit_row['High'])

                if position == 'Buy':
                    sl_hit_price = entry_price - sl_points
                    tp_hit_price = entry_price + tp_points

                    if current_lookahead_low <= sl_hit_price:
                        exit_price = sl_hit_price
                        pnl = exit_price - entry_price
                        exit_time = exit_row.name
                        trade_closed = True
                        break
                    elif current_lookahead_high >= tp_hit_price:
                        exit_price = tp_hit_price
                        pnl = exit_price - entry_price
                        exit_time = exit_row.name
                        trade_closed = True
                        break

                elif position == 'Sell':
                    sl_hit_price = entry_price + sl_points
                    tp_hit_price = entry_price - tp_points

                    if current_lookahead_high >= sl_hit_price:
                        exit_price = sl_hit_price
                        pnl = entry_price - exit_price
                        exit_time = exit_row.name
                        trade_closed = True
                        break
                    elif current_lookahead_low <= tp_hit_price:
                        exit_price = tp_hit_price
                        pnl = entry_price - exit_price
                        exit_time = exit_row.name
                        trade_closed = True
                        break

            # If trade was closed within lookahead window by SL/TP
            if trade_closed:
                trades.append({
                    'Entry Time': entry_time,
                    'Entry Price': round(entry_price, 2),
                    'Direction': entry_direction,
                    'Exit Time': exit_time,
                    'Exit Price': round(exit_price, 2),
                    'PnL': round(pnl, 2),
                    'Result': 'Profit' if pnl > 0 else 'Loss'
                })
                position = None  # Reset position after closing trade
            else:
                # If not closed by SL/TP within lookahead, check for counter-signal on current candle
                # This is important to prevent holding positions indefinitely
                if (position == 'Buy' and sell_condition) or \
                        (position == 'Sell' and buy_condition):
                    exit_price = close_15m  # Close at current candle's close due to counter-signal
                    pnl = exit_price - entry_price if position == 'Buy' else entry_price - exit_price
                    exit_time = current_candle.name
                    trades.append({
                        'Entry Time': entry_time,
                        'Entry Price': round(entry_price, 2),
                        'Direction': entry_direction,
                        'Exit Time': exit_time,
                        'Exit Price': round(exit_price, 2),
                        'PnL': round(pnl, 2),
                        'Result': 'Counter-Signal Exit'
                    })
                    position = None  # Reset position after closing trade

                # If still open after counter-signal check and end of loop, it's a timeout
                # This handles the very last open position if no exit condition is met
                elif i == len(df_15m) - 1:  # If this is the last candle in the dataframe
                    exit_price = close_15m
                    pnl = exit_price - entry_price if position == 'Buy' else entry_price - close_15m  # Use close_15m for final PnL calc
                    exit_time = current_candle.name
                    trades.append({
                        'Entry Time': entry_time,
                        'Entry Price': round(entry_price, 2),
                        'Direction': entry_direction,
                        'Exit Time': exit_time,
                        'Exit Price': round(exit_price, 2),
                        'PnL': round(pnl, 2),
                        'Result': 'Timeout'
                    })
                    position = None  # Reset position

st.success("Backtest simulation complete.")

# --- Convert trades to DataFrame and display results ---
trades_df = pd.DataFrame(trades)

if trades_df.empty:
    st.warning("No trades were executed based on the defined strategy and parameters. Adjust parameters or data range.")
else:
    st.subheader("Backtest Summary")

    total_pnl = trades_df['PnL'].sum()
    profitable_trades = trades_df[trades_df['PnL'] > 0]
    losing_trades = trades_df[trades_df['PnL'] <= 0]

    # Calculate win rate for Profit vs Loss trades only
    wins = len(trades_df[trades_df['Result'] == 'Profit'])
    losses = len(trades_df[trades_df['Result'] == 'Loss'])
    total_win_loss_trades = wins + losses
    win_rate = (wins / total_win_loss_trades) * 100 if total_win_loss_trades > 0 else 0.0

    st.metric("Total Net PnL (points)", f"{total_pnl:.2f}")
    st.metric("Win Rate (Profit vs Loss)", f"{win_rate:.2f}%")
    st.write(f"Total Trades: {len(trades_df)}")
    st.write(f"Profitable Trades: {wins}")
    st.write(f"Losing Trades: {losses}")
    st.write(f"Counter-Signal Exits: {len(trades_df[trades_df['Result'] == 'Counter-Signal Exit'])}")
    st.write(f"Timeout Trades: {len(trades_df[trades_df['Result'] == 'Timeout'])}")

    st.subheader("Trade Log (Last 30 Trades)")
    st.dataframe(trades_df.tail(30))

    # --- CSV Export ---
    csv_export = trades_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Trades as CSV",
        data=csv_export,
        file_name="dual_supertrend_trades.csv",
        mime="text/csv",
    )

    # --- Trade Outcome Distribution Chart ---
    st.subheader("Trade Outcome Distribution")
    outcome_counts = trades_df['Result'].value_counts().reset_index()
    outcome_counts.columns = ['Outcome', 'Count']
    fig_outcome_pie = px.pie(outcome_counts,
                             values='Count',
                             names='Outcome',
                             title="Distribution of Trade Outcomes",
                             color_discrete_map={'Profit': 'green', 'Loss': 'red', 'Counter-Signal Exit': 'orange',
                                                 'Timeout': 'blue'})
    st.plotly_chart(fig_outcome_pie, use_container_width=True)

    # --- Cumulative PnL Chart ---
    trades_df['Cumulative PnL'] = trades_df['PnL'].cumsum()
    fig_cumulative_pnl = px.line(trades_df, x='Exit Time', y='Cumulative PnL',
                                 title="Cumulative PnL Over Time",
                                 labels={'Cumulative PnL': 'Cumulative PnL (points)'})
    st.plotly_chart(fig_cumulative_pnl, use_container_width=True)

    # --- Candlestick Chart with Supertrends ---
    st.subheader("Gold Price Candlestick Chart with Supertrends")
    fig_candlestick = go.Figure(data=[go.Candlestick(
        x=df_15m.index,  # Use the Datetime index for x-axis
        open=df_15m['Open'],
        high=df_15m['High'],
        low=df_15m['Low'],
        close=df_15m['Close'],
        name='15m Candles'  # Changed name
    )])

    # Add 15m Supertrend
    fig_candlestick.add_trace(go.Scatter(x=df_15m.index, y=df_15m[f'SUPERT_{st_length}_{st_multiplier}'],
                                         mode='lines', name=f'15m Supertrend ({st_length},{st_multiplier})',
                                         # Changed name
                                         line=dict(color='blue', width=1.5)))

    # Add 60m Supertrend (aligned to 15m timeframe)
    fig_candlestick.add_trace(
        go.Scatter(x=df_15m.index, y=df_15m[f'SUPERT_60m_{st_length}_{st_multiplier}'],  # Changed to 60m Supertrend
                   mode='lines', name=f'60m Supertrend ({st_length},{st_multiplier})',  # Changed name
                   line=dict(color='purple', width=1.5, dash='dot')))

    fig_candlestick.update_layout(xaxis_rangeslider_visible=False,
                                  title="Gold Price (15-min Candles with Supertrends)")  # Changed title
    st.plotly_chart(fig_candlestick, use_container_width=True)

    # --- PnL Distribution ---
    st.subheader("PnL Distribution")
    fig_pnl_hist = px.histogram(trades_df, x='PnL', nbins=20, title="Distribution of Trade PnL (points)")
    st.plotly_chart(fig_pnl_hist, use_container_width=True)

    # --- Daily Performance Analysis ---
    st.subheader("Daily Performance Analysis")
    if not trades_df.empty:
        # Ensure 'Exit Time' is datetime for day_name()
        trades_df['Exit Time'] = pd.to_datetime(trades_df['Exit Time'])
        trades_df['DayOfWeek'] = trades_df['Exit Time'].dt.day_name()

        # Calculate metrics per day of week
        daily_summary = trades_df.groupby('DayOfWeek').agg(
            Total_PnL=('PnL', 'sum'),
            Profitable_Trades=('Result', lambda x: (x == 'Profit').sum()),
            Losing_Trades=('Result', lambda x: (x == 'Loss').sum()),
            Timeout_Trades=('Result', lambda x: (x == 'Timeout').sum()),
            Counter_Signal_Exits=('Result', lambda x: (x == 'Counter-Signal Exit').sum())
        ).reset_index()

        # Calculate Win Rate for each day
        daily_summary['Total_Win_Loss_Trades'] = daily_summary['Profitable_Trades'] + daily_summary['Losing_Trades']
        daily_summary['Win_Rate (%)'] = (daily_summary['Profitable_Trades'] / daily_summary[
            'Total_Win_Loss_Trades']) * 100
        daily_summary['Win_Rate (%)'] = daily_summary['Win_Rate (%)'].fillna(0).round(2)  # Handle division by zero

        # Order days of the week for consistent display
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_summary['DayOfWeek'] = pd.Categorical(daily_summary['DayOfWeek'], categories=ordered_days, ordered=True)
        daily_summary = daily_summary.sort_values('DayOfWeek')

        # Display table
        st.dataframe(daily_summary.set_index('DayOfWeek'))

        # Find day with highest win rate
        if not daily_summary.empty:
            highest_win_rate_day = daily_summary.loc[daily_summary['Win_Rate (%)'].idxmax()]
            st.info(
                f"**Day with Highest Win Rate:** {highest_win_rate_day['DayOfWeek']} with {highest_win_rate_day['Win_Rate (%)']:.2f}%")
        else:
            st.info("No daily performance data to analyze.")
    else:
        st.info("No trades to analyze for daily performance.")
