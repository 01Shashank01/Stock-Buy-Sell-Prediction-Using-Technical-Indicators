def run_model(stock_name, interval, window):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    from alpha_vantage.timeseries import TimeSeries
    import time                                                                                                                                                          # User Inputs
    import os
    stock_name = stock_name
    interval = interval
    window=int(window)

    # Convert to Alpha Vantage format

    symbol = f"{stock_name}.BO"  # BSE format


    # Initialize Alpha Vantage API
    API_KEY = "LXPW27XHTU181BV3"
    ts = TimeSeries(key=API_KEY, output_format="pandas")

    def fetch_stock_data(ts, symbol, interval):
        for attempt in range(2):
            try:
                if interval == "1d":
                    data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")
                else:
                    data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize="full")
    
                if data.empty:
                    raise ValueError("No data received. Retrying...")
    
                return data  # ✅ Successfully fetched data
    
            except ValueError as e:
                print(f"⚠ API Error: {e} | Attempt {attempt + 1}/2")
                if attempt < 1:
                    time.sleep(60)  # Retry after delay
    
        # ❌ All attempts failed
        print("❌ Failed to retrieve data after multiple attempts.")
        return None
    # Call data fetcher
    data = fetch_stock_data(ts, symbol, interval)
                                                                                                                                                      # Rename columns
    data.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Print sample data
    print("✅ Downloaded Data:")
    print(data.head())

    # Save to CSV
    data.to_csv("data.csv")
    print("💾 Data saved to 'data.csv'.")

    # Load the saved data
    data = pd.read_csv("data.csv", index_col=0, parse_dates=True)

    # ✅ Sort data by date to ensure latest data is at the bottom
    data = data.sort_index(ascending=True)

    # ✅ Check first and last available dates
    print(f"📅 First Date in Dataset: {data.index[0]}")
    print(f"📅 Last Date in Dataset: {data.index[-1]}")  # Should be recent (2024-2025)


    # Convert necessary columns to numeric
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Step 5: Calculate indicators
    data['Daily Return'] = data['Close'].pct_change()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Step 6: Handle NaN values by forward filling
    data.fillna(method='ffill', inplace=True)

    # Step 7: Initialize scalers
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    # Step 8: Scale feature columns
    scaled_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaled_features = scaler_features.fit_transform(data[scaled_columns])

    # Step 9: Scale target column separately
    scaled_target = scaler_target.fit_transform(data[['Close']])

    # Step 10: Convert scaled features back into a DataFrame
    scaled_features_df = pd.DataFrame(scaled_features, columns=scaled_columns, index=data.index)

    # Step 11: Merge scaled features with calculated columns
    final_data = pd.concat([scaled_features_df, data[['Daily Return', 'SMA_20', 'SMA_50']]], axis=1)

    # Print final dataset
    print(final_data.tail())


     # Compute Moving Average and Standard Deviation


    data['SMA'] = data['Close'].rolling(window=window).mean()                                                                 # Step 1: Calculate MACD Components
    data['Short_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA
    data['Long_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()   # 26-day EMA
    data['MACD'] = data['Short_EMA'] - data['Long_EMA']  # MACD Line
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']  # MACD Histogram

    # Step 2: Compute RSI for momentum confirmation
    def calculate_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    data['RSI'] = calculate_rsi(data)

    # Step 3: Calculate ADX for trend strength
    def calculate_adx(data, window=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift(1))
        low_close = np.abs(data['Low'] - data['Close'].shift(1))
        tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum)

        data['ATR'] = tr.rolling(window=window).mean()

        data['+DM'] = np.where((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low']), 
                            data['High'] - data['High'].shift(1), 0)
        data['-DM'] = np.where((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1)), 
                            data['Low'].shift(1) - data['Low'], 0)

        data['+DI'] = 100 * (data['+DM'].ewm(span=window, adjust=False).mean() / data['ATR'])
        data['-DI'] = 100 * (data['-DM'].ewm(span=window, adjust=False).mean() / data['ATR'])

        data['DX'] = 100 * (np.abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI']))
        return data['DX'].ewm(span=window, adjust=False).mean()

    data['ADX'] = calculate_adx(data)

    # Step 4: Define MACD + ADX Trading Strategy
    data['MACD_Signal'] = 'Hold'

    for i in range(1, len(data)):
        # Buy Conditions:
        if (data['MACD'].iloc[i] > data['Signal_Line'].iloc[i]) and (data['MACD'].iloc[i-1] <= data['Signal_Line'].iloc[i-1]):
            if data['MACD'].iloc[i] > 0 and data['RSI'].iloc[i] > 70 and data['ADX'].iloc[i] > 25:  # Trend & momentum confirmation
                data.at[data.index[i], 'MACD_Signal'] = 'Buy'

        # Sell Conditions:
        elif (data['MACD'].iloc[i] < data['Signal_Line'].iloc[i]) and (data['MACD'].iloc[i-1] >= data['Signal_Line'].iloc[i-1]):
            if data['MACD'].iloc[i] < 0 and data['RSI'].iloc[i] < 70 and data['ADX'].iloc[i] > 25:  # Trend & momentum confirmation
                data.at[data.index[i], 'MACD_Signal'] = 'Sell'

    # Step 5: Plot MACD, ADX, and Buy/Sell Signals
    plt.figure(figsize=(14, 10))

    # Subplot 1: Close Price & Buy/Sell signals
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.scatter(data.index[data['MACD_Signal'] == 'Buy'], data['Close'][data['MACD_Signal'] == 'Buy'], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(data.index[data['MACD_Signal'] == 'Sell'], data['Close'][data['MACD_Signal'] == 'Sell'], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.title('Stock Price & MACD + ADX Trading Signals')
    plt.legend()

    # Subplot 2: MACD Indicator
    plt.subplot(3, 1, 2)
    plt.plot(data['MACD'], label='MACD', color='green')
    plt.plot(data['Signal_Line'], label='Signal Line', color='red')
    plt.bar(data.index, data['MACD_Histogram'], label='MACD Histogram', color='gray', alpha=0.3)
    plt.legend()

    # Subplot 3: ADX Indicator
    plt.subplot(3, 1, 3)
    plt.plot(data['ADX'], label='ADX', color='black')
    plt.axhline(y=25, color='gray', linestyle='--', label='Trend Strength Threshold')
    plt.legend()
    plt.title('ADX Indicator (Trend Strength)')

            # Create output folder if it doesn't exist
    output_dir = "static/images"
    os.makedirs(output_dir, exist_ok=True)

    # File name and full path
    filename1 = "chart1.png"
    full_path1 = os.path.join(output_dir, filename1)
    chart_paths=[]
    chart_paths.append(full_path1)

    # Save the chart
    plt.savefig(full_path1)

    # Print the latest MACD signal
    last_row = data.iloc[-1]
    string_text1=f"Latest MACD Signal: {last_row['MACD_Signal']}"  
    prediction_texts=[]
    prediction_texts.append(string_text1)
    # Step 1: Calculate Short-Term Support & Resistance
    window = 20  # 20-day window for short-term analysis
    data['Short_Resistance'] = data['High'].rolling(window=window).max()  # 20-day High
    data['Short_Support'] = data['Low'].rolling(window=window).min()  # 20-day Low

    # Step 2: Calculate Long-Term Support & Resistance (Pivot Points)
    data['Pivot'] = (data['High'].shift(1) + data['Low'].shift(1) + data['Close'].shift(1)) / 3
    data['Resistance1'] = (2 * data['Pivot']) - data['Low'].shift(1)
    data['Support1'] = (2 * data['Pivot']) - data['High'].shift(1)

    # Plotting Support & Resistance Levels
    plt.figure(figsize=(14, 7))

    # Plot Close Price
    plt.plot(data['Close'], label='Close Price', color='blue')

    # Plot Short-Term Support & Resistance
    plt.plot(data['Short_Support'], label='Short-Term Support (20-day Low)', linestyle='--', color='green')
    plt.plot(data['Short_Resistance'], label='Short-Term Resistance (20-day High)', linestyle='--', color='red')

    # Plot Long-Term Support & Resistance (Pivot Points)
    plt.plot(data['Support1'], label='Long-Term Support (Pivot)', linestyle=':', color='green')
    plt.plot(data['Resistance1'], label='Long-Term Resistance (Pivot)', linestyle=':', color='red')

    # Add Titles and Legends
    plt.title('Support & Resistance Levels')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
        # Create output folder if it doesn't exist
    output_dir = "static/images"
    os.makedirs(output_dir, exist_ok=True)

    # File name and full path
    filename2 = "chart2.png"
    full_path2 = os.path.join(output_dir, filename2)
    chart_paths.append(full_path2)
    # Save the chart
    plt.savefig(full_path2)

    # Find Recent Support & Resistance Levels
    recent_support = data['Short_Support'].iloc[-1]
    recent_resistance = data['Short_Resistance'].iloc[-1]
    long_term_support = data['Support1'].iloc[-1]
    long_term_resistance = data['Resistance1'].iloc[-1]

    string_text2=f"Recent Short-Term Support: {recent_support}"
    string_text3=f"Recent Short-Term Resistance: {recent_resistance}"
    string_text4=f"Recent Long-Term Support (Pivot): {long_term_support}"
    string_text5=f"Recent Long-Term Resistance (Pivot): {long_term_resistance}"
    prediction_texts.append(string_text2)
    prediction_texts.append(string_text3)
    prediction_texts.append(string_text4)
    prediction_texts.append(string_text5)
    data['STD'] = data['Close'].rolling(window=window).std() 

    # Compute Bollinger Bands
    data['Upper Band'] = data['SMA'] + (2 * data['STD'])
    data['Lower Band'] = data['SMA'] - (2 * data['STD'])

    # Plot the data
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['SMA'], label='SMA (20)', color='orange')
    plt.plot(data['Upper Band'], label='Upper Band', color='green')
    plt.plot(data['Lower Band'], label='Lower Band', color='red')
    plt.fill_between(data.index, data['Lower Band'], data['Upper Band'], color='grey', alpha=0.1)
    plt.title('Bollinger Bands')
    plt.legend()
    # Create output folder if it doesn't exist
    output_dir = "static/images"
    os.makedirs(output_dir, exist_ok=True)

    # File name and full path
    filename3 = "chart3.png"
    full_path3 = os.path.join(output_dir, filename3)
    chart_paths.append(full_path3)
    # Save the chart
    plt.savefig(full_path3)

    # Generate trading signals
    data['Signal'] = 'Hold'
    data.loc[data['Close'] < data['Lower Band'], 'Signal'] = 'Buy'  # Buy signal
    data.loc[data['Close'] > data['Upper Band'], 'Signal'] = 'Sell'  # Sell signal

    # Get the last row for the most recent prediction
    last_row = data.iloc[-1]
    last_date = data.index[-1]  # Use index instead of 'Date' column
    last_close = last_row['Close']
    last_signal = last_row['Signal']

    # Print the prediction in text format
    string_text6=f"Date: {last_date}"
    string_text7=f"Last Closing Price: {last_close}"
    string_text8=f"Prediction: {last_signal}"
    prediction_texts.append(string_text6)
    prediction_texts.append(string_text7)
    prediction_texts.append(string_text8)
    return {
    "success": True,
    "prediction": prediction_texts,  # list of strings
    "image_paths": chart_paths       # list of file paths (for <img> in HTML)
    
}
