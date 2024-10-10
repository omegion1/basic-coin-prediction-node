import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import requests
import retrying
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from config import data_base_path

forecast_price = {}

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
MAX_DATA_SIZE = 100
INITIAL_FETCH_SIZE = 100

# Mô hình LSTM sử dụng PyTorch
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h_0 = Variable(torch.zeros(1, input_seq.size(1), self.hidden_layer_size))
        c_0 = Variable(torch.zeros(1, input_seq.size(1), self.hidden_layer_size))
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        predictions = self.linear(lstm_out[-1])
        return predictions

def download_data(token):
    symbols = f"{token.upper()}USDT"
    interval = "5m"
    current_datetime = datetime.now()
    download_path = os.path.join(binance_data_path, token.lower())
    
    # Đường dẫn file CSV để lưu trữ
    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")
    # file_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")

    # Kiểm tra xem file có tồn tại hay không
    if os.path.exists(file_path):
        # Tính thời gian bắt đầu cho 100 cây nến 5 phút
        start_time = int((current_datetime - timedelta(minutes=500)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, 100, start_time, end_time)
    else:
        # Nếu file không tồn tại, tải về số lượng INITIAL_FETCH_SIZE nến
        start_time = int((current_datetime - timedelta(minutes=INITIAL_FETCH_SIZE*5)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, INITIAL_FETCH_SIZE, start_time, end_time)

    # Chuyển dữ liệu thành DataFrame
    new_df = pd.DataFrame(new_data, columns=[
        "start_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])

    # Kiểm tra và đọc dữ liệu cũ nếu tồn tại
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        # Kết hợp dữ liệu cũ và mới
        combined_df = pd.concat([old_df, new_df])
        # Loại bỏ các bản ghi trùng lặp dựa trên 'start_time'
        combined_df = combined_df.drop_duplicates(subset=['start_time'], keep='last')
    else:
        combined_df = new_df

    # Giới hạn số lượng dữ liệu tối đa
    if len(combined_df) > MAX_DATA_SIZE:
        combined_df = combined_df.iloc[-MAX_DATA_SIZE:]

    # Lưu dữ liệu đã kết hợp vào file CSV
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    combined_df.to_csv(file_path, index=False)
    print(f"Updated data for {token} saved to {file_path}. Total rows: {len(combined_df)}")

def format_data(token):
    path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(path, f"{token.lower()}_5m_data.csv")

    if not os.path.exists(file_path):
        print(f"No data file found for {token}")
        return

    df = pd.read_csv(file_path)

    # Sử dụng các cột sau (đúng với dữ liệu bạn đã lưu)
    columns_to_use = [
        "start_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]

    # Kiểm tra nếu tất cả các cột cần thiết tồn tại trong DataFrame
    if set(columns_to_use).issubset(df.columns):
        df = df[columns_to_use]
        df.columns = [
            "start_time", "open", "high", "low", "close", "volume",
            "end_time", "quote_asset_volume", "n_trades", 
            "taker_volume", "taker_volume_usd"
        ]
        df.index = pd.to_datetime(df["start_time"], unit='ms')
        df.index.name = "date"

        output_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")
        df.sort_index().to_csv(output_path)
        print(f"Formatted data saved to {output_path}")
    else:
        print(f"Required columns are missing in {file_path}. Skipping this file.") 

def train_model(token):
    # Load the token price data
    price_data = pd.read_csv(os.path.join(data_base_path, f"{token.lower()}_price_data.csv"))

    # Preprocess the data
    df = pd.DataFrame()
    price_data["date"] = pd.to_datetime(price_data["date"])
    price_data.set_index("date", inplace=True)
    df = price_data.resample('10T').mean()
    df = df[['close']].dropna()

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df['close'] = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    # Prepare the training data
    train_data = df.values
    seq_length = 10
    x_train, y_train = [], []

    for i in range(seq_length, len(train_data)):
        x_train.append(train_data[i-seq_length:i, 0])
        y_train.append(train_data[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Reshape for LSTM [batch_size, sequence_length, input_dimension]
    x_train = torch.FloatTensor(x_train).view(-1, seq_length, 1)
    y_train = torch.FloatTensor(y_train)

    # Create LSTM model
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_function(y_pred.view(-1), y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    # Predict the next price
    model.eval()
    x_input = train_data[-seq_length:].reshape(-1, seq_length, 1)
    x_input = torch.FloatTensor(x_input)
    predicted_price = model(x_input).item()

    # Inverse transform to get the actual predicted price
    predicted_price = scaler.inverse_transform(np.array(predicted_price).reshape(-1, 1))[0][0]

    # Calculate fluctuation range and random forecast
    fluctuation_range = 0.001 * predicted_price
    min_price = predicted_price - fluctuation_range
    max_price = predicted_price + fluctuation_range

    price_predict = random.uniform(min_price, max_price)
    forecast_price[token] = price_predict

    print(f"Predicted_price: {predicted_price}, Min_price: {min_price}, Max_price: {max_price}")
    print(f"Forecasted price for {token}: {forecast_price[token]}")

def update_data():
    tokens = ["ETH", "BTC", "SOL"]
    for token in tokens:
        download_data(token)
        format_data(token)
        train_model(token)

if __name__ == "__main__":
    update_data()
