import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import random

# Cấu hình đường dẫn dữ liệu
data_base_path = "./data"
binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
MAX_DATA_SIZE = 200  # Giới hạn kích thước dữ liệu tối đa
INITIAL_FETCH_SIZE = 100  # Số lượng dữ liệu khởi tạo khi tải ban đầu

# Định nghĩa mô hình LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.dropout = nn.Dropout(dropout)  # Thêm dropout

    def forward(self, input_seq):
        h_0 = Variable(torch.zeros(1, input_seq.size(0), self.hidden_layer_size))
        c_0 = Variable(torch.zeros(1, input_seq.size(0), self.hidden_layer_size))
        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        predictions = self.linear(self.dropout(lstm_out[:, -1, :]))  # Lấy đầu ra cuối cùng
        return predictions

# Hàm lưu mô hình
def save_model(model, token):
    model_path = os.path.join(data_base_path, f"{token.lower()}_lstm_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved for {token} at {model_path}")

# Hàm tải mô hình
def load_model(token):
    model = LSTM()
    model_path = os.path.join(data_base_path, f"{token.lower()}_lstm_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded for {token} from {model_path}")
    return model

# Hàm tải dữ liệu từ Binance và lưu trữ
def download_data(token):
    symbols = f"{token.upper()}USDT"
    interval = "5m"
    current_datetime = datetime.now()
    download_path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")

    # Giả lập việc tải dữ liệu từ Binance
    if os.path.exists(file_path):
        start_time = int((current_datetime - timedelta(minutes=500)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, 100, start_time, end_time)  # Cần hàm fetch_prices thực tế
    else:
        start_time = int((current_datetime - timedelta(minutes=INITIAL_FETCH_SIZE * 5)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        new_data = fetch_prices(symbols, interval, INITIAL_FETCH_SIZE, start_time, end_time)  # Cần hàm fetch_prices thực tế

    new_df = pd.DataFrame(new_data, columns=[
        "start_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])
    
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        combined_df = pd.concat([old_df, new_df]).drop_duplicates(subset=['start_time'], keep='last')
    else:
        combined_df = new_df

    if len(combined_df) > MAX_DATA_SIZE:
        combined_df = combined_df.iloc[-MAX_DATA_SIZE:]

    if not os.path.exists(download_path):
        os.makedirs(download_path)
    combined_df.to_csv(file_path, index=False)
    print(f"Updated data for {token} saved to {file_path}. Total rows: {len(combined_df)}")

# Hàm định dạng và lưu trữ dữ liệu
def format_data(token):
    path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(path, f"{token.lower()}_5m_data.csv")
    if not os.path.exists(file_path):
        print(f"No data file found for {token}")
        return
    df = pd.read_csv(file_path)
    columns_to_use = [
        "start_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]
    if set(columns_to_use).issubset(df.columns):
        df = df[columns_to_use]
        df.columns = [
            "start_time", "open", "high", "low", "close", "volume",
            "end_time", "quote_asset_volume", "n_trades", 
            "taker_volume", "taker_volume_usd"
        ]
        df.index = pd.to_datetime(df["start_time"], unit='ms')
        df.index.name = "date"
        output_path = os.path.join(data_base_path, f"{token.lower()}_formatted.csv")
        df.to_csv(output_path)
        print(f"Formatted data for {token} saved at {output_path}")

# Hàm tính toán phạm vi biến động giá
def calculate_fluctuation_range(token):
    file_path = os.path.join(data_base_path, f"{token.lower()}_formatted.csv")
    if not os.path.exists(file_path):
        print(f"No formatted data found for {token}.")
        return None
    df = pd.read_csv(file_path, index_col='date')
    df['fluctuation'] = (df['high'] - df['low']) / df['low'] * 100
    mean_fluctuation = df['fluctuation'].mean()
    std_fluctuation = df['fluctuation'].std()
    print(f"Average fluctuation for {token}: {mean_fluctuation:.2f}% ± {std_fluctuation:.2f}%")
    return mean_fluctuation, std_fluctuation

# Hàm dự báo ngẫu nhiên dựa trên phạm vi dao động
def random_forecast(token, steps=10):
    mean_fluctuation, std_fluctuation = calculate_fluctuation_range(token)
    if mean_fluctuation is None:
        return
    print(f"Generating {steps} random forecasts for {token}...")
    random_fluctuations = [random.uniform(mean_fluctuation - std_fluctuation, mean_fluctuation + std_fluctuation)
                           for _ in range(steps)]
    print(f"Random forecast fluctuations: {random_fluctuations}")
    return random_fluctuations

# Hàm tạo chuỗi vào ra
def create_inout_sequences(input_data, tw):
    inout_seq = []
    for i in range(len(input_data) - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

# Hàm huấn luyện mô hình LSTM
def train_model(token, epochs=10, learning_rate=0.001):
    model = load_model(token)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    data_path = os.path.join(data_base_path, f"{token.lower()}_formatted.csv")
    df = pd.read_csv(data_path, index_col='date')

    # Sử dụng giá 'close' để dự đoán
    scaler = MinMaxScaler()
    all_data = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    train_data = torch.FloatTensor(all_data)
    train_inout_seq = create_inout_sequences(train_data, 12)

    best_loss = float('inf')
    patience = 5  # số epoch cho phép không cải thiện
    counter = 0  # đếm số epoch không cải thiện

    for i in range(epochs):
        epoch_loss = 0
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            y_pred = model(seq.view(len(seq), 1, -1))
            single_loss = loss_function(y_pred, labels.view(len(labels), -1))
            single_loss.backward()
            optimizer.step()
            epoch_loss += single_loss.item()

        avg_loss = epoch_loss / len(train_inout_seq)
        print(f'Epoch {i+1}/{epochs}, Loss: {avg_loss:.4f}')

        # Theo dõi loss tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, token)
            counter = 0  # reset counter
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {i+1} due to no improvement.")
                break

# Ví dụ sử dụng các hàm
if __name__ == "__main__":
    tokens = ["ETH", "BTC", "SOL"]  # tokens = ["ETH", "BNB", "ARB"] Có thể thay đổi token tại đây
    download_data(token)
    format_data(token)
    random_forecast(token)
    train_model(token)
