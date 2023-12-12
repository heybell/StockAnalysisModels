import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor

from src.early_stopping import EarlyStopping
from src.utils import get_device, get_model
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.data_manager import load_stock_data, scale_min_max

def main():
    feature_cols = ['Open', 'High', 'Low']
    label_cols = ['Close']

    early_stop_patience = 1000

    input_size = len(feature_cols)
    hidden_size = 16
    output_size = 1

    symbol = '034220.KS'
    start_date = '2010-01-01'
    end_date = '2021-12-31'

    batch_size = 16
    lr = 0.0001
    epochs = 1000
    device = get_device()

    model_params = {
        'input_size' : input_size,
        'hidden_size' : hidden_size,
        'output_size' : output_size
    }

    stock_data = load_stock_data(symbol, start_date, end_date)

    # 데이터 정규화
    X = scale_min_max(stock_data[feature_cols].values)
    y = scale_min_max(stock_data[label_cols].values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    model = get_model('rnn', model_params)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stop = EarlyStopping(patience=early_stop_patience)

    train_dataset = TensorDataset(FloatTensor(X_train), FloatTensor(y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(FloatTensor(X_test), FloatTensor(y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # hidden : 입력 데이터의 시간적 의존성을 캡처, 이를 통해 RNN은 이전 시간 단계의 컨텍스트를 기반으로 예측을 수행
    hidden = None

    for epoch in range(epochs):
        train_loss = []

        for data, target in train_dataloader:

            if data.shape[0] != batch_size:
                continue
            else:
                optimizer.zero_grad()
                prediction, hidden = model(data, hidden)

                target = target.view(-1)

                hidden = hidden.data
                batch_size = data.shape[0]

                loss = criterion(prediction.squeeze(), target)

                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

        print("Epoch:", epoch, "Training Loss: ", np.mean(train_loss))

        # early_stopping 추가
        EARLY_STOP = early_stop(loss)

        if EARLY_STOP:
            break

    # 예측
    future_steps = 255

    last_seq = torch.tensor(X_test[-batch_size:], dtype=torch.float32)
    pred_list = []

    model.eval()
    with torch.no_grad():
        for i in range(future_steps):
            prediction, hidden = model(last_seq, hidden)
            pred_list.append(prediction.numpy())

            # 예측 값을 다음 시퀀스에 추가 : pred shape은 [8, 1]인데... 이걸 cat 하려하다니 어휴...😣
            # last_seq = torch.cat([last_seq, prediction], dim=0)

if __name__ == "__main__":
     main()