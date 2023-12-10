import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from torch import FloatTensor

from src.models.lstm import LSTM
from src.models.mlp import MLP
from src.models.rnn import RNN
from src.utils import get_device
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.models.linear_regression import LinearRegression
from src.data_manager import load_stock_data, scale_min_max
from src.visualization import plot_scatter, plot_loss_and_accuracy

# Decision Tree 모델 인스턴스 생성 및 학습
def decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X_train, y_train)

    return model

# Random Forest 모델 인스턴스 생성 및 학습
def random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

# 모델 테스트 및 예측
def test_and_predict(model, X_test, y_test):
    # 테스트 데이터에 대한 예측
    predictions = model.predict(X_test)

    # 모델 평가
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    return predictions



def main():
    #
    symbol = '005930.KS'
    start_date = '2011-01-01'
    end_date = '2021-12-31'

    # 하이퍼파라미터
    batch_size = 15
    lr = 0.01
    epochs = 1000
    device = get_device()

    feature_cols = ['Open', 'High', 'Low', 'Volume']
    label_cols = ['Close']

    # 주가 데이터 정규화
    X = load_stock_data(symbol, '2010-01-01', '2020-12-31')
    X = scale_min_max(X[feature_cols].values)
    
    # 종가 데이터 정규화
    y = load_stock_data(symbol, start_date, end_date)
    y = scale_min_max(y[label_cols].values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Decision Tree 모델 학습
    model = decision_tree(X_train, y_train)
    # model = random_forest(X_train, y_train)

    # 테스트 및 예측
    predictions = test_and_predict(model, X_test, y_test)

    # 다음날 주가 예측
    last_data = X_test[-1].reshape(1, -1)
    next_day_prediction = model.predict(last_data)
    print(f'다음날 주가 예측: {next_day_prediction}')

    # 시각화: 실제 vs. 예측 (선 그래프)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Values', marker='o')
    plt.plot(predictions, label='Predicted Values', marker='o')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Actual vs. Predicted Values')
    plt.legend()
    plt.show()

    # 모델 인스턴스 생성
    # model = LinearRegression(input_size=len(feature_cols), output_size=1)
    # model = MLP(input_size=len(feature_cols), hidden_size=32, output_size=1)
    # model = RNN(input_size=len(feature_cols), hidden_size=512, output_size=1)
    # model = LSTM(input_dim=1, hidden_dim=32, output_dim=1)

    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(FloatTensor(X_train), FloatTensor(y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(FloatTensor(X_test), FloatTensor(y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = decision_tree(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on Test Data: {mse}')



    '''
    hidden = None  # initial hidden
    for epoch in range(30):  ## run the model for 30 epochs
        train_loss = []

        for data, target in train_dataloader:

            if data.shape[0] != batch_size:  # to verify if the batch no is 256 or not
                # print('Batch Size Validation- Input shape Issue:',format(data.shape))
                continue
            else:
                optimizer.zero_grad()
                ## 1. forward propagation
                prediction, hidden = model(data, hidden)

                target = target.view(-1)

                hidden = hidden.data
                batch_size = data.shape[0]

                loss = criterion(prediction.squeeze(), target)  # squeeze (256,1) -> (256) - to match target shape

                loss.backward()

                optimizer.step()

                train_loss.append(loss.item())

        print("Epoch:", epoch, "Training Loss: ", np.mean(train_loss))
    '''

if __name__ == "__main__":
    main()