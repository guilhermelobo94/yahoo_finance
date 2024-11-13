import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input


def create_dataset(data, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def normalizacao_dados(data):
    train_size = int(len(data) * 0.7)

    train_data = data[:train_size]
    test_data = data[train_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    window_size = 30
    X_train, y_train = create_dataset(scaled_train_data, window_size)
    X_test, y_test = create_dataset(scaled_test_data, window_size)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler

def create_model():
    model = Sequential()
    model.add(Input(shape=(30, 1)))
    model.add(LSTM(units=50, return_sequences=True, input_shape=(30, 1)))
    model.add(Dropout(0.2))  # 20% de dropout
    model.add(LSTM(units=50, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def kfold_cross_validation(data):
    X_train, y_train, X_test, y_test, scaler = normalizacao_dados(data)

    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_mse = float('inf')
    best_fold = 0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"Treinando fold {fold + 1}...")

        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

        model = create_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32,
                  validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping], verbose=1)

        val_predictions = model.predict(X_val_fold)
        val_mse = mean_squared_error(y_val_fold, val_predictions)

        print(f"MSE para o fold {fold + 1}: {val_mse}")

        if val_mse < best_mse:
            best_mse = val_mse
            best_fold = fold
            best_model = model
            model.save('best_model.h5')

    print(f"Melhor desempenho no fold {best_fold + 1} com MSE: {best_mse}")

    y_pred_scaled = best_model.predict(X_test)

    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inverse = scaler.inverse_transform(y_pred_scaled)

    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    print(f"MAE no conjunto de teste: {mae}")

    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    print(f"RMSE no conjunto de teste: {rmse}")

    mape = np.mean(np.abs((y_test_inverse - y_pred_inverse) / y_test_inverse)) * 100
    print(f"MAPE no conjunto de teste: {mape}%")

    plt.figure(figsize=(14, 5))
    plt.plot(y_test_inverse, color='blue', label='Valor Real')
    plt.plot(y_pred_inverse, color='red', label='Valor Previsto')
    plt.title('Previsão de Preços das Ações')
    plt.xlabel('Dias')
    plt.ylabel('Preço')
    plt.legend()
    plt.show()

    return best_model

base = 'base/historico_ativo.csv'
data = pd.read_csv(base, index_col=0)

data = data.apply(pd.to_numeric, errors='coerce').fillna(method='ffill').values

if __name__ == "__main__":
    best_model = kfold_cross_validation(data)
