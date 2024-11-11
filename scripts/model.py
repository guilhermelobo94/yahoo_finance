from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os

# Função de criação do dataset
def create_dataset(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Função de normalização e treinamento
def normalizacao_dados(data, dates):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Definir o tamanho do conjunto de treinamento e teste
    train_size = int(len(scaled_data) * 0.7)  # 70% para treino
    test_size = len(scaled_data) - train_size

    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    window_size = 30
    X_train, y_train = create_dataset(train_data, window_size)
    X_test, y_test = create_dataset(test_data, window_size)

    # Ajuste de forma para LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaler, dates

# Função para criar o modelo
def create_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(30, 1)))
    model.add(Dropout(0.2))  # 20% de dropout
    model.add(LSTM(units=50, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Cross-validation KFold com EarlyStopping e salvamento do melhor modelo
def kfold_cross_validation(data, dates):
    X_train, y_train, X_test, y_test, scaler, dates = normalizacao_dados(data, dates)

    # K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_mse = float('inf')
    best_fold = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Treinando fold {fold + 1}...")

        # Dividir os dados para treino e validação
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

        # Criar o modelo e adicionar early stopping
        model = create_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Treinamento
        model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=32,
                  validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping])

        # Avaliar o desempenho
        val_predictions = model.predict(X_val_fold)
        val_mse = mean_squared_error(y_val_fold, val_predictions)

        print(f"MSE para o fold {fold + 1}: {val_mse}")

        # Salvar o modelo se o MSE for o melhor até agora
        if val_mse < best_mse:
            best_mse = val_mse
            best_fold = fold
            best_model = model
            model.save('best_model.keras')  # Salvar o modelo com o melhor desempenho

    print(f"Melhor desempenho no fold {best_fold + 1} com MSE: {best_mse}")
    return best_model

# Carregar os dados
base = '~/Documentos/fiap/yahoo_finance/base/historico_multiplos_ativos.csv'
data = pd.read_csv(base, header=1)
data.columns.values[0] = 'Date'

# Converter a coluna 'Date' para o formato datetime
dates = pd.to_datetime(data['Date'])

# Remover a coluna 'Date' e preencher os valores numéricos
data = data.drop(columns=['Date'])
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

# Chamar a função de cross-validation
best_model = kfold_cross_validation(data, dates)
