from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import yfinance as yf
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import logging
from typing import Optional

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrevisaoRequest(BaseModel):
    ativo: str = Field(..., example="PETR4.SA")
    passos: Optional[int] = Field(default=30, example=30)

try:
    modelo = tf.keras.models.load_model('best_model.h5')
    logger.info("Modelo carregado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo: {str(e)}")
    modelo = None


def consultar_acao(ativo, period='2y'):
    try:
        historico = yf.download(ativo, period=period, progress=False)
        if historico.empty:
            raise ValueError("Dados históricos não encontrados para o ativo fornecido.")
        historico_diario = historico['Close'].resample('D').asfreq()
        historico_filled = historico_diario.ffill()
        return historico_filled
    except Exception as e:
        logger.error(f"Erro ao consultar ação: {str(e)}")
        raise


def preparar_dados_para_previsao(historico, window_size=30):
    data = historico.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    if len(scaled_data) < window_size:
        return None, None, None

    return scaled_data, scaler, data


@app.post("/prever_acao/")
async def prever_acao(request: PrevisaoRequest):
    if modelo is None:
        raise HTTPException(status_code=500, detail="Modelo não está disponível.")

    try:
        historico = consultar_acao(request.ativo)

        scaled_data, scaler, data = preparar_dados_para_previsao(historico)
        if scaled_data is None:
            raise HTTPException(status_code=400, detail="Dados insuficientes para a previsão.")

        window_size = 30
        previsoes = []
        input_data = scaled_data[-window_size:].reshape(1, window_size, 1)

        for _ in range(request.passos):
            previsao_scaled = modelo.predict(input_data)

            previsoes.append(previsao_scaled[0, 0])

            input_data = np.append(input_data[:, 1:, :], [[previsao_scaled[0]]], axis=1)

        previsoes = scaler.inverse_transform(np.array(previsoes).reshape(-1, 1))

        ultima_data = historico.index[-1]
        datas_futuras = []
        dias_adicionados = 0
        while len(datas_futuras) < request.passos:
            proxima_data = ultima_data + pd.Timedelta(days=1 + dias_adicionados)
            if proxima_data.weekday() < 5:
                datas_futuras.append(proxima_data)
            dias_adicionados += 1

        resultado = {
            "previsoes": [
                {"data": str(data.date()), "preco_previsto": float(previsao[0])}
                for data, previsao in zip(datas_futuras, previsoes)
            ]
        }

        logger.info(f"Previsão realizada com sucesso para o ativo {request.ativo}.")

        return resultado

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        logger.error(f"Erro ao processar a previsão: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor.")
