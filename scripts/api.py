from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import tensorflow as tf

app = FastAPI()

# Modelo de dados para receber os parâmetros da requisição
class PrevisaoRequest(BaseModel):
    data_inicio: str
    ativo: str

# Carregar o modelo previamente treinado
modelo = tf.keras.models.load_model('./scripts/best_model.keras')

# Função para consultar o histórico de uma ação e fazer a previsão
def consultar_acao(data_inicio, ativo):
    historico = yf.download(ativo, start=data_inicio, progress=False)
    historico_diario = historico['Close'].resample('D').asfreq()
    historico_filled = historico_diario.ffill()
    return historico_filled

# Função para fazer a previsão e calcular o valor ajustado
@app.post("/prever_acao/")
async def prever_acao(request: PrevisaoRequest):
    # Obter o histórico da ação
    historico = consultar_acao(request.data_inicio, request.ativo)

    # Obter o valor mais recente da ação (último preço de fechamento)
    preco_atual = historico.iloc[-1]  # Último valor de fechamento

    # Fazer a previsão usando o modelo (exemplo com uma previsão de variação percentual)
    previsao_percentual = modelo.predict(historico.values[-30:].reshape(1, -1))  # Previsão com os últimos 30 dias
    previsao_percentual = previsao_percentual[0][0]  # Pegando o valor da previsão

    # Calcular o novo valor da ação com a correção (valor atual + variação percentual)
    preco_futuro = preco_atual * (1 + previsao_percentual)

    # Retornar a previsão ajustada
    return {"previsao": preco_futuro}

