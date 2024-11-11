import yfinance as yf
from datetime import datetime, timedelta

def bases():
    ativos = ["PETR4.SA", "PETR3.SA", "AAPL", "GOOG", "MSFT", "IBM", "KNRI11.SA", "HGLG11.SA"]

    # Definir o intervalo de 20 anos
    data_inicio = (datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d')

    # Baixar os dados históricos
    historico = yf.download(ativos, start=data_inicio, progress=False)

    # Preencher os valores de fim de semana com NaN, mas mantendo os dias
    historico_diario = historico['Close'].resample('D').asfreq()

    # Preencher os valores de fim de semana com o último valor útil
    historico_filled = historico_diario.ffill()

    # Salvar os dados no arquivo CSV
    historico_filled.to_csv("/home/lobo/Documentos/fiap/yahoo_finance/base/historico_multiplos_ativos.csv")


def consultar_acao(data_inicio, ativo):
    historico = yf.download(ativo, start=data_inicio, progress=False)

    historico_diario = historico['Close'].resample('D').asfreq()

    historico_filled = historico_diario.ffill()
    return historico_filled

data_inicio = "2010-01-01"

consultar_acao(data_inicio=data_inicio, ativo='PETR4.SA')