import yfinance as yf
from datetime import datetime, timedelta

ativos = ["PETR4.SA", "PETR3.SA", "AAPL", "GOOG", "MSFT", "IBM", "KNRI11.SA", "HGLG11.SA"]

data_inicio = (datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d')

historico = yf.download(ativos, start=data_inicio, progress=False)

historico_mensal = historico['Close'].resample('ME').last()

historico.to_csv("/home/lobo/Documentos/fiap/yahoo_finance/base/historico_multiplos_ativos.csv")
