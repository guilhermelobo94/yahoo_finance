import yfinance as yf
from datetime import datetime, timedelta
import os

def bases():
    ativo = "PETR4.SA"

    data_inicio = (datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d')
    data_fim = datetime.now().strftime('%Y-%m-%d')

    historico = yf.download(ativo, start=data_inicio, end=data_fim, progress=False)

    historico_diario = historico['Close'].resample('D').asfreq()

    historico_filled = historico_diario.ffill()

    if not os.path.exists('base'):
        os.makedirs('base')

    historico_filled.to_csv("base/historico_ativo.csv")

if __name__ == "__main__":
    bases()
