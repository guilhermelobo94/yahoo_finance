Trabalho do quarto módulo da pós graduação de machine learning
=======
# Projeto Deep learning - Desenvolvedores

- Gabriel Sargeiro ([LinkedIn](https://www.linkedin.com/in/gabriel-sargeiro/))
- Guilherme Lobo ([LinkedIn](https://www.linkedin.com/in/guilhermegclobo/))
- Matheus Moura ([LinkedIn](https://www.linkedin.com/in/matheus-moura-pinho-55a25b186/))

# **1º Passo:** 

### 1.1 Instalar Dependências Localmente

> pip install -r requirements.txt

### 1.2 Preparar Dados e Treinar Modelo
#### Este script finance.py coleta e processa os dados históricos das ações, preparando-os para o treinamento do modelo.

> python finance.py

#### Este script model.py treina o modelo LSTM usando os dados coletados e salva o modelo treinado para uso na API.

> python model.py

### 1.3 Rodar API Localmente (Opcional)
#### Para rodar a API localmente para testes, execute:
> uvicorn api:app --host 0.0.0.0 --port 8000
#### Acessar:
> localhost:8000/docs

# **2º Passo:**

### Comandos pra deploy da API (Amazon Linux AMI)
#### Apos criar chave, EC2, elastic IP e configurar portas:

> icacls "C:\.projetos\yahoo_finance\chave2.pem" /inheritance:r
> 
> icacls "C:\.projetos\yahoo_finance\chave2.pem" /grant:r "mathe:R"

#### Subir arquivos:
> scp -i "C:\.projetos\yahoo_finance\chave2.pem" C:\.projetos\yahoo_finance\requirements_ec2.txt ec2-user@54.236.103.139:~/
> 
> scp -i "C:\.projetos\yahoo_finance\chave2.pem" C:\.projetos\yahoo_finance\api.py ec2-user@54.236.103.139:~/ 
> 
> scp -i "C:\.projetos\yahoo_finance\chave2.pem" C:\.projetos\yahoo_finance\best_model.h5 ec2-user@54.236.103.139:~/

#### Dependencias e execução: 

> ssh -i chave2.pem ec2-user@54.236.103.139
> 
> python3 -m venv venv
> 
> source ~/venv/bin/activate
> 
> sudo yum update -y
> 
> sudo yum install -y python3 python3-pip
> 
> sudo yum groupinstall -y "Development Tools"
> 
> sudo yum install -y python3-devel openssl-devel libffi-devel
> 
> pip install --upgrade pip
> 
> pip install -r requirements_ec2.txt
> 
> sudo yum install -y gcc
> 
> uvicorn api:app --host 0.0.0.0 --port 8000


#### Ligar API:

> ssh -i chave2.pem ec2-user@54.236.103.139
> 
> source ~/venv/bin/activate
> 
> uvicorn api:app --host 0.0.0.0 --port 8000

#### Acessar API:

> http://54.236.103.139:8000/docs