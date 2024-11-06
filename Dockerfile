FROM python:3.10-slim

WORKDIR /yahoo_finance

COPY ./scripts .

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && install -y wget unzip gnupg ca-certificates fonts-liberation

RUN pip install -r requirements.txt

CMD ["python3", "api.py"]