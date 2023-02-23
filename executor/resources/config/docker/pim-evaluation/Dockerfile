FROM python:3.10-slim

RUN apt update \
    && apt install -y --no-install-recommends git \
    && apt purge -y --auto-remove \ 
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt



