FROM python:3.10-slim

COPY requirements.txt requirements.txt
COPY .env .env

RUN export $(grep -v '^#' .env | xargs)
RUN pip install -r requirements.txt



