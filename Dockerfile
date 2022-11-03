FROM python:3.10-slim

COPY requirements.txt requirements.txt

RUN apt-get -y update && apt-get install -y git 
RUN pip install -r requirements.txt



