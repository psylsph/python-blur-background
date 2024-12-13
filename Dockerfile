# syntax=docker/dockerfile:1

FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt update -y && apt-get install -y python3-dev build-essential && \
    pip install -r requirements.txt 

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host", "0.0.0.0", "--port","5000"]
