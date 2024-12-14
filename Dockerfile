# syntax=docker/dockerfile:1

FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt update -y && apt-get install -y curl python3-dev build-essential libgl1-mesa-glx libglib2.0-0 && pip install -r requirements.txt 
RUN mkdir -p ai-model
RUN curl -L https://huggingface.co/stoned0651/isnet_dis.onnx/resolve/main/isnet_dis.onnx -o ai-model/isnet_dis.onnx

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host", "0.0.0.0", "--port","5000"]
