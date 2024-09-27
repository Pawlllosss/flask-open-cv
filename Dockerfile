# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster

ENV AWS_DEFAULT_REGION=us-east-1

WORKDIR /python-docker

RUN apt-get update && apt-get install -y python3-opencv

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN mkdir -p ./static/downloads
RUN mkdir -p ./static/uploads


EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]