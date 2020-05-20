FROM python:3
MAINTAINER Aayush Agrawal (aayushagrawal135@gmail.com)

RUN mkdir app
RUN mkdir app/src
COPY nn_numpy/src /app/src

COPY requirements.txt /app
RUN pip3 install -r app/requirements.txt

CMD ["ls"]