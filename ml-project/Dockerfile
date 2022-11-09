#!/usr/bin/env python3
FROM python:3.6
WORKDIR /app

COPY ./. .

RUN apt update && apt install -y cmake
RUN pip install -r requirements.txt
RUN chmod 644 app.py
CMD ["python", "-u", "app.py"]
