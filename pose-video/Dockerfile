FROM python:3.6
WORKDIR /app

COPY ./. .

RUN apt update && apt install -y cmake
RUN pip install -r requirements.txt
CMD ["python", "-u", "app.py"]
