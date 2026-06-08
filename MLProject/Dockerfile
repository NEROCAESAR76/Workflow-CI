FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install pandas scikit-learn mlflow==2.19.0

CMD ["python", "modelling.py"]