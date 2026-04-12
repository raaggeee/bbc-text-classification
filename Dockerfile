FROM python:3.12.3-slim

WORKDIR /app

COPY serve/ /app/

RUN pip install -r requirements.txt
RUN pip install uvicorn

EXPOSE 8080

CMD ["uvicorn", "main:app", "--reload", "--port", "8080", "--host", "0.0.0.0"]
