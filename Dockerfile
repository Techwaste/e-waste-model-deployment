FROM python:3.10.10

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt project-aing-dcea2bba16f3.json /build/ 

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

ENV PYTHONUNBUFFERED=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/build/project-aing-dcea2bba16f3.json

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "main:app"]