# 1. select a base image with Python 3.12 slim version
FROM python:3.12-slim
# 2. choose the working directory inside the container
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 3. copy the files from your project into the image
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]