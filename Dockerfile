FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./
CMD ["python", "-m", "app.demo", "--csv", "sample.csv", "--q", "What are the columns?"]
