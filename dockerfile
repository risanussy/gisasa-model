# Gunakan base image Python (versi boleh disesuaikan)
FROM python:3.9-slim

# Set workdir di dalam container
WORKDIR /app

# Copy file requirements.txt dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh isi folder (termasuk app.py dan model .h5)
COPY . .

# Gunakan gunicorn untuk menjalankan server Flask di port 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
