# Gunakan image dasar Python yang sesuai
FROM python:3.9

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app

# Instal dependensi yang dibutuhkan
RUN pip install --no-cache-dir flask opencv-python tensorflow numpy

# Ekspose port Flask (5000)
EXPOSE 5000

# Jalankan aplikasi Flask
CMD ["python", "app.py"]

