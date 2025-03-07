#!/bin/bash
echo "Pilih mode:"
echo "1) Deteksi dari gambar"
echo "2) Deteksi dari video"
echo "3) Deteksi dari kamera"
read -p "Masukkan pilihan (1-3): " choice

if [ "$choice" == "1" ]; then
    python3 test_model.py 2>/dev/null
elif [ "$choice" == "2" ]; then
    python3 test_model_with_video.py 2>/dev/null
elif [ "$choice" == "3" ]; then
    python3 test_model_real-time.py 2>/dev/null
else
    echo "Pilihan tidak valid!"
    exit 1
fi
