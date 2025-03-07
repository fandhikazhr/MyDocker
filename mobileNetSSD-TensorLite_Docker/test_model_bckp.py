import sys
import os
import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, Response
import threading

# Inisialisasi Flask
app = Flask(__name__)

# ANSI escape codes untuk warna di terminal
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"

# Daftar label COCO
coco_labels = {
    1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle", 5: "Airplane",
    6: "Bus", 7: "Train", 8: "Truck", 9: "Boat", 10: "Traffic Light",
    11: "Fire Hydrant", 13: "Stop Sign", 14: "Parking Meter", 15: "Bench",
    16: "Bird", 17: "Cat", 18: "Dog", 19: "Horse", 20: "Sheep", 
    21: "Cow", 22: "Elephant", 23: "Bear", 24: "Zebra", 25: "Giraffe"
}

# Fungsi untuk warna tetap berdasarkan kelas
def get_color_for_class(class_id):
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, 3).tolist())

# Load model TensorFlow
model_path = "mobilenet_ssd/saved_model"
model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]

# Global variable untuk menyimpan frame terbaru
output_frame = None
lock = threading.Lock()

def detect_objects():
    global output_frame
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ Kamera tidak bisa diakses!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        img_resized = cv2.resize(frame, (320, 320))

        input_tensor = tf.convert_to_tensor([img_resized], dtype=tf.uint8)
        detections = infer(input_tensor)

        # Ambil hasil deteksi
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)

        for i in range(len(scores)):
            if scores[i] > 0.3:  # Tampilkan hanya deteksi dengan skor > 30%
                class_id = classes[i]
                class_name = coco_labels.get(class_id, f"Unknown ({class_id})")
                y1, x1, y2, x2 = boxes[i]
                y1, x1, y2, x2 = int(y1 * h), int(x1 * w), int(y2 * h), int(x2 * w)

                # Pilih warna untuk tiap kelas
                color = get_color_for_class(class_id)

                # Gambar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Tambahkan teks label di atas kotak
                label = f"{class_name} ({scores[i]:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Cetak output di terminal dengan warna
                print(
                    f"{CYAN}Deteksi: {RESET}"
                    f"{GREEN}Kelas={class_name}{RESET}, "
                    f"{YELLOW}Skor={scores[i]:.2f}{RESET}, "
                    f"{MAGENTA}Kotak={[x1, y1, x2, y2]}{RESET}"
                )

        # Simpan frame ke variabel global dengan lock untuk menghindari race condition
        with lock:
            output_frame = frame.copy()

    cap.release()

# Fungsi untuk streaming ke browser
def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue

            # Encode frame ke format JPEG
            _, jpeg = cv2.imencode('.jpg', output_frame)
            frame_bytes = jpeg.tobytes()

        # Kirim frame ke client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# Rute untuk streaming video
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Jalankan Flask dalam thread terpisah
if __name__ == '__main__':
    threading.Thread(target=detect_objects, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

