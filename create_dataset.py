import threading
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time

value = 250

# Po podłączeniu kamery Intel, można uruchomić skrypt. Rozpocznie się proces wczytywania
# i zapisywania zdjęć

def camera_thread():
    global value
    pipeline = rs.pipeline()
    config = rs.config()

    # Konfiguracja urządzenia
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    # Sprawdzenie obecności kamery RGB
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("Alignment process requires both Depth camera and Color sensor")
        exit(0)

    # Konfiguracja strumieni
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)     # format .bgr8 - wychodzą ładne zdjęcia
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Rozpoczęcie streamowania
    profile = pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            folder = "data/zbior_scena"
            if not os.path.exists(folder):
                os.makedirs(folder)
            file_name = f"image_scena_{value}.png"
            cv2.imwrite(os.path.join(folder, file_name), color_image)
            print(f"Klatka {value} otrzymana i zapisana jako {file_name}")
            value += 1 # każde kolejne zdjęcie ze sceny ma numer jeden więcej (numeracja zdjęć)

            plt.imshow(color_image) # sprawdzenie jakie zdjęcia wychodzą na bieżąco
            plt.show()

            time.sleep(14)  # Chwila na zmianę sceny
    finally:
        pipeline.stop()

if __name__ == "__main__":
    camera_thread_instance = threading.Thread(target=camera_thread)
    camera_thread_instance.daemon = True
    camera_thread_instance.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Zatrzymano skrypt.")
