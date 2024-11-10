from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from pydantic import BaseModel  # Definiowanie modeli danych aby przesyłać dane w żądaniach i odpowiedziach
from ultralytics import YOLO
from PIL import Image
from uuid import UUID, uuid4
from typing import Optional
import threading    # do włącznia aktywnej kamery na osobnym wątku
import uvicorn
import pickle   #otworzenie modelu model.pkl
import matplotlib.pyplot as plt
import base64
import time
import asyncio
import os
import base64
import cv2
import numpy as np
import pyrealsense2 as rs   # wrapper umożliwiający wywoływanie akcji za pomocą kodu python
import io
import tempfile


# URL w którym aplikacja jest obsługiwana w lokalnej maszynie: http://127.0.0.1:8000
app = FastAPI(debug=True)     # Zainicjowanie aplikacji FastAPI - instancja klasy FastAPI


# Stworzenie ścieżki, zaczyna się od pierwszego ukośnika '/'
@app.get("/")
def read_root():
    return {"message": "Serwer API działa poprawnie", "data": "123"}

@app.get("/info_api")
def welcome():
    return {"Wspęp": "Te API daje połącznie z dwoma modelami: Model Sieci Sjamskich oraz Model Detekcji Obiektów",
            "API": "API jest zbudowane na FastAPI, znaczna część danych wyświetlanych w aplikacji pochodzą z tego API",
            "Model Sieci Sjamskie": "Zwraca zdjęcie obiektu i 3 etykiety najbardziej prawdopodobnych obiektów",
            "Model Detekcja Obiektow": "Zwraca zdjęcie, ramki ograniczające, etykiety i prawdopodobieństwa przewidywań - dla każdego z obiektów na zdjęciu"}

@app.on_event("startup")
async def startup_event():
    """Inicjalizacja zasobów podczas startu aplikacji"""
    global yolo_model
    print("Pobieranie wag modelu YOLO")
    yolo_model = YOLO("yolov5s.pt")
    print("Model yolov5s.pt załadowany")

    global pipeline, align, depth_scale
    print("Inicjalizacja pipeline kamery Intel RealSense")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align_to = rs.stream.color
    align = rs.align(align_to)
    print("Pipeline kamery Intel RealSense załadowany")

    global camera_thread_instance
    camera_thread_instance = threading.Thread(target=camera_thread)
    camera_thread_instance.daemon = True
    camera_thread_instance.start()
    print("Wątek dla kamery jest włączony")

@app.on_event("shutdown")
async def shutdown_event():
    """Zwalnianie zasobów podczas zamykania aplikacji"""
    global pipeline
    print("Zatrzymywanie pipeline kamery Intel RealSense")
    pipeline.stop()
    print("Pipeline zatrzymany")


class PredictionResponseSiameseModel(BaseModel):
    labels: List[str]
    image_base64: str


class Detection(BaseModel):
    """Dane zwracane z modelu Detekcji Obiektów dla każdego z wykrytych obiektów"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    confidence: float
    depth: Optional[float]


class PredictionObjectDetectionModel(BaseModel):
    detections: List[Detection]
    image_base64: str


camera_thread_instance = None
latest_color_image = None
latest_depth_image = None
latest_frame_lock = threading.Lock()


def camera_thread():
    """
    Metoda jest uruchamiana na osobnym wątku i cały czas się wykonuje
    :return:
    """
    global pipeline, align, depth_scale, latest_color_image, latest_depth_image, latest_frame_lock
    while True:
        time.sleep(0.1)
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if depth_frame and color_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                with latest_frame_lock:
                    latest_color_image = color_image
                    latest_depth_image = depth_image
            else:
                print("Nie udało się pobrać klatek z kamery w wątku.")
        except Exception as e:
            print(f"Błąd w wątku kamery: {str(e)}")
            break


def capture_image():
    global latest_color_image, latest_depth_image, depth_scale, latest_frame_lock
    with latest_frame_lock:
        if latest_color_image is not None and latest_depth_image is not None:
            return latest_color_image.copy(), latest_depth_image.copy(), depth_scale
        else:
            raise HTTPException(status_code=500, detail="No image available from camera.")


@app.get("/items/{item_id}")
def read_item(item_id: str, q: str | None = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}


@app.post("/predict_siamese", response_model=PredictionResponseSiameseModel, tags=["Image top3 predictions and image"])    # zdefiniowanie Endpointa
async def predict_siamase():
    #TODO pobieranie obrazu
    #TODO wykonanie predykcji
    await asyncio.sleep(1)

    # odczytanie zapisanego obrazu z kamery:
    with open("../data/images/Banan.jpg", 'rb') as imageCamera:
        content = imageCamera.read()

    labels = ["Pomarancza", "Banan", "Jablko"]
    image_base64 = base64.b64encode(content).decode("utf-8")

    return PredictionResponseSiameseModel(labels=labels, image_base64=image_base64)


class ObjectDetectionResponse(BaseModel):
    detections: List[Detection]
    image_base64: str

def draw_bboxes(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    for det in detections:
        cv2.rectangle(image, (int(det.x_min), int(det.y_min)), (int(det.x_max), int(det.y_max)), (0, 255, 0), 2)
        cv2.putText(image, f"{det.label}{det.confidence:.2f}", (int(det.x_min), int(det.y_min)-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


@app.post("/object_detection", response_model=PredictionObjectDetectionModel, tags=["image, bbox, labels, accuracy"])
async def object_detection():
    await asyncio.sleep(1)

    try:
        color_image, depth_image, depth_scale = capture_image()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=505, detail=f"Unexpected error: {str(e)}")

    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    results = yolo_model(image_rgb, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = yolo_model.names[cls]

            x_min = int(xyxy[0])
            y_min = int(xyxy[1])
            x_max = int(xyxy[2])
            y_max = int(xyxy[3])

            # Czy współrzędne mieszczą się w granicach obrazu
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(depth_image.shape[1] - 1, x_max)
            y_max = min(depth_image.shape[0] - 1, y_max)

            x_center = int((x_min + x_max)/2)
            y_center = int((y_min + y_max)/2)

            depth_value = depth_image[y_center, x_center].astype(float)
            depth_in_meters = depth_value * depth_scale

            if depth_value == 0:
                depth_in_meters = None  # Brak danych głębokości
            else:
                depth_in_meters = round(depth_in_meters, 3)

            detection = Detection(
                x_min=xyxy[0],
                y_min=xyxy[1],
                x_max=xyxy[2],
                y_max=xyxy[3],
                label=label,
                confidence=conf,
                depth=depth_in_meters
            )
            detections.append(detection)

    image_with_boxes = draw_bboxes(color_image.copy(), detections)

    image_bgr = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)

    # Zapisanie obrazu do bufora w pamięci
    _, buffer = cv2.imencode('.png', image_bgr)
    image_bytes = buffer.tobytes()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = ObjectDetectionResponse(detections=detections, image_base64=image_base64)

    return response


# uruchamianie : (oba są bardzo podobne) robią to samo
# $fastapi dev main.py
# lub
# #uvicorn main:app --reload

# Dodatkowo można używać komendy 'curl' dla komunikacji z serwerem http od strony terminala
# wywoływanie metod z API - uzyskiwanie wyjść które zwracają wywołane funkcje
# : $curl http://127.0.0.1:8000/items/3
# przykłądowe wyjście: {"item_id":3}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port="8001")





# Archiwalny kod
# @app.on_event("startup")
# async def load_yolo_model():
#     """Inicjalizacja modelu YOLO podczas startu aplikacji"""
#     global yolo_model
#     #weights_to_yolo_path = "../backend_3D/weights/yolov9-c.pt"
#     # if not os.path.exists(weights_to_yolo_path):
#     print("Pobieranie wag modelu YOLO")
#     yolo_model = YOLO("yolov5s.pt")
#     # else:
#     #     yolo_model = YOLO(weights_to_yolo_path)
#     print("Model Yolov5s.pt załadowany")

###DODANE:
# @app.on_event("startup")
# async def startup_event():
#     """Inicjalizacja zasobów podczas startu aplikacji"""
#     global yolo_model
#     print("Pobieranie wag modelu YOLO")
#     yolo_model = YOLO("yolov5s.pt")
#     print("Model yolov5s.pt załadowany")
#
#     global pipeline, align, depth_scale
#     print("Inicjalizacja pipeline kamery Intel RealSense")
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#     profile = pipeline.start(config)
#     depth_sensor = profile.get_device().first_depth_sensor()
#     depth_scale = depth_sensor.get_depth_scale()
#     align_to = rs.stream.color
#     align = rs.align(align_to)
#     print("Pipeline kamery Intel RealSense załadowany")




# def capture_image():
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
#
#     profile = pipeline.start(config)
#     device = profile.get_device()
#
#     color_sensor = None
#     for sensor in device.sensors:
#         if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
#             color_sensor = sensor
#             break
#
#     if color_sensor is None:
#         print("Nie znaleziono sensora kolorowego.")
#         pipeline.stop()
#         raise HTTPException(status_code=500, detail="Sensor RGB Camera not found.")
#
#     # Ustawienia kamery
#     color_sensor.set_option(rs.option.enable_auto_exposure, True)
#     color_sensor.set_option(rs.option.exposure, 166)
#     color_sensor.set_option(rs.option.gain, 64)
#     color_sensor.set_option(rs.option.brightness, 0.0)
#     color_sensor.set_option(rs.option.contrast, 50)
#     color_sensor.set_option(rs.option.gamma, 300)
#     color_sensor.set_option(rs.option.hue, 0)
#     color_sensor.set_option(rs.option.saturation, 64)
#     color_sensor.set_option(rs.option.sharpness, 50)
#     color_sensor.set_option(rs.option.white_balance, 4600)
#
#     try:
#         # Pobierz jedną klatkę
#         frames = pipeline.wait_for_frames(timeout_ms=2000)
#         color_frame = frames.get_color_frame()
#         if not color_frame:
#             raise HTTPException(status_code=500, detail="Failed to capture image from camera.")
#
#         # Konwersja do tablicy NumPy
#         color_image = np.asanyarray(color_frame.get_data())
#
#         # Zamknięcie pipeline
#         pipeline.stop()
#
#         return color_image
#     except Exception as e:
#         pipeline.stop()
#         raise HTTPException(status_code=500, detail=f"Error capturing image: {str(e)}")


# def capture_image():
#     # pipeline = rs.pipeline()  # stworzenie pipeline
#     # config = rs.config()
#     #
#     # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#     # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#     #
#     # profile = pipeline.start(config)
#     # depth_sensor = profile.get_device().first_depth_sensor()
#     # depth_scale = depth_sensor.get_depth_scale()
#     # align_to = rs.stream.color
#     # align = rs.align(align_to)
#
#     try:
#         frames = pipeline.wait_for_frames()
#         aligned_frames = align.process(frames)
#         depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()
#
#         if not depth_frame or not color_frame:
#             raise HTTPException(status_code=501, detail="Failed to capture image from camera.")
#
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())
#         pipeline.stop()
#         return color_image, depth_image, depth_scale
#     except Exception as e:
#         pipeline.stop()
#         raise HTTPException(status_code=502, detail=f"Error capturing image: {str(e)}")


#_________________
# metoda używające modelu Sieci Sjamskich
# @app.post("/predict_siamase", response_model=PredictionResponseSiameseModel)    # zdefiniowanie Endpointa
# def predict_siamase():
#     #TODO pobieranie obrazu
#     #TODO wykonanie predykcji
#     labels = ["produkt1", "produkt2", "produkt3"]
#
#     return PredictionResponseSiameseModel(labels=labels)

#___________
# @app.post("/object_detection", response_model=PredictionObjectDetectionModel, tags=["image, bbox, labels, accuracy"])
# async def object_detection():
#     await asyncio.sleep(1)
#
#     #________________________
#     try:
#         image = capture_image()
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
#
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     results = yolo_model(image_rgb, verbose=False)
#
#     detections = []
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             xyxy = box.xyxy[0].cpu().numpy()
#             conf = box.conf[0].cpu().numpy()
#             cls = int(box.cls[0].cpu().numpy())
#             label = yolo_model.names[cls]
#
#             detection = Detection(
#                 x_min=xyxy[0],
#                 y_min=xyxy[1],
#                 x_max=xyxy[2],
#                 y_max=xyxy[3],
#                 label=label,
#                 confidence=conf
#             )
#             detections.append(detection)
#
#     image_with_boxes = draw_bboxes(image.copy(), detections)
#
#     image_bgr = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
#
#     # Zapisanie obrazu do bufora w pamięci
#     _, buffer = cv2.imencode('.png', image_bgr)
#     image_bytes = buffer.tobytes()
#
#     image_base64 = base64.b64encode(image_bytes).decode("utf-8")
#
#     response = ObjectDetectionResponse(detections=detections, image_base64=image_base64)
#
#     return response

#________________________inna opcja

    # with open("../data/images/image_good.png", 'rb') as imageCamera:
    #     content = imageCamera.read()
    #
    # detections = [
    #     Detection(x=100.0, y=150.0, depth=400, label="Krzeslo", confidence=0.88),
    #     Detection(x=250.0, y=250.0, depth=290, label="Object2", confidence=0.85),
    # ]
    # image_base64 = base64.b64encode(content).decode('utf-8')
    # return PredictionObjectDetectionModel(detections=detections, image_base64=image_base64)