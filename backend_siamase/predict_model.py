import os.path
import cv2
import tensorflow as tf
import numpy as np
import keras
from keras import Model
from model.SiameseModel import SiameseModel
from distance.DistanceLayer import DistanceLayer
from utilities.data_reader import DataReader
from utilities.mapping import Mapper
from keras.api.applications.densenet import DenseNet201
from keras import layers
import os
from distance.EuclideanDistance import EuclideanDistance
from metrics.Product import Product
from metrics.Metrics import Metrics

image_size = 224, 224
margin = 0.5
representatives_path = r"data/represent_our_5"


def load_representatives(path):
    """
    Create representatives dictionary with all necessary info
    Args:
        path: Get the images from this directory

    Returns:
        'representatives' dictionary. Key: 'label' , value: object of Product() class with all necessary attributes
    """
    products_paths = DataReader.read_types_of_products(path)
    products_dictionary = DataReader.generate_images_classes_dict(products_paths)
    representatives = {}
    mapper = Mapper(image_size)
    for label, products in products_dictionary.items():
        lab = label.rsplit("/", 1)[1]
        representatives[lab] = []
        for product in products:
            rep = Product()
            rep.label, rep.image = mapper.map_single_product(lab, product)
            rep.image_path = product
            representatives[lab].append(rep)
    for label, products in representatives.items():
        for product in products:
            product.image_to_predict = tf.reshape(product.image, shape=(1,) + image_size + (3,))
    return representatives


def add_representatives(image_to_add, label):
    """
    Creating new folder of "label" name if not exists.
    Create object of class Product() with all info about this image and save to Representatives.
    save new object to global representatives.
    save new image to new folder : 'label' representatives.

    Args:
        image_to_add: choosen image from camera capture / folder.
        label: name of object written in TextBox() on Gradio

    Returns:
        info about all done staff

    """
    rep_add = Product()
    mapper = Mapper(image_size)
    label_add, image_add = mapper.map_single_product(label=label, image_path=image_to_add)
    embedding_add = embedding_layer(tf.expand_dims(image_add, axis=0)).numpy()

    new_dir = f"{representatives_path}/{label_add}"
    #dodać sparwdzenie czy nie ma zdjęcia o już istniejącej nazwie
    image_name = f"{label_add}_{np.random.randint(low=0, high=100000, dtype=np.uint32)}.jpg"
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    rep_add.image = image_add
    rep_add.label = label_add
    rep_add.embedding = embedding_add
    rep_add.image_path = f"{new_dir}/{image_name}"
    rep_add.image_to_predict = tf.reshape(rep_add.image, shape=(1,) + image_size + (3,))

    if label_add not in representatives:
        representatives[label_add] = []

    representatives[label_add].append(rep_add)

    tf.keras.preprocessing.image.save_img(f"{new_dir}/{image_name}", rep_add.image)

    info = f"Added {label_add} to path {new_dir}. Name of image: {image_name}."
    return info


def evaluate_representatives(embedding_layer, representatives):
    """
    Create embedding for all images in representatives
    Returns:
        representatives (class Product() ) with .embedding attribute
    """
    for label, products in representatives.items():
        for product in products:
            product.embedding = np.asarray(embedding_layer(product.image_to_predict)).astype(
                "float32"
            )
    return representatives


def create_contrastive_model():
    """
    Create Contrastive Model.
    Choose architecture and load previously trained weights.

    Returns: embedding_network - for create embedding from image
    """
    modelVGG = DenseNet201(weights='imagenet', include_top=False)

    input_layer = keras.layers.Input((224, 224, 3))
    x = modelVGG(input_layer)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    embedding_network = keras.Model(input_layer, x)

    input_1 = keras.layers.Input(image_size + (3,))
    input_2 = keras.layers.Input(image_size + (3,))

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer_distance = keras.layers.Lambda(EuclideanDistance(), output_shape=(1,))([tower_1, tower_2])
    siamese = keras.Model(inputs=[input_1, input_2], outputs=merge_layer_distance)

    for layer in modelVGG.layers:
        layer.trainable = False
    siamese.load_weights(r"/home/c100rczyk/Projekty/Thesis_project_PWR/backend_siamase/trained_models/Net201_VegFru_CL_10epok_50tys_5tys_64embed_top5_90_RruitRecognition.weights.h5")
    #siamese.load_weights(r"../trained_models/Net201_VegFru_CL_10epok_50tys_5tys_64embed_top5_90_RruitRecognition.weights.h5")

    return embedding_network, siamese


def predict(image):
    """
    Create embedding from captured image.
    Calculate mean distance between this image and each of representatives.
    Get top 5 most similar representatives.
    Args:
        image: captured from camera / get from folder

    Returns: top 5 labels and corresponding images
    """
    mapper = Mapper(image_size)
    label, image = mapper.map_single_product("", image)
    image = tf.reshape(image, shape=(1,) + image_size + (3,))
    embedding = np.asarray(embedding_layer(image)).astype("float32")
    distances = np.zeros(len(representatives.keys()))
    labels = []
    j = 0
    for rep_label, products in representatives.items():
        sum = 0
        for product in products:
            sum += EuclideanDistance.calculate_distance((embedding, product.embedding))
        mean = sum / len(products)
        distances[j] = mean
        j += 1
        labels.append(rep_label)

    top5_dist, top5_labels = Metrics.find_top_5(distances, labels)
    predicted_images = []

    for label in top5_labels:
        img = tf.image.convert_image_dtype(
            representatives[label][0].image, dtype=tf.uint8
        )
        predicted_image = img.numpy()
        predicted_images.append(predicted_image)
        #print(len(predicted_images))

    print(top5_labels)
    return (top5_labels, predicted_images)

# def predict(image, embedding_layer, evaluated_reps):
#     """
#     Wykonuje predykcję dla zadanego obrazu.
#     Zwraca top 3 etykiety i odpowiadające im obrazy reprezentantów.
#     """
#     # Przetwarzanie obrazu
#     image = cv2.resize(image, (224, 224))
#     image = tf.keras.applications.vgg16.preprocess_input(image)
#     image = np.expand_dims(image, axis=0)
#
#     # Oblicz embedding dla obrazu wejściowego
#     image_embedding = embedding_layer.predict(image)
#
#     # Oblicz dystanse do reprezentantów
#     distances = {}
#     for label, embeddings in evaluated_reps.items():
#         # Oblicz średni dystans do wszystkich embeddingów dla danego labela
#         dist = np.mean(np.linalg.norm(embeddings - image_embedding, axis=1))
#         distances[label] = dist
#
#     # Sortuj etykiety według dystansu
#     sorted_labels = sorted(distances, key=distances.get)
#     top3_labels = sorted_labels[:3]
#
#     # Pobierz odpowiadające obrazy reprezentantów
#     top3_images = []
#     for label in top3_labels:
#         # Pobierz pierwszy obraz reprezentanta dla danego labela
#         img_array = representatives[label][0]
#         img_array = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2BGR)
#         top3_images.append(img_array)
#
#     return top3_labels, top3_images


def capture_and_load_image(object=6):
    """
    Capture image from camera and convert this image appropriate
    Args:
        object: id of camera (int)
            run command: ls dev/video*     there is all accessible video devices from which we cen get image
            There can be 2 objects from one camera. One is form image flow (we want this one)
            second for technical configuration.

    Returns:
        image in numpy format
    """
    camera = cv2.VideoCapture(object)
    return_value, image = camera.read()
    #output_path = f"przechwycone_obrazy/zdj_<built-in function id>.png"  #f"przechwycone_obrazy/zdj_{id}.png"
    output_path = f"przechwycone_obrazy/zdj_1.png"
    cv2.imwrite(output_path, image)
    camera.release()
    print("Przechwycono obraz (ok)")
    if type(output_path) is np.ndarray:
        captured_image = tf.convert_to_tensor(output_path)
    else:
        captured_image = tf.io.read_file(output_path)
        captured_image = tf.image.decode_jpeg(captured_image, channels=3)
    captured_image = tf.image.convert_image_dtype(captured_image, dtype=tf.float32)
    captured_image = tf.image.resize(captured_image, image_size)
    captured_image_np = captured_image.numpy()
    print("zmieniono obraz na tablicę numpy")
    return captured_image_np


embedding_layer, model = create_contrastive_model()
representatives = load_representatives(representatives_path)
representatives = evaluate_representatives(embedding_layer, representatives)


#TODO Stworzyć metodę która może być wywoływana przez endpoint /predict_siamase
# Po stworzeniu przewidywania, będzie zwracać 3 etykiety najbardziej prawdopodobnych klas w postaci JSON.
# + Przemyśleć w jaki sposób obserwować wejście z kamery. Czy ma być na jakiś przycisk, czy jaki
# + Przygotować zbiór danych na którym wytrenuję model






