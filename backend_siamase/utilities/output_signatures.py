from dataclasses import dataclass
from tensorflow import TypeSpec
import tensorflow as tf


@dataclass
class OutputSignature:
    """
    Different Specification of data, depends on loss function (contrastive/triplets)

    Input: name
    Return: Structure of data

    """
    contrastive_loss: TypeSpec = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
    triplet_loss: TypeSpec = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )
    single_loss: TypeSpec = (
        tf.TensorSpec(shape=(), dtype=tf.string),  # Ścieżka do obrazu
        tf.TensorSpec(shape=(), dtype=tf.int32),  # Etykieta klasy jako liczba całkowita
    )

    representatives: TypeSpec = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )
