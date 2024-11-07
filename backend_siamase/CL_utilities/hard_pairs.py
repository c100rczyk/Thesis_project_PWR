import tensorflow as tf

def batch_contrastive_loss(margin, squared=False, num_positives=3, num_negatives=3):
    return lambda labels, embeddings: real_batch_contrastive_loss(labels, embeddings, margin, squared=squared, num_positives=num_positives, num_negatives=num_negatives)

def real_batch_contrastive_loss(labels, embeddings, margin, squared=False, num_positives=3, num_negatives=3):
    reshaped_labels = tf.squeeze(labels)#, axis=[1])
    reshaped_embeddings = embeddings

    positive_mask = _get_positive_mask(reshaped_labels)
    negative_mask = tf.logical_not(positive_mask)
    positive_mask = tf.cast(positive_mask, dtype=tf.float32)
    negative_mask = tf.cast(negative_mask, dtype=tf.float32)

    positive_mask = tf.linalg.band_part(positive_mask, 0, -1)
    negative_mask = tf.linalg.band_part(negative_mask, 0, -1)
    distances = _pairwise_distances(reshaped_embeddings, squared=squared)
    positive_distances = tf.multiply(positive_mask, distances)

    negative_losses = tf.multiply(negative_mask, tf.square(tf.maximum(margin - distances, 0)))
    negative_losses_is_zero = tf.equal(tf.reduce_sum(negative_losses), 0.0)
    positive_losses = tf.square(positive_distances)
    positive_losses_is_zero = tf.equal(tf.reduce_sum(positive_losses), 0.0)

    return tf.cond(
        tf.logical_and(positive_losses_is_zero, negative_losses_is_zero),
        lambda: tf.reduce_sum(positive_losses),
        lambda: calculate_result(negative_losses, num_negatives, num_positives, positive_losses)
    )

def calculate_result(negative_losses, num_negatives, num_positives, positive_losses):
    flat_positive_losses = tf.reshape(positive_losses, [-1])
    hard_positives, _ = tf.math.top_k(flat_positive_losses, k=tf.minimum(tf.size(flat_positive_losses), num_positives))
    not_zero_mask = tf.not_equal(hard_positives, 0.0)
    hard_positives = tf.boolean_mask(hard_positives, not_zero_mask)
    flat_negative_losses = tf.reshape(negative_losses, [-1])
    hard_negatives, _ = tf.math.top_k(flat_negative_losses, k=tf.minimum(tf.size(flat_negative_losses), num_negatives))
    not_zero_mask = tf.not_equal(hard_negatives, 0.0)
    hard_negatives = tf.boolean_mask(hard_negatives, not_zero_mask)
    all_hard_losses = tf.concat([hard_positives, hard_negatives], axis=0)
    losses_mean = tf.reduce_mean(all_hard_losses)
    return losses_mean

def _get_positive_mask(labels):
    labels_expanded = tf.expand_dims(labels, 1)
    result = tf.equal(labels_expanded, tf.transpose(labels_expanded))
    return result

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances