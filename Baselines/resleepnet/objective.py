"""Native ResSleepNet two-task objective, with correct global denominators."""
import tensorflow as tf


def objective(model, batch, training, ahi_scale=500.):
    r, labels, mask, ahi = batch
    probs, prediction = model(r, training=training)
    probs = tf.debugging.check_numerics(tf.cast(probs, tf.float32), 'stage probabilities')
    prediction = tf.debugging.check_numerics(tf.cast(prediction[:, 0], tf.float32), 'AHI prediction')
    valid = tf.cast(mask, tf.float32)
    # Invalid reference values must never enter a loss operator.
    labels = tf.where(mask, labels, tf.zeros_like(labels))
    ce = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
    pieces = dict(ce_sum=tf.reduce_sum(ce * valid),
                  mse_sum=tf.reduce_sum(tf.square(prediction - ahi)),
                  mae_sum=tf.reduce_sum(tf.abs(prediction - ahi)),
                  correct=tf.reduce_sum(tf.cast(tf.argmax(probs, -1, output_type=tf.int32) == labels, tf.float32) * valid),
                  epochs=tf.reduce_sum(valid), nights=tf.cast(tf.shape(r)[0], tf.float32))
    context = tf.distribute.get_replica_context()
    def global_sum(x):
        return context.all_reduce(tf.distribute.ReduceOp.SUM, x) if context else x
    loss = pieces['ce_sum'] / global_sum(pieces['epochs']) + pieces['mse_sum'] / (global_sum(pieces['nights']) * ahi_scale)
    # Upstream declares L2 regularizers but comments them out of its objective.
    return tf.debugging.check_numerics(loss, 'loss'), pieces
