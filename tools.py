import tensorflow as tf

def create_model():
  base_model = tf.keras.applications.VGG16(include_top=False)
  model = tf.keras.Sequential(
      [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', input_shape=(1, 7*7*512)),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.Dense(128, activation='relu')
      ]
  )

  return model

@tf.function
def triplet_loss(y_pred, alpha = 0.2):
  anchor, positive, negative = tf.split(y_pred, [1,1,1], axis = 0)
  # tf.print(anchor, positive, negative)
  gap_ap = tf.reduce_sum(tf.square(anchor - positive), -1)
  gap_an = tf.reduce_sum(tf.square(anchor - negative), -1)
  # tf.print(gap_ap, gap_an)
  loss = gap_ap - gap_an + alpha
  loss = tf.maximum(loss, 0.0)
  tf.print(loss)
  return tf.reduce_mean(loss)