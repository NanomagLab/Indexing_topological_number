import tensorflow as tf
import numpy as np
from utils import AddNoise, PeriodicPadding, L2Normalize

tf.random.set_seed(0)

NOISE_LEVEL = 0.1

x_train = np.load("TrainData.npy")[:30000]

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((None, None, 1)),
    AddNoise(noise_level=NOISE_LEVEL),
    PeriodicPadding(padding=1),
    tf.keras.layers.Conv2D(1, 3, activation='tanh'),
    PeriodicPadding(padding=1),
    tf.keras.layers.Conv2D(3, 3),
    L2Normalize()
])
model.compile('adam', 'mse')
model.summary()
model.fit(
    x_train[..., -1:],
    x_train,
    batch_size=100,
    epochs=20
)

model = tf.keras.models.Sequential([tf.keras.layers.Input((None, None, 1))] + model.layers[1:])
model.save("model")
