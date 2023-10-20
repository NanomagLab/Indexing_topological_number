import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from utils import spin2rgb, compute_skyrmion_number

images = [Image.open("examples/" + path, 'r') for path in os.listdir("examples")]
inputs = [-np.array(image)[..., :1] * 2. / 255. + 1. for image in images]
model = tf.keras.models.load_model("model")
outputs = [model.predict(x[None, ...], verbose = 0)[0] for x in inputs]
skyrmion_numbers = [compute_skyrmion_number(output[None, ...])[0] for output in outputs]

fig, axes = plt.subplots(2, len(images),  figsize=(len(images) * 1.5,4))
for i in range(len(images)):
    axes[0][i].imshow(images[i])
    axes[0][i].axis('off')
    axes[0][i].set_title(os.listdir("examples")[i])
    axes[1][i].imshow(spin2rgb(outputs[i]))
    axes[1][i].axis('off')
    axes[1][i].set_title("n={:0.2f}".format(skyrmion_numbers[i]), y=-0.2)
plt.tight_layout()
plt.show()
