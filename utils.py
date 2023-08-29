import tensorflow as tf
import numpy as np

class PeriodicPadding(tf.keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
    def call(self, inputs, *args, **kwargs):
        x = tf.concat([inputs[:, -self.padding:], inputs, inputs[:, :self.padding]], axis=1)
        x = tf.concat([x[:, :, -self.padding:], x, x[:, :, :self.padding]], axis=2)
        return x


class AddNoise(tf.keras.layers.Layer):
    def __init__(self, noise_level=0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level
    def call(self, inputs, *args, **kwargs):
        return inputs * (1. - self.noise_level) + tf.random.normal(tf.shape(inputs), stddev=self.noise_level)


class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs, *args, **kwargs):
        return tf.math.l2_normalize(inputs, axis=-1)


def spin2rgb(X):
    # X.shape == (height, width, channels)
    def normalize(v, axis=-1):
        norm = np.linalg.norm(v, ord=2, axis=axis, keepdims=True)
        return norm, np.nan_to_num(v / norm)

    def hsv2rgb(hsv):
        hsv = np.asarray(hsv)
        if hsv.shape[-1] != 3: raise ValueError(
            "Last dimension of input array must be 3; " "shape {shp} was found.".format(shp=hsv.shape))
        in_shape = hsv.shape
        hsv = np.array(hsv, copy=False, dtype=np.promote_types(hsv.dtype, np.float32), ndmin=2)

        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        r, g, b = np.empty_like(h), np.empty_like(h), np.empty_like(h)

        i = (h * 6.0).astype(int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        idx = i % 6 == 0
        r[idx], g[idx], b[idx] = v[idx], t[idx], p[idx]

        idx = i == 1
        r[idx], g[idx], b[idx] = q[idx], v[idx], p[idx]

        idx = i == 2
        r[idx], g[idx], b[idx] = p[idx], v[idx], t[idx]

        idx = i == 3
        r[idx], g[idx], b[idx] = p[idx], q[idx], v[idx]

        idx = i == 4
        r[idx], g[idx], b[idx] = t[idx], p[idx], v[idx]

        idx = i == 5
        r[idx], g[idx], b[idx] = v[idx], p[idx], q[idx]

        idx = s == 0
        r[idx], g[idx], b[idx] = v[idx], v[idx], v[idx]

        rgb = np.stack([r, g, b], axis=-1)
        return rgb.reshape(in_shape)

    norm, normed_X = normalize(X)
    norm = np.clip(norm, 0, 1)
    X = norm * normed_X
    sxmap, symap, szmap = np.split(X, 3, axis=-1)
    szmap = 0.5 * szmap + (norm / 2.)
    H = np.clip(-np.arctan2(sxmap, -symap) / (2 * np.pi) + 0.5, 0, 1)
    S = np.clip(2 * np.minimum(szmap, norm - szmap), 0, norm)
    V = np.clip(2 * np.minimum(norm, szmap + norm / 2.) - 1.5 * norm + 0.5, 0.5 - 0.5 * norm, 0.5 + 0.5 * norm)
    img = np.concatenate((H, S, V), axis=-1)
    for i, map in enumerate(img): img[i] = hsv2rgb(map)
    return img


def compute_skyrmion_number(spin_map):
    # spin_map.shape == (batch, height, width, channels)
    spin_map_a = spin_map
    spin_map_b = tf.concat([spin_map[:, 1:, :, :], spin_map[:, 0:1, :, :]], axis=1)
    spin_map_c = tf.concat([spin_map[:, :, 1:, :], spin_map[:, :, 0:1, :]], axis=2)
    spin_map_d = tf.concat([spin_map_b[:, :, 1:, :], spin_map_b[:, :, 0:1, :]], axis=2)

    absabc = tf.reduce_sum(tf.multiply(spin_map_a, tf.linalg.cross(spin_map_b, spin_map_c)), axis=-1)
    absdbc = tf.reduce_sum(tf.multiply(spin_map_d, tf.linalg.cross(spin_map_b, spin_map_c)), axis=-1)

    omega1 = 2 * tf.atan(absabc / (1 + tf.reduce_sum(tf.multiply(spin_map_a, spin_map_b), axis=-1)
                                   + tf.reduce_sum(tf.multiply(spin_map_a, spin_map_c), axis=-1)
                                   + tf.reduce_sum(tf.multiply(spin_map_b, spin_map_c), axis=-1)))
    omega2 = 2 * tf.atan(absdbc / (1 + tf.reduce_sum(tf.multiply(spin_map_d, spin_map_b), axis=-1)
                                   + tf.reduce_sum(tf.multiply(spin_map_d, spin_map_c), axis=-1)
                                   + tf.reduce_sum(tf.multiply(spin_map_b, spin_map_c), axis=-1)))
    solid_angle = tf.reduce_sum(omega1 - omega2, axis=[1, 2])
    return solid_angle/((4. * np.pi))
