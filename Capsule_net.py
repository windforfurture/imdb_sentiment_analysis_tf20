import tensorflow as tf
from tensorflow.keras import backend as bk
from tensorflow.keras.layers import Layer, Activation


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = bk.sum(bk.square(x), axis, keepdims=True) + bk.epsilon()
    # scale = bk.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = bk.sum(bk.square(x), axis, keepdims=True)
    scale = bk.sqrt(s_squared_norm + bk.epsilon())
    return x / scale


def custom_batch_dot(x, y, axes=None):
    x_ndim = bk.ndim(x)
    y_ndim = bk.ndim(y)

    diff = y_ndim - x_ndim
    x = tf.reshape(x, tf.concat([tf.shape(x), [1] * diff], axis=0))

    adj_x = None if axes[0] == bk.ndim(x) - 1 else True
    adj_y = True if axes[1] == bk.ndim(y) - 1 else None
    out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)

    idx = x_ndim - 1
    out = tf.squeeze(out, list(range(idx, idx + diff)))

    return out


class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        self.W = 0
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs,  **kwargs):
        if self.share_weights:
            u_hat_vecs = bk.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = bk.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = bk.shape(u_vecs)[0]
        input_num_capsule = bk.shape(u_vecs)[1]
        u_hat_vecs = bk.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                             self.num_capsule, self.dim_capsule))

        u_hat_vecs = bk.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = bk.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        outputs = None
        for i in range(self.routings):
            b = bk.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = bk.softmax(b)
            c = bk.permute_dimensions(c, (0, 2, 1))
            b = bk.permute_dimensions(b, (0, 2, 1))
            # print(c.shape) # [None, 32, None]
            # print(u_hat_vecs.shape) # [None , 10, None, 32]

            cal = custom_batch_dot(c, u_hat_vecs, [2, 2])
            outputs = self.activation(cal)
            # print(outputs.shape)

            # outputs = self.activation(bk.batch_dot(c, u_hat_vecs, [2, 2]))

            if i < self.routings - 1:
                b = custom_batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return None, self.num_capsule, self.dim_capsule
