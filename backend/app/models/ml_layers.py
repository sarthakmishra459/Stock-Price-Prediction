import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, Dropout, LayerNormalization, Conv1D
)

class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len

    def build(self, input_shape):
        self.wl = self.add_weight(shape=(self.seq_len,), initializer='uniform')
        self.bl = self.add_weight(shape=(self.seq_len,), initializer='uniform')
        self.wp = self.add_weight(shape=(self.seq_len,), initializer='uniform')
        self.bp = self.add_weight(shape=(self.seq_len,), initializer='uniform')

    def call(self, x):
        x = tf.reduce_mean(x[:, :, :4], axis=-1)
        linear = tf.expand_dims(self.wl * x + self.bl, -1)
        periodic = tf.expand_dims(tf.sin(x * self.wp + self.bp), -1)
        return tf.concat([linear, periodic], axis=-1)

    def get_config(self):
        return {"seq_len": self.seq_len}


class SingleAttention(Layer):
    def __init__(self, d_k, d_v, **kwargs):
        super().__init__(**kwargs)
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k)
        self.key = Dense(self.d_k)
        self.value = Dense(self.d_v)

    def call(self, inputs):
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn = tf.matmul(q, k, transpose_b=True)
        attn = attn / tf.sqrt(tf.cast(self.d_k, tf.float32))
        attn = tf.nn.softmax(attn)

        v = self.value(inputs[2])
        return tf.matmul(attn, v)


class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.heads = [SingleAttention(self.d_k, self.d_v) for _ in range(self.n_heads)]
        self.linear = Dense(input_shape[0][-1])

    def call(self, inputs):
        attn = [h(inputs) for h in self.heads]
        concat = tf.concat(attn, axis=-1)
        return self.linear(concat)


class TransformerEncoder(Layer):
    def __init__(
        self,
        d_k,
        d_v,
        n_heads,
        ff_dim,
        dropout_rate=0.1,
        attn_heads=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # IMPORTANT: accept but ignore attn_heads (rebuild later)
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)

        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_norm = LayerNormalization(epsilon=1e-6)

        self.ff1 = Conv1D(self.ff_dim, 1, activation="relu")
        self.ff2 = Conv1D(input_shape[0][-1], 1)

        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = inputs[0]

        attn = self.attn_multi(inputs)
        x = self.attn_norm(x + self.attn_dropout(attn))

        ff = self.ff1(x)
        ff = self.ff2(ff)
        return self.ff_norm(x + self.ff_dropout(ff))

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_k": self.d_k,
            "d_v": self.d_v,
            "n_heads": self.n_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "attn_heads": []  # keep for compatibility
        })
        return config
