import tensorflow as tf
from keras import Model
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Layer,
    LayerNormalization,
    LeakyReLU,
    MaxPooling1D,
    Reshape,
    TimeDistributed,
)

SEED = 42
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
INITIALIZER = tf.keras.initializers.GlorotUniform(SEED)


def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.15)


class ConvolutionBlock(Layer):
    def __init__(self, n_filter, n_kernel=3):
        super().__init__()
        self.convs = [
            Conv1D(
                filters=n_filter,
                kernel_size=n_kernel,
                padding="same",
                activation=leaky_relu,
                kernel_initializer=INITIALIZER,
                bias_initializer=INITIALIZER,
            )
            for _ in range(3)
        ]
        self.pool = MaxPooling1D(pool_size=2, strides=2)
        self.residual_conv = Conv1D(
            filters=n_filter,
            kernel_size=1,
            padding="same",
            activation=leaky_relu,
            kernel_initializer=INITIALIZER,
            bias_initializer=INITIALIZER,
        )
        self.residual_pool = MaxPooling1D(pool_size=2, strides=2)
        self.add = Add()

    def call(self, inputs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)

        residual = self.residual_conv(inputs)
        residual = self.residual_pool(residual)
        return self.add([x, residual])


class DilatedConvBlock(Layer):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.convs = [
            Conv1D(
                filters=128,
                kernel_size=7,
                dilation_rate=rate,
                padding="same",
                activation=leaky_relu,
                kernel_initializer=INITIALIZER,
                bias_initializer=INITIALIZER,
            )
            for rate in (1, 2, 4, 8, 16, 32)
        ]
        self.dropout = Dropout(dropout_rate)
        self.add = Add()

    def call(self, inputs, training=None):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        x = self.dropout(x, training=training)
        return self.add([x, inputs])


def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    depth = tf.cast(tf.shape(k)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MyMultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.wq = Dense(d_model, dtype="float32")
        self.wk = Dense(d_model, dtype="float32")
        self.wv = Dense(d_model, dtype="float32")
        self.out = Dense(d_model, dtype="float32")

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        attention, weights = scaled_dot_product_attention(q, k, v)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.d_model))
        return self.out(attention), weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            Dense(dff, activation="relu"),
            Dense(d_model),
        ]
    )


class TransformerEncoder(Layer):
    def __init__(self, d_model, num_heads, rate=0.1):
        super().__init__()
        self.mha = MyMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_model * 4)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output, _ = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class FeatureShortRange(Layer):
    def __init__(self):
        super().__init__()
        self.blocks = [
            ConvolutionBlock(16),
            ConvolutionBlock(16),
            ConvolutionBlock(32),
            ConvolutionBlock(32),
            ConvolutionBlock(64),
            ConvolutionBlock(64),
            ConvolutionBlock(128),
            ConvolutionBlock(256),
        ]
        self.window = Reshape(target_shape=(1024,))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(inputs, [batch_size * 1200, 1024, 1])
        for block in self.blocks:
            x = block(x)
        x = self.window(x)
        return tf.reshape(x, [batch_size, 1200, 1024])


class StackedConv(Layer):
    def __init__(self, filters, rate, n_kernel=9):
        super().__init__()
        self.convs = [
            Conv1D(
                filters=filters,
                kernel_size=n_kernel,
                dilation_rate=dilation,
                padding="same",
                activation=leaky_relu,
                kernel_initializer=INITIALIZER,
                bias_initializer=INITIALIZER,
            )
            for dilation in (rate, 2 * rate, 4 * rate)
        ]
        self.batchnormal = BatchNormalization(dtype="float32")
        self.maxpool = MaxPooling1D(pool_size=2, strides=2, padding="same")
        self.dropout = Dropout(0.2)

    def call(self, inputs, training=None):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        x = self.batchnormal(x, training=training)
        x = self.maxpool(x)
        return self.dropout(x, training=training)


class AHIConvModel(Layer):
    def __init__(self):
        super().__init__()
        self.blocks = [
            StackedConv(filters=64, rate=1),
            StackedConv(filters=64, rate=2),
            StackedConv(filters=32, rate=3),
            StackedConv(filters=16, rate=4),
        ]
        self.flatten = Flatten()
        self.fc = Dense(256, dtype="float32")

    def call(self, inputs, training=None):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        x = self.flatten(x)
        return self.fc(x)


class AHIPredModel(Layer):
    def __init__(self, rate=0.2):
        super().__init__()
        widths = (64, 32, 16)
        self.fcs = [Dense(units=width, dtype="float32") for width in widths]
        self.acts = [LeakyReLU(alpha=0.15) for _ in widths]
        self.drops = [Dropout(rate) for _ in widths]
        self.fc_out = Dense(units=1, dtype="float32")
    def call(self, inputs, training=None):
        x = inputs
        for fc, act, drop in zip(self.fcs, self.acts, self.drops):
            x = fc(x)
            x = act(x)
            x = drop(x, training=training)
        return self.fc_out(x)

# class AHIPredModel(Layer):
#     def __init__(self, rate=0.2):
#         super().__init__()
#         widths = (64, 32, 16)
#         self.fcs = [Dense(units=width, dtype="float32") for width in widths]
#         # LayerNorm is more stable than BatchNorm for the subject-level MLP
#         # because the effective per-replica batch size is very small in training.
#         self.norms = [LayerNormalization(epsilon=1e-6, dtype="float32") for _ in widths]
#         self.acts = [LeakyReLU(alpha=0.15) for _ in widths]
#         self.drops = [Dropout(rate) for _ in widths]
#         self.fc_out = Dense(units=1, dtype="float32")

#     def call(self, inputs, training=None):
#         x = inputs
#         for fc, norm, act, drop in zip(self.fcs, self.norms, self.acts, self.drops):
#             x = fc(x)
#             x = norm(x)
#             x = act(x)
#             x = drop(x, training=training)
#         return self.fc_out(x)
class OximetryResampleToEpoch1024(Layer):
    def __init__(self, n_epoch=1200, src_len_per_epoch=30, tgt_len_per_epoch=1024, **kwargs):
        super().__init__(**kwargs)
        self.n_epoch = n_epoch
        self.src_len_per_epoch = src_len_per_epoch
        self.tgt_len_per_epoch = tgt_len_per_epoch

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(inputs, (batch_size, self.n_epoch, self.src_len_per_epoch, 1))
        x = tf.image.resize(
            x,
            size=(self.n_epoch, self.tgt_len_per_epoch),
            method="bilinear",
        )
        mean, variance = tf.nn.moments(x, axes=[2], keepdims=True)
        x = (x - mean) / tf.sqrt(variance + 1e-6)
        return tf.reshape(x, (batch_size, self.n_epoch * self.tgt_len_per_epoch, 1))


class MaskedJointSelfAttention(Layer):
    def __init__(self, d_model=64, num_heads=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.out_proj = Dense(d_model)
        self.attn_dropout = Dropout(dropout)
        self.out_dropout = Dropout(dropout)

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, key_bias=None, training=None):
        batch_size = tf.shape(inputs)[0]
        compute_dtype = inputs.dtype

        q = self._split_heads(self.wq(inputs), batch_size)
        k = self._split_heads(self.wk(inputs), batch_size)
        v = self._split_heads(self.wv(inputs), batch_size)

        # Keep projections compatible with mixed precision, but compute the
        # attention logits/softmax path in float32 for numerical stability.
        q = tf.cast(q, tf.float32)
        k = tf.cast(k, tf.float32)
        v = tf.cast(v, tf.float32)

        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if key_bias is not None:
            scores += tf.cast(key_bias[:, tf.newaxis, tf.newaxis, :], tf.float32)

        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)
        attn_weights = tf.cast(attn_weights, tf.float32)

        context = tf.matmul(attn_weights, v)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.d_model))
        context = tf.cast(context, compute_dtype)
        context = self.out_proj(context)
        context = self.out_dropout(context, training=training)
        return context


class CrossModalFusion3(Layer):
    def __init__(
        self,
        feature_len=128,
        d_model=64,
        num_heads=2,
        dropout=0.1,
        mask_drop_rate=0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.feature_len = feature_len
        self.mask_drop_rate = mask_drop_rate
        self.modality_count = 3
        self.concat_len = self.modality_count * self.feature_len
        # Use a half-precision-safe large negative bias instead of materializing
        # a full (kL, kL) mask tensor.
        self.neg_large = tf.constant(-1e4, dtype=tf.float32)

        self.input_projection = Dense(d_model)
        self.joint_attention = MaskedJointSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.post_attn_norm = LayerNormalization(epsilon=1e-6)
        self.readout_norm = LayerNormalization(epsilon=1e-6)
        self.readout_proj = Dense(1)


    def _concat_modalities(self, e_respi, e_cardi, e_oxi):
        # Inputs are (B, Epoch, L), so concatenation produces a joint sequence (B, Epoch, kL).
        return tf.concat([e_respi, e_cardi, e_oxi], axis=-1)

    def _expand_keep_mask(self, keep_mask):
        return tf.concat(
            [
                tf.repeat(keep_mask[:, :, idx : idx + 1], repeats=self.feature_len, axis=-1)
                for idx in range(self.modality_count)
            ],
            axis=-1,
        )

    def _build_key_bias(self, keep_mask):
        if keep_mask is None:
            return None, None

        token_keep = self._expand_keep_mask(keep_mask)
        key_bias = (1.0 - token_keep) * self.neg_large
        key_bias = tf.reshape(key_bias, (-1, self.concat_len))
        return key_bias, token_keep

    def _build_keep_mask(self, batch_size, n_epoch, dtype, training):
        if training is None or self.mask_drop_rate <= 0.0:
            return None

        if isinstance(training, bool):
            if not training:
                return None
            training_scale = tf.constant(1.0, dtype=dtype)
        else:
            training_scale = tf.cast(training, dtype)

        apply_drop = tf.cast(
            tf.random.uniform((batch_size, n_epoch, 1)) < self.mask_drop_rate,
            dtype,
        )
        apply_drop = apply_drop * training_scale

        # Drop exactly one modality for each epoch where dropout is activated.
        drop_index = tf.random.uniform(
            (batch_size, n_epoch),
            maxval=self.modality_count,
            dtype=tf.int32,
        )
        drop_mask = tf.one_hot(drop_index, depth=self.modality_count, dtype=dtype)
        return 1.0 - apply_drop * drop_mask

    def call(self, e_respi, e_cardi, e_oxi, training=None):
        e_respi = tf.cast(e_respi, tf.float32)
        e_cardi = tf.cast(e_cardi, tf.float32)
        e_oxi = tf.cast(e_oxi, tf.float32)

        batch_size = tf.shape(e_respi)[0]
        n_epoch = tf.shape(e_respi)[1]
        keep_mask = self._build_keep_mask(batch_size, n_epoch, e_respi.dtype, training)
        x = self._concat_modalities(e_respi, e_cardi, e_oxi)
        key_bias, token_keep = self._build_key_bias(keep_mask)

        if token_keep is not None:
            x = x * tf.cast(token_keep, x.dtype)

        token_count = batch_size * n_epoch
        x = tf.reshape(x, (token_count, self.concat_len, 1))
        x = self.input_projection(x)
        if key_bias is not None:
            key_bias = tf.cast(key_bias, x.dtype)
        x = self.joint_attention(x, key_bias=key_bias, training=training)
        x = self.post_attn_norm(x)
        x = self.readout_norm(x)
        x = tf.reshape(
            x,
            (batch_size, n_epoch, self.modality_count, self.feature_len, self.d_model),
        )

        if keep_mask is not None:
            modality_keep = tf.cast(keep_mask[..., tf.newaxis, tf.newaxis], x.dtype)
            denom = tf.cast(
                tf.reduce_sum(keep_mask, axis=-1, keepdims=True)[..., tf.newaxis],
                x.dtype,
            )
            z = tf.reduce_sum(x * modality_keep, axis=2) / tf.maximum(
                denom, tf.cast(1.0, x.dtype)
            )
        else:
            z = tf.reduce_mean(x, axis=2)

        z = self.readout_proj(z)
        z = tf.cast(tf.squeeze(z, axis=-1), tf.float32)
        return z


class TemporalInformationBottleneck(Layer):
    def __init__(self, bottleneck_dim=128):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.dense_mu = Dense(bottleneck_dim, dtype="float32")
        self.dense_log_var = Dense(bottleneck_dim, dtype="float32")
        self.log_var_clip_min = -10.0
        self.log_var_clip_max = 10.0

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        n_epoch = tf.shape(inputs)[1]
        dim = tf.shape(inputs)[2]

        x = tf.reshape(inputs, (-1, dim))
        mu = tf.cast(self.dense_mu(x), tf.float32)
        log_var = tf.cast(self.dense_log_var(x), tf.float32)
        log_var = tf.clip_by_value(
            log_var, self.log_var_clip_min, self.log_var_clip_max
        )

        if training:
            std = tf.exp(0.5 * log_var)
            eps = tf.random.normal(shape=tf.shape(std), dtype=tf.float32)
            z = mu + eps * std
        else:
            z = mu

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
        )

        z = tf.reshape(z, (batch_size, n_epoch, dim))
        z = tf.cast(inputs, tf.float32) + z
        return z, kl_loss


def private_decorr_loss(u_s, u_a, eps=1e-6):
    batch_size = tf.shape(u_s)[0]
    n_epoch = tf.shape(u_s)[1]
    dim = tf.shape(u_s)[2]

    u_s = tf.reshape(u_s, (batch_size * n_epoch, dim))
    u_a = tf.reshape(u_a, (batch_size * n_epoch, dim))

    u_s = (u_s - tf.reduce_mean(u_s, axis=0, keepdims=True)) / (
        tf.math.reduce_std(u_s, axis=0, keepdims=True) + eps
    )
    u_a = (u_a - tf.reduce_mean(u_a, axis=0, keepdims=True)) / (
        tf.math.reduce_std(u_a, axis=0, keepdims=True) + eps
    )

    corr = tf.matmul(u_s, u_a, transpose_a=True) / tf.cast(tf.shape(u_s)[0], tf.float32)
    return tf.reduce_mean(tf.square(corr))


class CardioPulmoSleepNet(Model):
    def __init__(self):
        super().__init__()
        self.short_range_fea_respi = FeatureShortRange()
        self.short_range_fea_cardi = FeatureShortRange()
        self.short_range_fea_oxi = FeatureShortRange()

        self.dense_respi = Dense(128, dtype="float32")
        self.dense_cardi = Dense(128, dtype="float32")
        self.dense_oxi = Dense(128, dtype="float32")
        self.oxi_resample = OximetryResampleToEpoch1024(
            n_epoch=1200,
            src_len_per_epoch=30,
            tgt_len_per_epoch=1024,
        )

        self.fuse3 = CrossModalFusion3(
            feature_len=128,
            d_model=64,
            num_heads=2,
            dropout=0.1,
            mask_drop_rate=0.15,
        )
        self.shared_proj = Dense(128, dtype="float32")
        self.stage_proj = Dense(128, dtype="float32")
        self.ahi_proj = Dense(128, dtype="float32")

        self.stage_in_proj = Dense(128, dtype="float32")
        self.ahi_in_proj = Dense(128, dtype="float32")
        self.concat = Concatenate(axis=-1)
        self.lambda_decorr = 1.0

        self.ahiconv = AHIConvModel()
        self.ahipred = AHIPredModel()

        self.dila_block1 = DilatedConvBlock()
        self.mha_block1 = TransformerEncoder(d_model=128, num_heads=4)
        self.dila_block2 = DilatedConvBlock()
        self.mha_block2 = TransformerEncoder(d_model=128, num_heads=4)

        self.conv_pred = Conv1D(filters=4, kernel_size=1, dtype="float32")
    def _stage_aware_calibration(self, logits_sleep):
        sleep_probs = tf.nn.softmax(tf.cast(logits_sleep, tf.float32), axis=-1)
        # Down-weight epochs that are predicted as wake when estimating night-level AHI.
        calibration = 1.0 - sleep_probs[..., 0:1]
        return tf.stop_gradient(calibration)
    
    def call(self, inputs_respi, inputs_cardi, inputs_oximetry, training=None):
        e_respi = self.short_range_fea_respi(inputs_respi)
        e_cardi = self.short_range_fea_cardi(inputs_cardi)

        e_respi = self.dense_respi(e_respi)
        e_cardi = self.dense_cardi(e_cardi)

        x_oxi = self.oxi_resample(inputs_oximetry)
        e_oxi = self.short_range_fea_oxi(x_oxi)
        e_oxi = self.dense_oxi(e_oxi)

        z = self.fuse3(e_respi, e_cardi, e_oxi, training=training)

        shared = self.shared_proj(z)
        stage_private = self.stage_proj(z)
        ahi_private = self.ahi_proj(z)


        regularization_loss = private_decorr_loss(stage_private, ahi_private)
        
        stage_in = self.concat([shared, stage_private])
        stage_in = self.stage_in_proj(stage_in)
        sleep_features = self.dila_block1(stage_in, training=training)
        sleep_features = self.mha_block1(sleep_features, training=training)
        sleep_features = self.dila_block2(sleep_features, training=training)
        sleep_features = self.mha_block2(sleep_features, training=training)
        logits_sleep = tf.cast(self.conv_pred(sleep_features), tf.float32)
        
        stage_calibration = self._stage_aware_calibration(logits_sleep)
        
        ahi_in = self.concat([shared, ahi_private])
        ahi_in = self.ahi_in_proj(ahi_in)
        ahi_in = ahi_in * tf.cast(stage_calibration, ahi_in.dtype)
        ahi_feat = self.ahiconv(ahi_in, training=training)
        ahi_pred = self.ahipred(ahi_feat, training=training)

        return logits_sleep, ahi_pred, regularization_loss


if __name__ == "__main__":
    model = CardioPulmoSleepNet()
    batch_size = 1
    inputs_respi = tf.random.normal((batch_size, 1200 * 1024, 1))
    inputs_cardi = tf.random.normal((batch_size, 1200 * 1024, 1))
    inputs_oximetry = tf.random.normal((batch_size, 36000, 1))

    outputs_sleep, outputs_ahi, outputs_kl = model(
        inputs_respi,
        inputs_cardi,
        inputs_oximetry,
        training=True,
    )
    model.summary()
    print(f"Sleep staging output shape: {outputs_sleep.shape}")
    print(f"AHI prediction output shape: {outputs_ahi.shape}")
    print(f"KL loss shape: {outputs_kl.shape}")
