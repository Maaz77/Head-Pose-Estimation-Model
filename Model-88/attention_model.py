import tensorflow as tf
from keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Reshape,
    Dense,
    Multiply,
    Conv2D,
    MultiHeadAttention,
    Add,
    LayerNormalization,
    Lambda
)
import keras 

def se_transformer_regr_head(
    input_channels=88,
    reduction=16,
    num_heads=4,
    key_dim=16,
    ff_dim=64,
    hidden_channels=128
):
    """
    Builds a regressor head with:
      • input: (batch, H, W, input_channels)
      • channel SE gating
      • small Transformer encoder on channel descriptors
      • output: (batch, H, W, 3)
    """
    x_in = Input(shape=(None, None, input_channels))  # dynamic H, W

    # 1) Squeeze-and-Excitation channel attention
    se = GlobalAveragePooling2D()(x_in)                                # → (batch, C)
    se = Dense(input_channels // reduction, activation='relu')(se)      # → (batch, C/reduction)
    se = Dense(input_channels, activation='sigmoid')(se)               # → (batch, C)
    se = Reshape((1, 1, input_channels))(se)                           # → (batch,1,1,C)
    x = Multiply()([x_in, se])                                         # → (batch,H,W,C)

    # 2) Prepare channel descriptors for self-attention
    #    flatten spatial dims: (batch, H*W, C)
    def reshape_flat(t):
        B = tf.shape(t)[0]
        H = tf.shape(t)[1]
        W = tf.shape(t)[2]
        C = tf.shape(t)[3]
        return tf.reshape(t, (B, H * W, C))

    flat = Lambda(reshape_flat)(x)                                      # → (batch, HW, C)

    # 3) Self-attention across spatial positions (optional for >1x1)
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(flat, flat)                                                       # → (batch, HW, C)
    x_attn = Add()([flat, attn_out])                                    # residual
    x_attn = LayerNormalization()(x_attn)

    # 4) Feed-Forward within Transformer block
    ff = Dense(ff_dim, activation='relu')(x_attn)
    ff = Dense(input_channels)(ff)
    x_ff = Add()([x_attn, ff])                                          # residual
    x_ff = LayerNormalization()(x_ff)

    # 5) reshape back to (batch, H, W, C)
    def reshape_back(tensors):
        t, orig = tensors
        B = tf.shape(orig)[0]
        H = tf.shape(orig)[1]
        W = tf.shape(orig)[2]
        C = tf.shape(t)[2]
        return tf.reshape(t, (B, H, W, C))

    x2 = Lambda(reshape_back)([x_ff, x_in])                             # → (batch,H,W,C)

    # 6) Regression head via 1×1 convs
    x2 = Conv2D(hidden_channels, kernel_size=1, activation='relu')(x2)  # → (batch,H,W,hidden)
    out = Conv2D(3, kernel_size=1, activation=None)(x2)                  # → (batch,H,W,3)

    return keras.Model(inputs=x_in, outputs=out, name='SE_Transformer_Regr')

def create_modelC():
    inp = keras.Input((None,None,88))
    # 1) SE block (r=8 → 88/8=11):
    se = keras.layers.GlobalAveragePooling2D()(inp)       # → (batch, 88)
    se = keras.layers.Dense(11, activation='relu')(se)   # 88*11+11= 979
    se = keras.layers.Dense(88, activation='sigmoid')(se)# 11*88+88=1 056
    se = keras.layers.Reshape((1,1,88))(se)
    x  = keras.layers.Multiply()([inp, se])              # → gated (H,W,88)

    # 2) 1×1 conv head
    x  = keras.layers.Conv2D(42, 1, activation='relu')(x)  # 88*42+42 = 3 738
    out= keras.layers.Conv2D( 3, 1, activation=None)(x)    # 42*3+3   =   129

    return keras.Model(inp, out)

def create_model_complex(reg , dr):
    """
    Fully convolutional regressor with multiple residual blocks (skip connections),
    spatial dropout, and L2 regularization.

    Input: (batch, H, W, 88)
    Output: (batch, H, W, 3)
    """
    reg = keras.regularizers.l2(reg)
    inputs = keras.Input(shape=(None, None, 88))

    # --- Initial projection ---
    x = keras.layers.Conv2D(
        filters=16,
        kernel_size=1,
        padding='same',
        activation='softsign',
        kernel_regularizer=reg,
        kernel_initializer='glorot_uniform'
    )(inputs)
    x = keras.layers.SpatialDropout2D(dr)(x)

    # --- Residual block definition ---
    def res_block(inp, filters=16):
        y = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation='softsign',
            kernel_regularizer=reg,
            kernel_initializer='glorot_uniform'
        )(inp)
        y = keras.layers.SpatialDropout2D(dr)(y)
        y = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation='softsign',
            kernel_regularizer=reg,
            kernel_initializer='glorot_uniform'
        )(y)
        y = keras.layers.SpatialDropout2D(dr)(y)
        out = keras.layers.Add()([inp, y])
        return keras.layers.Activation('relu')(out)

    # --- Stack several residual blocks ---
    x = res_block(x, 16)
    x = res_block(x, 16)
    x = res_block(x, 16)

    # --- Bottleneck projection ---
    x = keras.layers.Conv2D(
        filters=8,
        kernel_size=1,
        padding='same',
        activation='softsign',
        kernel_regularizer=reg,
        kernel_initializer='glorot_uniform'
    )(x)
    x = keras.layers.SpatialDropout2D(dr)(x)

    # --- Final output conv ---
    outputs = keras.layers.Conv2D(
        filters=3,
        kernel_size=1,
        padding='same',
        activation=None,
        kernel_regularizer=reg,
        kernel_initializer='glorot_uniform'
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='Complex_Conv_Skip_Model')
    return model











if __name__ == '__main__':
    # Example usage
    pass
    # model = se_transformer_regr_head(
    #     input_channels=88,
    #     reduction=8,
    #     num_heads=4,
    #     key_dim=16,
    #     ff_dim=64,
    #     hidden_channels=64
    # )
    # model.summary()

