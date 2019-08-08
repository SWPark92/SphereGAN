import tensorflow as tf
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Conv2D, Deconv2D, Input, Dense, Reshape, Flatten, Lambda, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal, glorot_normal

def stereo_graphic_proj(u): 
    p = tf.divide(2*tf.transpose(u), tf.pow(tf.norm(u, axis=1),2) +1)   
    tmp = tf.divide((tf.pow(tf.norm(u, axis=1),2) -1 ), (tf.pow(tf.norm(u, axis=1),2) +1 ) )
    p = tf.concat([p, [tmp]],axis=0)
    p = tf.transpose(p)
    return p

def LN(u):
    u = tf.contrib.layers.layer_norm(u)
    return u

def Generator(noise_dim, batch_size, f=512):    
    img_dim = (32,32,3)
    s = img_dim[1]
    output_channels = img_dim[-1]
    start_dim = 4
    nb_upconv = 3
 
    reshape_shape = (start_dim, start_dim, f)
    gen_input = Input(shape=noise_dim)
    x = Dense(f * start_dim * start_dim, input_dim=noise_dim, use_bias=False)(gen_input)
    x = Reshape(reshape_shape)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)

    for i in range(nb_upconv):
        nb_filters = int(f / (2 ** (i + 1)))
        s = start_dim * (2 ** (i + 1))
        o_shape = (batch_size, s, s, nb_filters)
        x = Deconv2D(nb_filters, (4, 4),
                     output_shape=o_shape, strides=(2, 2),
                     padding="same", use_bias=False,
                     kernel_initializer = glorot_normal())(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

    x = Conv2D(output_channels, (3, 3), strides=(1, 1), padding="same", use_bias=False,
               kernel_initializer = glorot_normal())(x)
    x = Activation("tanh")(x)

    generator_model = Model(inputs=[gen_input], outputs=[x])
    return generator_model

def Discriminator(feature_dim, f=64):
    img_dim = (32,32,3)
    nb_conv = 3
    slope = 0.2
    norm_ = True
    g_b = 'GAP'
    
    disc_input = Input(shape=img_dim)
    x = disc_input
     
    for i in range(nb_conv):
        x = Conv2D(f, (3, 3), strides=(1, 1), padding="same", use_bias=False,
                   kernel_initializer = glorot_normal())(x)
        if norm_:
            x = Lambda(LN)(x) 
        x = LeakyReLU(slope)(x)   
        x = Conv2D(f, (4, 4), strides=(2, 2), padding="same", use_bias=False,
                   kernel_initializer = glorot_normal())(x)
        if norm_:
             x = Lambda(LN)(x)
             x = x
        x = LeakyReLU(slope)(x)     
        f *= 2

    x = Conv2D(f, (3, 3), strides=(1, 1), padding="same", use_bias=False,
                   kernel_initializer = glorot_normal())(x)
    if norm_:
        x = Lambda(LN)(x)
    x = LeakyReLU(slope)(x)         
    
    if g_b == 'GAP':
        x = GlobalAveragePooling2D()(x)
        x = Dense(feature_dim)(x)  
    if g_b == 'FC':
        x = Flatten()(x)
        x = Dense(units = feature_dim, kernel_initializer = RandomNormal(stddev=0.5))(x)
    x = Lambda(stereo_graphic_proj)(x)
    discriminator_model = Model(inputs=[disc_input], outputs=[x])
    return discriminator_model