import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.datasets import cifar10
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from model_sphere_gan import Generator, Discriminator

class SphereGAN():
    def get_reference_point(self, coord=None):
        if coord == None:
            ref_p_np = np.zeros( (1,self.feature_dim+1 ) ).astype(np.float32)
            ref_p_np[0,self.feature_dim] = 1.0
            return tf.constant(ref_p_np)
        else:
            return coord
    def _dist_sphere(self, a,b,r):
        return tf.acos( tf.matmul(a , tf.transpose(b))) ** r
    def dist_weight_mode(self, r):
        if self.weight_mode == 'normalization':
            decayed_dist = ( (1/self.decay_ratio)*np.pi)**r
        elif self.weight_mode == 'half':
            decayed_dist = (np.pi)**r
        else:
            decayed_dist = 1
        return decayed_dist
    
    def eval_moments(self, y_true, y_pred):
        ref_p = self.get_reference_point()
        d = 0.0
        for r in range(1, self.moments + 1):
            d = d + self._dist_sphere(y_pred, ref_p, r) / self.dist_weight_mode(r)
        return K.mean(y_true * d)
    def __init__(self):
        self.img_shape = (32, 32, 3)
        self.batch_size = 64
        self.latent_dim = 128
        self.feature_dim = 1024
        self.nb_learning = 1
        self.moments = 1 # [3] is suggested but [1] is enough.
        self.epochs = int(5E+5)
        self.sample_interval = 100
        self.weight_mode = None
        self.loss_mode = None
        self.decay_ratio = 3
        
        
        optimizer_D = Adam(lr=1e-4, beta_1=0.0, beta_2=0.9)
        optimizer_G = Adam(lr=1e-4, beta_1=0.0, beta_2=0.9)

        self.generator = Generator( (self.latent_dim,) ,self.batch_size)
        self.discriminator = Discriminator(self.feature_dim, self.batch_size)
        
        self.generator.summary()
        self.discriminator.summary()
        
        self.generator.trainable = False
        real_img = Input(shape=self.img_shape)
        z_disc = Input(shape=(self.latent_dim,))
        fake_img = self.generator(z_disc)
        fake = self.discriminator(fake_img)
        real = self.discriminator(real_img)

        self.discriminator_model = Model(inputs=[real_img, z_disc], outputs=[real, fake])
        self.discriminator_model.compile(loss=[self.eval_moments,self.eval_moments], optimizer=optimizer_D)
        self.discriminator.trainable = False
        self.generator.trainable = True
        
        z_gen = Input(shape=(self.latent_dim,))
        img = self.generator(z_gen)
        fake_img = self.discriminator(img)
        self.generator_model = Model(z_gen, fake_img)
        self.generator_model.compile(loss=self.eval_moments, optimizer=optimizer_G)

    def train(self):
        batch_size = self.batch_size
        epochs = self.epochs
        (X_train, _), (_, _) = cifar10.load_data()
        X_train = (X_train.astype(np.float32)) / 127.5 - 1.0
        positive_y = np.ones((batch_size, 1), dtype=np.float32)
        negative_y = -positive_y
        for epoch in range(1,epochs+1):
            for _ in range(self.nb_learning):
                imgs = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                d_loss = self.discriminator_model.train_on_batch([imgs, noise], [negative_y, positive_y])

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.generator_model.train_on_batch(noise, negative_y)        
            d_loss = d_loss[0] + d_loss[1]
            
if __name__ == '__main__':
    SphereGAN = SphereGAN()
    SphereGAN.train()