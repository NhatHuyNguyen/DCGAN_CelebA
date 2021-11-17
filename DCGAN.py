import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *

IMG_PATH = 'D:/tensorflow_datasets'
BATCH_SIZE = 32
DIS_RELU_ALPHA = 0.2
DROPOUT = 0.2
INPUT_SHAPE = (64, 64, 3)
KERNEL = 4
STRIDE = 2
PADDING = "same"
LATENT_DIM = 128
GEN_RELU_ALPHA = 0.2

# fix blast XGEMM failed
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Load dataset
ds = tfds.load('celeb_a', split='train+test+validation', data_dir=IMG_PATH, shuffle_files=True, download=False)
ds = ds.batch(BATCH_SIZE)

# resize image
def transform_images(row, size):
    x_train = tf.image.resize(row['image'], size)
    x_train = (x_train - 127.5)/127.5
    return x_train

ds = ds.map(lambda row:transform_images(row, (64, 64)))
ds = ds.shuffle(1024).prefetch(tf.data.experimental.AUTOTUNE)

# Discriminator
dis = tf.keras.Sequential()
dis.add(layers.InputLayer(input_shape=INPUT_SHAPE))
dis.add(layers.Conv2D(64, kernel_size=KERNEL, strides=STRIDE, padding=PADDING))
dis.add(layers.LeakyReLU(alpha=DIS_RELU_ALPHA))
# dis.add(layers.BatchNormalization())        
dis.add(layers.Conv2D(128, kernel_size=KERNEL, strides=STRIDE, padding=PADDING))
dis.add(layers.LeakyReLU(alpha=DIS_RELU_ALPHA))
# dis.add(layers.BatchNormalization())
dis.add(layers.Conv2D(128, kernel_size=KERNEL, strides=STRIDE, padding=PADDING))
dis.add(layers.LeakyReLU(alpha=DIS_RELU_ALPHA))
# dis.add(layers.BatchNormalization())
dis.add(layers.Flatten())
dis.add(layers.Dropout(DROPOUT))
dis.add(layers.Dense(1, activation="sigmoid"))


# Generator
gen = tf.keras.Sequential()
gen.add(layers.InputLayer(input_shape=(LATENT_DIM,)))
gen.add(layers.Dense(8*8*LATENT_DIM))
gen.add(layers.Reshape((8, 8, LATENT_DIM)))
gen.add(layers.Conv2DTranspose(128, kernel_size=KERNEL, strides=STRIDE, padding=PADDING))
# gen.add(layers.LeakyReLU(alpha=GEN_RELU_ALPHA))
gen.add(layers.ReLU())
# gen.add(layers.BatchNormalization())
gen.add(layers.Conv2DTranspose(256, kernel_size=KERNEL, strides=STRIDE, padding=PADDING))
# gen.add(layers.LeakyReLU(alpha=GEN_RELU_ALPHA))
gen.add(layers.ReLU())
# gen.add(layers.BatchNormalization())
gen.add(layers.Conv2DTranspose(512, kernel_size=KERNEL, strides=STRIDE, padding=PADDING))
# gen.add(layers.LeakyReLU(alpha=GEN_RELU_ALPHA))
gen.add(layers.ReLU())
# gen.add(layers.BatchNormalization())
gen.add(layers.Conv2D(3, kernel_size=5, padding=PADDING, activation="tanh"))
KERNEL

# GAN model, consisting of discriminator and generator
class GAN(tf.keras.Model):
    def __init__(self, dis, gen, latent_dim):
        super(GAN, self).__init__()
        self.dis = dis
        self.gen = gen
        self.latent_dim = latent_dim
        
    def compile(self, d_opt, g_opt, loss_fn):
        super(GAN, self).compile()
        self.d_opt = d_opt
        self.g_opt = g_opt
        self.loss_fn = loss_fn
        self.d_loss = Mean()
        self.g_loss = Mean()

    @property
    def metrics(self):
        return [self.d_loss, self.g_loss]

    def train_step(self, imgs):
        # Sample random noise in the latent space
        batch_size = tf.shape(imgs)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # fake image
        fake_imgs = self.gen(noise)

        # Combine with real imgs
        combined_imgs = tf.concat([fake_imgs, imgs], axis=0)

        # Make real and fake labels
        real_labels = tf.zeros((batch_size, 1))
        fake_labels = tf.ones((batch_size, 1))
        # fake = 1; real = 0
        labels = tf.concat([fake_labels, real_labels], axis=0)
        # label smoothing
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
    
        # Train the discriminator - try to differentiate between real and fake
        # Generator not trained here
        with tf.GradientTape() as tape:
            preds = self.dis(combined_imgs)
            # loss(true labels, pred labels)
            d_loss = self.loss_fn(labels, preds)
        grads = tape.gradient(d_loss, self.dis.trainable_weights)
        self.d_opt.apply_gradients(zip(grads, self.dis.trainable_weights))
 
        # Sample random points in the latent space
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the generator - try to fool the discriminator into classifying all real
        # Discriminator not trained here
        with tf.GradientTape() as tape:
            preds = self.dis(self.gen(noise))
            g_loss = self.loss_fn(real_labels, preds)
        grads = tape.gradient(g_loss, self.gen.trainable_weights)
        self.g_opt.apply_gradients(zip(grads, self.gen.trainable_weights))

        # Update d and g losses
        self.d_loss.update_state(d_loss)
        self.g_loss.update_state(g_loss)
        
        metric_dict = {"d_loss": self.d_loss.result(),"g_loss": self.g_loss.result()}
        return metric_dict

# Callback after an epoch
class GANCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt, ckpt_manager, num_img=1, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.ckpt_manager = ckpt_manager
        self.ckpt = ckpt

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt.start_epoch.assign_add(1)
        self.ckpt_manager.save() 
        if (epoch + 1) % 5 == 0:
            noise = tf.random.normal(shape=(self.num_img, latent_dim))
            generated_images = self.model.gen(noise)
            generated_images = (generated_images + 1) / 2. # convert back to [0, 1]
            generated_images = generated_images.numpy()
            
            plt.imshow(generated_images[0])
            plt.axis("off")
            plt.show()


gan = GAN(dis=dis, gen=gen, latent_dim=LATENT_DIM)
gan.compile(
    d_opt=Adam(learning_rate=0.0001),
    g_opt=Adam(learning_rate=0.0001),
    loss_fn=BinaryCrossentropy(),
)

# Save checkpoint for future training
checkpoint_dir = 'D:/ckpts'
checkpoint = tf.train.Checkpoint(
    start_epoch=tf.Variable(1),
    gen=gen,
    dis=dis,
    gan=gan)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# Retrain or train from start
RETRAIN = False
START_EPOCH = 1
if RETRAIN:
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    START_EPOCH = checkpoint.start_epoch.numpy()

print("Starting training from Epoch", START_EPOCH)

EPOCHS = 50
history = gan.fit(
    ds, epochs=EPOCHS, callbacks=[GANCallback(checkpoint, ckpt_manager, num_img=1, latent_dim=latent_dim)]
)

# Plot the generated images
num_imgs = 9
noise = tf.random.normal(shape=(num_imgs, latent_dim))
generated_images = gen(noise)
generated_images = (generated_images + 1) / 2. # convert back to [0, 1]
generated_images = generated_images.numpy()
plt.figure(1, figsize=(8,8))
for i in range(num_imgs):
    plt.subplot(3,3,i+1) 
    plt.imshow(generated_images[i])
    plt.axis("off")
    img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
    img.save("generated_%d.png" % (i))
plt.show()
