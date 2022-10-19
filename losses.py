import keras.backend as K
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model
import numpy as np

# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)

def color_different(y_true, y_pred):
    loss_rgb = K.square(y_true-y_pred)
    loss_r =K.mean(tf.multiply(3.0, loss_rgb[:,:,:,0]))
    loss_g =K.mean(tf.multiply(4.0, loss_rgb[:,:,:,1]))
    loss_b =K.mean(tf.multiply(2.0, loss_rgb[:,:,:,2]))
    loss_rg = tf.add(loss_r, loss_g)   
    loss_rgb = tf.add(loss_rg, loss_b)   
    return loss_rgb


def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.20):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

epsilon = 1e-5
smooth = 1

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def pix2pix_loss(y_true, y_pred):
    return K.mean(K.square(y_true-y_pred))


def perceptual_loss_1000(y_true, y_pred):
    return 1000 * perceptual_loss(y_true, y_pred)

def perceptual_loss_10(y_true, y_pred):
    return 10 * perceptual_loss(y_true, y_pred)

def perceptual_loss_10000(y_true, y_pred):
    return 10000 * perceptual_loss(y_true, y_pred)

def perceptual_loss_100(y_true, y_pred):
    return 100 * perceptual_loss(y_true, y_pred)


def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    a = 0.5 
    b = 0.7
    clear = 0
    if clear:
        a = 0.01
        b = 0.01
    return a*K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))+b*pix2pix_loss(y_true, y_pred)


def wasserstein_loss(y_true, y_pred):
    return 100*K.mean(y_true*y_pred)


def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)

