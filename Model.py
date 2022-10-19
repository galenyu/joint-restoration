import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import keras as kr
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.utils.vis_utils import plot_model
from sklearn import metrics
from keras.backend.common import normalize_data_format
from keras.regularizers import l2
from losses import *


# 本质上，是对细节纹理上的处理，maxpooling太影响细节，实际上对雨滴的处理并不好，应该在一开始就加大卷积核的size
# 在深层次的物体特征反而没那么重要
# 两条路：一条捕获雨滴特征：11*11，7*7，5*5，3*3的卷积层
#        一条捕获图像背景特征： unet（3*3）*6

Height = 256
Width = 256

def HazeUnet(input_size =  (Height,Width,3)):
    LayerName = 'SmallUnet_'
    kn = [64,64,64,64,64]
    inputs = Input(input_size)
    conv1 = Conv2D(kn[0], 3, name =LayerName+ "conv1", padding = 'valid')(inputs)
    act11 = LeakyReLU()(conv1)
    cotr1 = Conv2DTranspose(kn[0], 3, strides=(1,1), padding='valid', use_bias = True)(act11)
    act12 = LeakyReLU()(cotr1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(act12)
    bn1 = BatchNormalization(epsilon = 1.1e-5)(pool1)

    conv2 = Conv2D(kn[1], 3, name =LayerName+ "conv2", padding = 'valid')(bn1)
    act21 = LeakyReLU()(conv2)
    cotr2 = Conv2DTranspose(kn[1], 3, strides=(1,1), padding='valid', use_bias = True)(act21)
    act22 = LeakyReLU()(cotr2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(act22)
    bn2= BatchNormalization(epsilon = 1.1e-5)(pool2)

    Med0 = UpSampling2D(size = (2,2))(act22)
    merge0 = concatenate([Med0,act12], axis = 3)
    conv0 = Conv2D(kn[0], 3, name =LayerName+ "conv0", padding = 'valid')(merge0)
    act01 = LeakyReLU()(conv0)
    cotr0 = Conv2DTranspose(kn[0], 3, strides=(1,1), padding='valid', use_bias = True)(act01)
    act02 = LeakyReLU()(cotr0)
    bn0= BatchNormalization(epsilon = 1.1e-5)(act02)    


    up1 = Conv2D(kn[2], 3, name = LayerName+"_up1", padding = 'valid')(UpSampling2D(size = (2,2))(bn2))
    act31 = LeakyReLU()(up1)
    cotr3 = Conv2DTranspose(kn[2], 3, strides=(1,1), padding='valid', use_bias = True)(act31)
    act32 = LeakyReLU()(cotr3)

    merge3 = concatenate([act32,act22], axis = 3)
    bn3= BatchNormalization(epsilon = 1.1e-5)(merge3)

    up2 = Conv2D(kn[3], 3, name = LayerName+"_up2", padding = 'valid')(UpSampling2D(size = (2,2))(bn3))
    act41 = LeakyReLU()(up2)
    cotr4 = Conv2DTranspose(kn[3], 3, strides=(1,1), padding='valid', use_bias = True)(act41)
    act42 = LeakyReLU()(cotr4)

    merge4 = concatenate([act42,act12,bn0], axis = 3)
    bn4 = BatchNormalization(epsilon = 1.1e-5)(merge4)

    conv5 = Conv2D(kn[4], 3, name = LayerName+"_conv5",padding = 'valid')(bn4)
    act51 = LeakyReLU()(conv5)
    cotr5 = Conv2DTranspose(3, 3, strides=(1,1), padding='valid', use_bias = True)(act51)
    act52 = LeakyReLU()(cotr5)
    model = Model(inputs = inputs, outputs = act52)
    #act52 = kr.activations.tanh(cotr5)
    #plot_model(model, to_file='haze_model.png', show_shapes=True)

    return model

#if __name__ == "__main__":
#    haze_net = SmallUnet((256,256,3))
def HazeUnet_v2(input_size =  (Height,Width,3)):
    LayerName = 'HazeUnet_'
    kn = [64,64,64,64,256]
    inputs = Input(input_size)
    conv1 = Conv2D(kn[0], 3, name =LayerName+ "conv1", padding = 'valid')(inputs)
    act11 = LeakyReLU()(conv1)
    cotr1 = Conv2DTranspose(kn[0], 3, strides=(1,1), padding='valid', use_bias = True)(act11)
    act12 = LeakyReLU()(cotr1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(act12)
    bn1 = BatchNormalization(epsilon = 1.1e-5)(pool1)

    conv2 = Conv2D(kn[1], 3, name =LayerName+ "conv2", padding = 'valid')(bn1)
    act21 = LeakyReLU()(conv2)
    cotr2 = Conv2DTranspose(kn[1], 3, strides=(1,1), padding='valid', use_bias = True)(act21)
    act22 = LeakyReLU()(cotr2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(act22)
    bn2= BatchNormalization(epsilon = 1.1e-5)(pool2)

    Med0 = UpSampling2D(size = (2,2))(act22)
    merge0 = concatenate([Med0,act12], axis = 3)
    conv0 = Conv2D(kn[0], 3, name =LayerName+ "conv0", padding = 'valid')(merge0)
    act01 = LeakyReLU()(conv0)
    cotr0 = Conv2DTranspose(kn[0], 3, strides=(1,1), padding='valid', use_bias = True)(act01)
    act02 = LeakyReLU()(cotr0)
    bn0= BatchNormalization(epsilon = 1.1e-5)(act02)    


    up1 = Conv2D(kn[2], 3, name = LayerName+"_up1", padding = 'valid')(UpSampling2D(size = (2,2))(bn2))
    act31 = LeakyReLU()(up1)
    cotr3 = Conv2DTranspose(kn[2], 3, strides=(1,1), padding='valid', use_bias = True)(act31)
    act32 = LeakyReLU()(cotr3)

    merge3 = concatenate([act32,act22], axis = 3)
    bn3= BatchNormalization(epsilon = 1.1e-5)(merge3)

    up2 = Conv2D(kn[3], 3, name = LayerName+"_up2", padding = 'valid')(UpSampling2D(size = (2,2))(bn3))
    act41 = LeakyReLU()(up2)
    cotr4 = Conv2DTranspose(kn[3], 3, strides=(1,1), padding='valid', use_bias = True)(act41)
    act42 = LeakyReLU()(cotr4)

    merge4 = concatenate([act42,act12,bn0], axis = 3)
    bn4 = BatchNormalization(epsilon = 1.1e-5)(merge4)

    conv5 = Conv2D(kn[4], 3, name = LayerName+"_conv5",padding = 'valid')(bn4)
    act51 = LeakyReLU()(conv5)
    cotr5 = Conv2DTranspose(3, 3, strides=(1,1), padding='valid', use_bias = True)(act51)
    act52 = LeakyReLU()(cotr5)
    model = Model(inputs = inputs, outputs = act52)
    #act52 = kr.activations.tanh(cotr5)
    #plot_model(model, to_file='haze_model.png', show_shapes=True)

    return model

def Aod_layer(inputs):
    kn = [32,32,32,32,32]
    conv1 = LeakyReLU()(Conv2D(kn[0], 1, padding = 'same')(inputs))
    drop1= Dropout(0.3)(conv1)

    conv2 = LeakyReLU()(Conv2D(kn[0], 3, padding = 'same')(drop1))
    drop2= Dropout(0.3)(conv2)

    merge1 = concatenate([drop1,drop2],axis=3)
    concat1 = LeakyReLU()(Conv2D(kn[0], 3, padding = 'same')(merge1))
    conv3 = LeakyReLU()(Conv2D(kn[0], 5, padding = 'same')(concat1))
    drop3= Dropout(0.3)(conv3)

    merge2 = concatenate([drop2,drop3],axis=3)
    concat2 = LeakyReLU()(Conv2D(kn[0], 3, padding = 'same')(merge2))
    conv4 = LeakyReLU()(Conv2D(kn[0], 7, padding = 'same')(concat2))
    drop4= Dropout(0.3)(conv4)

    merge3 = concatenate([drop1,drop3,drop4],axis=3)
    concat3 = LeakyReLU()(Conv2D(kn[0], 3, padding = 'same')(merge3))
    conv5 = LeakyReLU()(Conv2D(kn[0], 3, padding = 'same')(concat3))
    return conv5


def HazeUnet_3(input_size =  (Height,Width,3)):
    LayerName = 'HazeUnet_v3'
    kn = [256,32,32,32,64]
    inputs = Input(input_size)
    conv0 = LeakyReLU()(Conv2D(kn[0], 7, padding = 'same')(inputs))
    bn0 = BatchNormalization(epsilon = 1.1e-5)(conv0)

    aod_layer = Aod_layer(bn0)

    merge2 = concatenate([conv0,aod_layer],axis=3)
    drop2= Dropout(0.3)(merge2)

    conv5 = LeakyReLU()(Conv2D(kn[4], 3,padding = 'valid')(drop2))
    cotr5 = LeakyReLU()(Conv2DTranspose(3, 3, strides=(1,1), padding='valid', use_bias = True)(conv5))

    model = Model(inputs = inputs, outputs = cotr5)
    #act52 = kr.activations.tanh(cotr5)
    #plot_model(model, to_file='haze_model.png', show_shapes=True)

    return model



def joint_layer(inputs):


    conv1 = Conv3D(24, 3,dilation_rate =(1,3,3), padding = 'same')(inputs)
    act1 = LeakyReLU()(conv1)
    drop1 = Dropout(0.2)(act1)

    conv2 = Conv3D(24, 3, dilation_rate =(1,2,2), padding = 'same')(inputs)
    act2 = LeakyReLU()(conv2)
    drop2 = Dropout(0.2)(act2)

    conv3 = Conv3D(24, 3, dilation_rate =(1,1,1), padding = 'same')(inputs)
    act3 = LeakyReLU()(conv3)
    drop3 = Dropout(0.2)(act3)

    merge0 = concatenate([drop1,drop3], axis = 4)
    bn0 = BatchNormalization(epsilon = 1e-4)(merge0)
    return bn0


def rain_generator_v2(input_size =  (3,Height,Width,3)):
    inputs = Input(input_size)

    joint1 = joint_layer(inputs)
    merge1 = concatenate([joint1,inputs], axis = 4)

    joint2 = joint_layer(merge1)
    merge2 = concatenate([joint2,inputs], axis = 4)

    joint3 = joint_layer(merge2)
    merge3 = concatenate([joint3,inputs], axis = 4)
    #joint4 = joint_layer(joint3)

    conv0 = Conv3D(64, 3 ,padding = 'same')(merge3)
    act0 = LeakyReLU()(conv0)

    pool0 = AveragePooling3D(pool_size=(3, 1, 1))(act0)
    Layer_out = Lambda(lambda x:x[:,0,:,:,:])(pool0)
    Cinputs=Lambda(lambda x:x[:,1,:,:,:])(inputs)

    merge = concatenate([Layer_out,Cinputs], axis = 3)

    conv4 = LeakyReLU()(Conv2D(64, 3, padding = 'same')(merge))
    conv5 = LeakyReLU()(Conv2D(32, 3, padding = 'same')(conv4))

    #conv4 = SmallUnet(inputs)
    #merge1 = concatenate([conv1,conv2,conv3,conv4], axis = 3)
    conv6 = Conv2DTranspose(16, 3, name = "conv6",padding = 'valid')(conv5)
    act6 = LeakyReLU()(conv6)
    Output = Conv2D(3, 3, activation = 'tanh', name = "Output",padding = 'valid')(act6)

    #print("Output.shape = ",Output.shape)
    model = Model(inputs = inputs, outputs = Output)
    #model.summary()
    #model.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
    #print (model.summary())
    return model

def conv_layer(inputs,kn,ks):
    conv0 = LeakyReLU()(Conv2D(kn,ks,padding = 'same')(inputs))
    bn0 = BatchNormalization(epsilon = 1.1e-5)(conv0)
    return conv_layer

def MulNet(inputs):
    LayerName = 'MulNet'
    kn = [24,32,32,32,32]
    conv1 = LeakyReLU()(Conv3D(kn[0], 3,dilation_rate =(1,1,1), padding = 'same')(inputs))
    drop1= Dropout(0.3)(conv1)

    conv2 = LeakyReLU()(Conv3D(kn[0], 5,dilation_rate =(1,2,2), padding = 'same')(drop1))
    drop2= Dropout(0.3)(conv2)

    conv3 = LeakyReLU()(Conv3D(kn[0], 3,dilation_rate =(1,2,2), padding = 'same')(drop2))
    drop3= Dropout(0.3)(conv3)

    merge3 = concatenate([drop1,drop2,drop3],axis=4)
    conv5 = LeakyReLU()(Conv3D(kn[0], 3,dilation_rate =(1,1,1), padding = 'same')(merge3))
    return conv5


def rain_generator_v3(input_size =  (3,Height,Width,3)):

    inputs = Input(input_size)
    Cinputs=Lambda(lambda x:x[:,1,:,:,:])(inputs)

    joint1 = joint_layer(inputs)
    merge1 = concatenate([joint1,inputs], axis = 4)

    conv1 = LeakyReLU()(Conv3D(80, 3,dilation_rate =(1,1,1), padding = 'same')(merge1))
    drop1= Dropout(0.3)(conv1)

    conv2 = LeakyReLU()(Conv3D(80, 3,dilation_rate =(1,2,2), padding = 'same')(conv1))
    drop2= Dropout(0.3)(conv2)

    conv3 = LeakyReLU()(Conv3D(80, 3 ,dilation_rate =(1,3,3), padding = 'same')(drop2))
    drop3= Dropout(0.3)(conv3)

    merge2 = concatenate([conv1,conv3], axis = 4)
    conv4 = LeakyReLU()(Conv3D(80, 3,dilation_rate =(1,3,3), padding = 'same')(merge2))
    drop4= Dropout(0.3)(conv4)

    conv5 = LeakyReLU()(Conv3D(80, 3,dilation_rate =(1,2,2), padding = 'same')(drop4))
    drop5= Dropout(0.3)(conv5)

    conv6 = LeakyReLU()(Conv3D(80, 3,dilation_rate =(1,1,1), padding = 'same')(drop5))
    drop6= Dropout(0.3)(conv6)
    merge3 = concatenate([conv4,conv6], axis = 4)

    #conv7 = LeakyReLU()(Conv3D(80, 1,dilation_rate =(1,1,1), padding = 'same')(drop6))
    #drop7= Dropout(0.3)(conv7)


    pool0 = MaxPooling3D(pool_size=(3, 1, 1))(merge3)
    Layer_out = Lambda(lambda x:x[:,0,:,:,:])(pool0)
    merge0 = concatenate([Layer_out,Cinputs], axis = 3)

    conv4 = LeakyReLU()(Conv2D(256, 3, padding = 'same')(merge0))
    Output = Conv2D(3, 3, activation = 'tanh', name = "Output",padding = 'same')(conv4)

    #print("Output.shape = ",Output.shape)
    model = Model(inputs = inputs, outputs = Output)
    #model.summary()
    #model.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
    #print (model.summary())
    return model


def rain_generator_v2(input_size =  (3,Height,Width,3)):
    inputs = Input(input_size)

    joint1 = joint_layer(inputs)
    merge1 = concatenate([joint1,inputs], axis = 4)

    joint2 = joint_layer(merge1)
    merge2 = concatenate([joint2,inputs], axis = 4)

    joint3 = joint_layer(merge2)
    merge3 = concatenate([joint3,inputs], axis = 4)
    #joint4 = joint_layer(joint3)

    conv0 = Conv3D(64, 3 ,padding = 'same')(merge3)
    act0 = LeakyReLU()(conv0)

    pool0 = AveragePooling3D(pool_size=(3, 1, 1))(act0)
    Layer_out = Lambda(lambda x:x[:,0,:,:,:])(pool0)
    Cinputs=Lambda(lambda x:x[:,1,:,:,:])(inputs)

    merge = concatenate([Layer_out,Cinputs], axis = 3)

    conv4 = LeakyReLU()(Conv2D(64, 3, padding = 'same')(merge))
    conv5 = LeakyReLU()(Conv2D(32, 3, padding = 'same')(conv4))

    #conv4 = SmallUnet(inputs)
    #merge1 = concatenate([conv1,conv2,conv3,conv4], axis = 3)
    conv6 = Conv2DTranspose(16, 3, name = "conv6",padding = 'valid')(conv5)
    act6 = LeakyReLU()(conv6)
    Output = Conv2D(3, 3, activation = 'tanh', name = "Output",padding = 'valid')(act6)

    #print("Output.shape = ",Output.shape)
    model = Model(inputs = inputs, outputs = Output)
    #model.summary()
    #model.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
    #print (model.summary())
    return model





def rain_generator(input_size =  (3,Height,Width,3)):
    inputs = Input(input_size)
    conv1 = Conv3D(32, (3,3,3),dilation_rate =(2,3,3) , name = "conv1", padding = 'same')(inputs)
    act1 = LeakyReLU()(conv1)
    drop1 = Dropout(0.3)(act1)
    conv2 = Conv3D(64, 3, dilation_rate =(1,3,3), name = "conv2", padding = 'same')(drop1)
    act2 = LeakyReLU()(conv2)
    drop2 = Dropout(0.3)(act2)

    conv11 = Conv3D(32, (3,3,3), dilation_rate =(2,2,2),name = "conv11", padding = 'same')(inputs)
    act11 = LeakyReLU()(conv11)
    drop11 = Dropout(0.3)(act11)
    conv22 = Conv3D(64, 3, dilation_rate = (1,2,2), name = "conv22", padding = 'same')(drop11)
    act22 = LeakyReLU()(conv22)
    drop22 = Dropout(0.3)(act22)

    conv111 = Conv3D(32, (3,3,3), dilation_rate = (1,1,1),name = "conv111", padding = 'same')(inputs)
    act111 = LeakyReLU()(conv111)
    drop111 = Dropout(0.3)(act111)
    conv222 = Conv3D(64, 3, dilation_rate = (1,1,1), name = "conv222", padding = 'same')(drop111)
    act222 = LeakyReLU()(conv222)
    drop222 = Dropout(0.3)(act222)


    merge0 = concatenate([drop2,drop22,drop222], axis = 4)
    bn0 = BatchNormalization(epsilon = 1e-4, name = "bn0")(merge0)
    conv01 = Conv3D(32, 3, name = "conv01",padding = 'same')(bn0)
    act01 = LeakyReLU()(conv01)
    conv02 = Conv3D(32, 3, name = "conv02",padding = 'same')(act01)
    act02 = LeakyReLU()(conv02)

    pool01 = AveragePooling3D(pool_size=(3, 1, 1))(act02)
    Layer_out = Lambda(lambda x:x[:,0,:,:,:])(pool01)
    Cinputs=Lambda(lambda x:x[:,1,:,:,:])(inputs)

    #Layer_2 = SmallUnet(Cinputs)

    merge = concatenate([Layer_out,Cinputs], axis = 3)

    conv4 = Conv2D(64, 3, name = "conv4", padding = 'same')(merge)
    act4 = LeakyReLU()(conv4)
    conv5 = Conv2D(64, 3, name = "conv5", padding = 'same')(act4)
    act5 = LeakyReLU()(conv5)

    #conv4 = SmallUnet(inputs)
    #merge1 = concatenate([conv1,conv2,conv3,conv4], axis = 3)
    conv6 = Conv2DTranspose(128, 3, name = "conv6",padding = 'valid')(act5)
    act6 = LeakyReLU()(conv6)
    Output = Conv2D(3, 3, activation = 'tanh', name = "Output",padding = 'valid')(act6)

    #print("Output.shape = ",Output.shape)
    model = Model(inputs = inputs, outputs = Output)
    #model.summary()
    #model.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
    #print (model.summary())
    return model



def rain_discriminator(input_size = (Height,Width,3)):
    n_classes = 1
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, padding = 'same')(inputs)
    act1 = LeakyReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.1)(pool1)

    conv2 = Conv2D(64, 3, padding = 'same')(drop1)
    act2 = LeakyReLU()(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(act2)
    drop1 = Dropout(0.1)(pool1)

    conv3 = Conv2D(128, 3, padding = 'same')(pool2)
    act3 = LeakyReLU()(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(act3)
    bn4 = BatchNormalization(epsilon = 1.1e-5)(pool3)

    conv4 = Conv2D(128, 3,padding = 'same')(bn4)
    act4 = LeakyReLU()(conv4)

    conv5 = Conv2D(n_classes, 3,padding = 'same')(act4)
    act5 = LeakyReLU()(conv1)

    conv6 = Conv2D(1, 3, padding = 'same')(act5)
    act6 = LeakyReLU()(conv6)

    fat = Flatten()(act6)
    de1 = Dense(128)(fat)
    act = LeakyReLU()(de1)
    de2 = Dense(n_classes)(act)
    #output = LeakyReLU()(de2)
    output = core.Activation('sigmoid')(de2)

    model = Model(inputs = inputs, outputs = output)
    discriminator_optimizer = RMSprop(lr=8e-4, clipvalue=1.0, decay=1e-8)
    model.compile(optimizer=discriminator_optimizer, loss=wasserstein_loss)
    model.trainable = False

    #model.compile(optimizer=discriminator_optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
    #print (model.summary())
    return model

