import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.utils.vis_utils import plot_model
from sklearn import metrics
import cv2

from keras.preprocessing import image
from losses import *

from PIL import Image
from PIL import ImageEnhance


def train_haze_model(Rain_images,groundTrues,MFD_net,iterations,batch_size = 2,loss_number = 4):
    # 设置起始点
    start = 0
    # 开始训练迭代
    real_image = groundTrues[:,:,:,:]
    Rain_image = Rain_images[:,1,:,:,:]

    #a_loss_1 = 0
    for step in range(iterations):
        # 取出雨滴图像
        stop = start + batch_size
        # 降雨图像
        Rain_image = Rain_images[start: stop,:,:,:,:]
        # 将假图像与真实图像进行比较
        real_image = groundTrues[start: stop,:,:,:]

        # 训练
        loss = MFD_net.train_on_batch(Rain_image, real_image)

        start += batch_size
        if start > len(Rain_images) - batch_size:
            start = 0
        if step>10 and (step+1) % 49 == 0: #step>10 and
            generated_images = MFD_net.predict(Rain_image)
            # 保存网络权值
            MFD_net_weigth_name = 'MFD_net_'+str(loss_number)+'.h5'
            MFD_net.save_weights('MFD_net.h5')
            MFD_net.save_weights(MFD_net_weigth_name)
            # 输出metrics
            print('--------------------------------------------------------')
            print('haze loss at step %s: %s' % (step, loss))
            save_predict_image(generated_images,real_image,Rain_image,loss_number,step,path = '.\\haze_image')

def train_haze_model(Rain_images,groundTrues,MFD_net,generator,iterations,batch_size = 2,loss_number = 4):
    # 设置起始点
    start = 0
    # 开始训练迭代
    real_image = groundTrues[:,:,:,:]
    Rain_image = Rain_images[:,:,:,:,:]

    #a_loss_1 = 0
    for step in range(iterations):
        # 取出雨滴图像
        stop = start + batch_size
        # 降雨图像
        Rain_image = Rain_images[start: stop,:,:,:,:]
        # 将假图像与真实图像进行比较
        real_image = groundTrues[start: stop,:,:,:]

        # 训练
        loss = MFD_net.train_on_batch(Rain_image, real_image)

        start += batch_size
        if start > len(Rain_images) - batch_size:
            start = 0
        if step>10 and (step+1) % 49 == 0: #step>10 and
            generated_images = MFD_net.predict(Rain_image)
            remove_rain_image = generator.predict(Rain_image)

            # 保存网络权值
            MFD_net_weigth_name = 'MFD_net_'+str(loss_number)+'.h5'
            MFD_net.save_weights('MFD_net.h5')
            MFD_net.save_weights(MFD_net_weigth_name)
            # 输出metrics
            print('--------------------------------------------------------')
            print('haze loss at step %s: %s' % (step, loss))
            save_predict_image(generated_images,real_image,Rain_image,loss_number,step,remove_rain_image,path = '.\\haze_image')


def train_rain_model_v2(Rain_images,groundTrues,generator,iterations,batch_size = 2,loss_number = 4):
    # 设置起始点
    start = 0
    # 开始训练迭代
    print("generated_images.shape = ",Rain_images.shape)
    print("groundTrues.shape = ",groundTrues.shape)

    generated_images = Rain_images[:,1,:,:,:]#4d
    True_rain_streaks = Rain_images[:,1,:,:,:]#4d
    real_image = groundTrues[:,:,:,:]

    Rain_image = Rain_images[:,1,:,:,:]

    loss_list = []
    #a_loss_1 = 0
    for step in range(iterations):
        # 取出雨滴图像
        stop = start + batch_size
        # 降雨图像
        Rain_image = Rain_images[start: stop,:,:,:,:]
        # 将假图像与真实图像进行比较
        real_image = groundTrues[start: stop,:,:,:]
        input_image = Rain_image[:,:,:,:]

        # 训练
        loss = generator.train_on_batch(input_image, real_image)
        loss_list.append(loss)
        start += batch_size
        if start > len(Rain_images) - batch_size:
            start = 0
        if step>10 and (step+1) % 11 == 0: #step>10 and
            # 恢复图像
            generated_images = generator.predict(input_image) 
            # 保存网络权值
            generator_name = 'generator_'+str(loss_number)+'.h5'
            generator.save_weights('generator.h5')
            generator.save_weights(generator_name)
            # 输出metrics
            print('--------------------------------------------------------')
            loss_list = np.array(loss_list)
            print('adversarial loss at step   %s: %s' % (step, np.mean(loss_list)))
            loss_list = []
            #print('adversarial_1 loss at step %s: %s' % (step, a_loss_1))
            print('--------------------------------------------------------')
            save_predict_image(generated_images,real_image,input_image,loss_number,step,path = '.\\rain_image')



def train_rain_model(Rain_images,groundTrues,gan,generator,discriminator,iterations,batch_size = 2,loss_number = 4):
    # 设置起始点
    start = 0
    # 开始训练迭代
    print("generated_images.shape = ",Rain_images.shape)
    print("groundTrues.shape = ",groundTrues.shape)

    generated_images = Rain_images[:,1,:,:,:]#4d
    True_rain_streaks = Rain_images[:,1,:,:,:]#4d
    real_image = groundTrues[:,:,:,:]

    Rain_image = Rain_images[:,1,:,:,:]

    a_loss = 0
    d_loss = 0

    #a_loss_1 = 0
    for step in range(iterations):
        # 取出雨滴图像
        stop = start + batch_size
        # 降雨图像
        Rain_image = Rain_images[start: stop,:,:,:,:]
        # 将假图像与真实图像进行比较
        real_image = groundTrues[start: stop,:,:,:]
        input_image = Rain_image[:,:,:,:]


        if step>=10 and step %15==0:
            discriminator.trainable = False
            # 汇集标有“所有真实图像”的标签
            misleading_targets = np.zeros((batch_size, 1))
            # 训练生成器（generator）（通过gan模型，鉴别器（discrimitor）权值被冻结）
            a_loss = gan.train_on_batch(input_image, [real_image,misleading_targets])
            generated_images = generator.predict(input_image)
        else:
            # 训练判别器
            discriminator.trainable = True
            # 恢复图像
            generated_images = generator.predict(input_image)
            # 合并真假图片
            combined_images = np.concatenate([generated_images, real_image])
            # 组装区别真假图像的标签
            labels = np.concatenate([np.ones((batch_size, 1)),np.zeros((batch_size, 1))])
            # 重要的技巧，在标签上添加随机噪声
            labels += 0.05 * np.random.random(labels.shape)
            # 训练鉴别器（discrimitor）
            d_loss = discriminator.train_on_batch(combined_images, labels)

        start += batch_size
        if start > len(Rain_images) - batch_size:
            start = 0
        if step>10 and (step+1) % 11 == 0: #step>10 and
            # 保存网络权值
            gan_weigth_name = 'gan_'+str(loss_number)+'.h5'
            gan.save_weights('gan.h5')
            generator.save_weights('generator.h5')
            gan.save_weights(gan_weigth_name)
            # 输出metrics
            print('--------------------------------------------------------')
            print('discriminator loss at step %s: %s' % (step, d_loss))
            print('adversarial loss at step   %s: %s' % (step, a_loss))
            #print('adversarial_1 loss at step %s: %s' % (step, a_loss_1))
            print('--------------------------------------------------------')
            save_predict_image(generated_images,real_image,input_image,loss_number,step,path = '.\\rain_image')



def save_predict_image(generated_images,real_image,input_image,loss_number,step,remove_rain_image= None,path=None):
    # 保存目录
    save_dir = path
    generated_image_255_RGB = np.array([])
    real_background  = np.array([])
    #try:
    if 1:
        # 保存生成的图像
        #255 - 1\2
        generated_image = np.uint8(255 -generated_images[0]*255.)
        generated_image_255 = generated_images[0]

        #generated_image_255 = generated_images
        #print("generated_images.shape = ",generated_images.shape)
        #print("generated_image_255.shape = ",generated_image_255.shape)
        r = generated_image_255[:,:,0]
        g = generated_image_255[:,:,1]
        b = generated_image_255[:,:,2]

        r[r>=1.0]=1.0
        r[r<=0.0]=0.0
        g[g>=1.0]=1.0
        g[g<=0.0]=0.0
        b[b>=1.0]=1.0
        b[b<=0.0]=0.0

        generated_image_255[:,:,0] = r
        generated_image_255[:,:,1] = g
        generated_image_255[:,:,2] = b

        generated_image_255 = np.uint8(generated_image_255*255.)
        real_background = np.uint8(real_image[0]*255.)
        rain =  np.uint8(input_image[0,1,:,:,:]*255.)
        #rain =  np.uint8(input_image[0,:,:,:]*255.)

        generated_image_255_RGB = cv2.cvtColor(generated_image_255, cv2.COLOR_BGR2RGB)
        #generated_image_255_RGB = generated_image_255
        real_background  = cv2.cvtColor(real_background, cv2.COLOR_BGR2RGB)
        rain  = cv2.cvtColor(rain, cv2.COLOR_BGR2RGB)
        #if remove_rain_image.any() != None:
        if 0:
            print("remove_rain_image.shape = ",remove_rain_image.shape)
            remove_rain_image = cv2.cvtColor(np.uint8(remove_rain_image[0]*255), cv2.COLOR_BGR2RGB)
            img = np.concatenate([rain,remove_rain_image,generated_image_255_RGB,real_background],axis = 1)
        else:
        #img = np.concatenate([rain,generated_image_255_RGB,generated_image_255,real_background],axis = 1)
            img = np.concatenate([rain,generated_image_255_RGB,real_background],axis = 1)
        img = image.array_to_img(img, scale=False)

        img.save(os.path.join(save_dir, str(loss_number)+'_' + str(step) + '.png'))

        result = ImageEvaluate(generated_image_255_RGB, real_background)  #图片质量评定类的构建  
        #调用 ssim与psnr 评价函数
        psnr = result.FunctionPsnr()
        ssim = result.FunctionSsim()

        print("psnr = ",psnr)
        print("ssim = ",ssim)
          

    else:
    #except:
        print("rain.shape = ",rain.shape)
        print("generated_image_255_RGB.shape = ",generated_image_255_RGB.shape)
        print("real_background.shape = ",real_background.shape)
        print('except')



def change_image(generated_images):
    print("generated_images.shape = ",generated_images.shape)
    # 绘图
    generated_image_255 = generated_images
    # 防止溢出
    r = generated_image_255[:,:,0]
    g = generated_image_255[:,:,1]
    b = generated_image_255[:,:,2]
    r[r>=1.0]=1.0
    r[r<=0.0]=0.0
    g[g>=1.0]=1.0
    g[g<=0.0]=0.0
    b[b>=1.0]=1.0
    b[b<=0.0]=0.0
    generated_image_255[:,:,0] = r
    generated_image_255[:,:,1] = g
    generated_image_255[:,:,2] = b
    generated_image_255 = np.uint8(generated_image_255*255.)
    # 通道转换
    generated_image_255_RGB = cv2.cvtColor(generated_image_255, cv2.COLOR_BGR2RGB)
    return generated_image_255_RGB

#图片质量评价类
class ImageEvaluate:

    '''''''''''''''''''''''''''''''''''''''''
                    Init
    '''''''''''''''''''''''''''''''''''''''''
    def __init__(self,ref,target): #ImageEvaluate类的初始化函数
        self.ref = np.array(ref)
        self.target = np.array(target)

    '''''''''''''''''''''''''''''''''''''''''
                 Function----------PSNR
    '''''''''''''''''''''''''''''''''''''''''
    def FunctionPsnr(self):

        mse = np.mean((self.ref/1.0-self.target/1.0)**2)#算出mse值
        psnr = 10 * np.log10(255 * 255 / mse)           #算出psnr值

        return psnr                                     #返回psnr值

    '''''''''''''''''''''''''''''''''''''''''
                 Function----------SSIM
    '''''''''''''''''''''''''''''''''''''''''
    def FunctionSsim(self):

        AverageX = self.target.mean() #目标图像的均值
        AverageY = self.ref.mean() #原图像的均值
        VarianceX = np.sqrt(((self.target - AverageX) ** 2).mean()) #目标图像的标准差
        VarianceY = np.sqrt(((self.ref - AverageY) ** 2).mean()) #原图像的标准差
        CovarianceXY = ((self.target - AverageX) * (self.ref - AverageY)).mean() #协方差

        k1, k2, L = 0.01, 0.03, 255 #参数设置
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        C3 = C2 / 2

        l = (2 * AverageX * AverageY + C1) / (AverageX ** 2 + AverageY ** 2 + C1) #亮度对比函数
        c = (2 * VarianceX * VarianceY + C2) / (VarianceX ** 2 + VarianceY ** 2 + C2) #对比度对比函数
        s = (CovarianceXY + C3) / (VarianceX * VarianceY + C3) #结构对比函数
        ssim = l * c * s # ssim 评价函数

        return ssim                                     #返回ssim值
