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
from Model import *


def net_predict(input_shape,rain_image,groundTrues,MFD_net):
    save_dir = './predict'
    print("rain_image.shape = ",rain_image.shape)
    batch,t,h,w,c = rain_image.shape
    input_shape = [t,h,w,c]
    #generator = MFD_net(input_shape)
    print("MFD_net creation.")
    count = 0
    psnr_list = []
    ssim_list = []

    for i in range(batch):    
        
        input_image = rain_image[i,:,:,:]
        input_image = np.expand_dims(input_image,axis=0)
        #img = np.concatenate([input_image[0,0,:,:,:],input_image[0,1,:,:,:],input_image[0,2,:,:,:]],axis=1)
        #img = np.uint8(img*255)
        #print("img.shape = ",img.shape)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        recover_images = MFD_net.predict(input_image)
        rain = input_image[:,:,:,:,:]
        #save_predict_image(recover_images,rain,i,'./predict')

        psnr,ssim = save_predict_image_v2(recover_images,groundTrues,rain,i,'./predict')

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        count+=1
        if count >=5:
            break
    
    print('mean psnr = ',np.mean(np.array(psnr_list)))
    print('mean ssim = ',np.mean(np.array(ssim_list)))

def save_predict_image(generated_images,input_image,num,path):
    # 保存目录
    save_dir = path
    generated_image_255_RGB = np.array([])

    # 保存生成的图像
    #255 - 1\2
    generated_image = np.uint8(255 -generated_images[0]*255.)
    generated_image_255 = generated_images[0]

    #generated_image_255 = generated_images
    #print("generated_images.shape = ",generated_images.shape)
    #print("generated_image_255.shape = ",generated_image_255.shape)
    '''
    generated_image_255[generated_image_255>1.0]=1.0
    generated_image_255[generated_image_255<0.0]=0.0
    '''
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

    rain =  np.uint8(input_image[0,1,:,:,:]*255.)
    #rain =  np.uint8(input_image[0,:,:,:]*255.)

    generated_image_255_RGB = cv2.cvtColor(generated_image_255, cv2.COLOR_BGR2RGB)
    #generated_image_255_RGB = generated_image_255

    rain  = cv2.cvtColor(rain, cv2.COLOR_BGR2RGB)

    #img = np.concatenate([rain,generated_image_255_RGB,generated_image_255,real_background],axis = 1)
    img = np.concatenate([rain,generated_image_255_RGB],axis = 1)
    img = image.array_to_img(img, scale=False)

    img.save(os.path.join(save_dir, str(num)+'.png'))



def save_predict_image_v2(generated_images,real_image,input_image,num,path):
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
        real_background = np.uint8(real_image[num]*255.)
        rain =  np.uint8(input_image[0,1,:,:,:]*255.)
        #rain =  np.uint8(input_image[0,:,:,:]*255.)

        generated_image_255_RGB = cv2.cvtColor(generated_image_255, cv2.COLOR_BGR2RGB)
        #generated_image_255_RGB = generated_image_255
        real_background  = cv2.cvtColor(real_background, cv2.COLOR_BGR2RGB)
        rain  = cv2.cvtColor(rain, cv2.COLOR_BGR2RGB)

        #img = np.concatenate([rain,generated_image_255_RGB,generated_image_255,real_background],axis = 1)
        img = np.concatenate([rain,generated_image_255_RGB,real_background],axis = 1)
        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.imshow('img_rgb',img_rgb)
        #cv2.waitKey(0)


        img = image.array_to_img(img, scale=False)

        result = ImageEvaluate(generated_image_255_RGB, real_background)  #图片质量评定类的构建  
        #调用 ssim与psnr 评价函数
        psnr = result.FunctionPsnr()
        ssim = result.FunctionSsim()


        img.save(os.path.join(save_dir, str(num)+'_' + str(psnr) +'_' + str(ssim) + '.png'))

       
        print("psnr = ",psnr)
        print("ssim = ",ssim)
        
    else:
    #except:
        print("rain.shape = ",rain.shape)
        print("generated_image_255_RGB.shape = ",generated_image_255_RGB.shape)
        print("real_background.shape = ",real_background.shape)
        print('except')
    return psnr,ssim

#图片质量评价类
class ImageEvaluate:

    '''''''''''''''''''''''''''''''''''''''''
                    Init
    '''''''''''''''''''''''''''''''''''''''''
    def __init__(self,ref,target): #ImageEvaluate类的初始化函数
        self.ref = np.array(ref)
        self.target = np.array(target)

    def cutImage(self):
        self.ref = self.ref[3:-3,3:-3,:]
        self.target = self.target[3:-3,3:-3,:]
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
