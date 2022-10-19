import numpy as np
import os
#import tensorflow as tf
import cv2
from Train import *
from dataset import *
from create import *
from predict import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配

def train_rain(input_shape,path,dir_name,loss_weigth_num):
    Rain_images,groundTrues,Haze_images = read_video(path+'/'+dir_name)
    Rain_images = Rain_images[30:]
    groundTrues = groundTrues[30:]
    Haze_images = Haze_images[30:]
    # 创建去雨模型
    #gan,generator,discriminator = create_rain_model(input_shape,False,loss_weigth_num)
    generator = create_rain_model_v2(input_shape,True,loss_weigth_num)
    #gan,generator,discriminator = create_rain_mode
    # 进行训练 
    train_rain_model_v2(Rain_images,Haze_images,generator,iterations = 500,batch_size = 2,loss_number = loss_weigth_num)
    #train_rain_model(Rain_images,Haze_images,gan,generator,discriminator,iterations = 3000,batch_size = 2,loss_number = loss_weigth_num+3)
    print("flag_num = ",loss_weigth_num)

def train_haze(input_shape,path,dir_name,loss_weigth_num):
    Rain_images,groundTrues,Haze_images = read_video(path+'/'+dir_name)
    Rain_images = Rain_images[30:]
    groundTrues = groundTrues[30:]
    Haze_images = Haze_images[30:]
    # 创建去雾模型
    MFD_net,generator = create_haze_model(input_shape,import_weigth_flag=True)
    # 进行训练
    train_haze_model(Rain_images, groundTrues, MFD_net,generator,iterations =500,batch_size = 2,loss_number = loss_weigth_num)

def predict_img(input_shape,path,dir_name,loss_weigth_num):
    # 进行预测
    Rain_images,groundTrues,_ = read_video(path+'/'+dir_name)
    #Rain_images,groundTrues,_ = read_video('./test')
    print("Rain_images.shape = ",np.array(Rain_images).shape)
    print("groundTrues.shape = ",np.array(groundTrues).shape)
    Rain_images = Rain_images[30:]
    groundTrues = groundTrues[30:]
    MFD_net,_ = create_haze_model(input_shape,import_weigth_flag=True)
    net_predict(input_shape[1:],Rain_images,groundTrues,MFD_net)
    #return 0


def main():
    input_shape = (3,256,256,3)
    path = './dataset'
    # 选择去雨/去雾/测试迭代轮数
    iter_num = 0
    for loss_weigth_num in range(iter_num):
        dir_list = os.listdir(path)
        for dir_name in dir_list:
            # 获取图片
            print("dir_name = ",dir_name)
            #train_rain(input_shape,path,dir_name,loss_weigth_num)
            #train_haze(input_shape,path,dir_name,loss_weigth_num)
            #predict_img(input_shape,path,dir_name,loss_weigth_num)
            #return 0
    # 选择去雨/去雾/测试迭代轮数
    iter_num = 1
    for loss_weigth_num in range(iter_num):
        dir_list = os.listdir(path)
        for dir_name in dir_list:
            # 获取图片
            print("dir_name = ",dir_name)
            #train_rain(input_shape,path,dir_name,loss_weigth_num)
            #train_haze(input_shape,path,dir_name,loss_weigth_num)
            predict_img(input_shape,path,dir_name,loss_weigth_num)
            #return 0

    print("running end")


if __name__ == '__main__':
    main()
cv2.destroyAllWindows()

