import numpy as np
import os
from PIL import Image
import cv2
import random

batch_width = 256
batch_Height = 256


def np_concat(data, inputs, axis = 0):
    if data == []:
        data = inputs
    else:
        data = np.concatenate([data,inputs],axis = 0)
    return data


def read_video(path,batch_image_size = 200):
    # 数据列表
    Rain_images = []
    groundTrues = []
    Haze_images = []
    # 获取视频目录列表
    #dir_list = os.listdir(path)
    dir_list = path
    Temp_Rain_images = []
    Temp_groundTrues = []
    Temp_Haze_images = []

    rand_three = random.randint(0,3)*3
    #for dir_name in dir_list:
    if 1:
        dir_name  = dir_list
        # 一般是只有三个文件名
        filename_list = os.listdir(dir_name)
        #filename_list = os.listdir(path+'/'+dir_name)
        #print("dir_name = ",dir_name)
        #print("filename = ",filename)
        for filename in filename_list:
            
            ret = True
            # 获取视频文件名
            #video_name = path+'/'+dir_name+'/'+filename
            video_name = dir_name+'/'+filename
            if filename == 'rain.avi':
                

                # 获取堆叠图像
                ret, rain_image = catch_video(video_name,rand_three, israin_flag = True,batch_image_size = batch_image_size)
                # 若读取失败
                if ret == True:
                    # 放入暂存变量
                    Temp_Rain_images = np_concat(Temp_Rain_images,rain_image)
                
            elif filename =='haze.avi':                
                # 获取堆叠图像
                ret,haze_image = catch_video(video_name,rand_three, israin_flag = False,batch_image_size = batch_image_size)
                if ret == True:
                    # 放入暂存变量
                    Temp_Haze_images = np_concat(Temp_Haze_images,haze_image)
            
            elif filename =='clear.avi':
                # 获取堆叠图像
                ret,background = catch_video(video_name,rand_three, israin_flag = False,batch_image_size = batch_image_size)
                if ret == True:
                    # 放入暂存变量
                    Temp_groundTrues = np_concat(Temp_groundTrues,background)
            #else:
            #    print('Not standard video name!')
            #    Temp_groundTrues = []
            #    Temp_Rain_images = []
            #    Temp_Haze_images = []
            #    break

            # 若读取失败
            if ret == False:
                print(filename," : reading error！")
                # 清除这次循环读取的变量
                Temp_groundTrues = []
                Temp_Rain_images = []
                Temp_Haze_images = []
                break
                    
        # 合并
        Rain_images = np_concat(Rain_images,Temp_Rain_images, axis = 0)
        groundTrues = np_concat(groundTrues,Temp_groundTrues, axis = 0)
        Haze_images = np_concat(Haze_images,Temp_Haze_images, axis = 0)

        #print("dir_name = ",float(dir_name))
        print("Rain_images.shape = ",Rain_images.shape)
        print("groundTrues.shape = ",np.array(groundTrues).shape)



    print("reading end")
    return Rain_images, groundTrues,Haze_images    


    
def catch_video(path,rand_three,israin_flag,each_frame = 6,batch_image_size = 80):
    # 返回值
    rand_frame_start = 1+rand_three
    ret = True               
    images = []
    image = []
    # 读取视频
    cap = cv2.VideoCapture(path)
    #print("path = ",path)
    # 判断是否读出
    if not cap.isOpened():
        print("open video error:",path)
        ret = False
        return ret,image

    # 随机开始位置
    for rand_index in range(rand_frame_start):
        ok,PreImage = cap.read()
    cv2.waitKey(1)

    ok,CorImage = cap.read()
    CorImage = cv2.resize(CorImage, (batch_width,batch_Height))
    cv2.waitKey(1)

    ok,BacImage = cap.read()
    BacImage = cv2.resize(BacImage, (batch_width,batch_Height))
    cv2.waitKey(1)

    # 读取记录
    i = 0
    # 持续时间内
    while cap.isOpened():
        #print("i = ",i)
        # 记录加一
        i += 1
        # 推下一帧
        PreImage = CorImage
        CorImage = BacImage
        # 读取一帧数据
        ok, BacImage = cap.read()
        cv2.waitKey(1)
        # 判断是否结束
        if not ok:
            print("break")
            break

        BacImage = cv2.resize(BacImage, (batch_width,batch_Height))
        # 每 3 帧记录一次
        if i%each_frame == 1:
            # 对于雨滴图像
            if israin_flag == True:
                # 在通道处叠加
                #print( "PreImage= ",PreImage.shape)
                #print( "CorImage= ",CorImage.shape)
                #print( "BacImage= ",BacImage.shape)
                PreImage = np.expand_dims(PreImage,axis = 0)
                CorImage = np.expand_dims(CorImage,axis = 0)
                BacImage = np.expand_dims(BacImage,axis = 0)
                image = np.concatenate([PreImage, CorImage,BacImage], axis = 0)


            # 对于非雨滴图像
            else:
                image = CorImage
            # 归一化
            image = image/255
            # 添加进列表
            images.append(image)
        
        #if i>=batch_image_size*3:
        #    break
    print("i=",i)
    images = np.array(images)
    #释放摄像头
    cap.release()
    return ret,images
