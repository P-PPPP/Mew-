# COURAGE COW
# It requires OpenCV installed for Python
import sys
import cv2
import os
import argparse
import numpy as np
from playsound import playsound
from threading import Thread

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + './openpose/bin/python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + './openpose/x64/Release;' +  dir_path + './openpose/bin;'
        import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
except Exception as e:
    print(e)
    sys.exit(-1)

def LoadModel(custom_params:dict={'net_resolution': '320x176'}):
    # Flags
    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./openpose/models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item
    params.update(custom_params)
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper

def Flow(datum,opWrapper):
    pngs = {
        "Hammer":cv2.imread("./assets/hammer.png",cv2.IMREAD_UNCHANGED),
        "Pan":cv2.imread("./assets/pan.png",cv2.IMREAD_UNCHANGED),
        "Cow":cv2.imread("./assets/cow.png",cv2.IMREAD_UNCHANGED),
    }
    videoCaputer = cv2.VideoCapture(0)
    cap=videoCaputer
    COW_TIMES = 0
    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        #cv2.imshow('frame',frame)
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        Processed_Frames , COW_TIMES = ProcessData(datum.poseKeypoints,frame,pngs,COW_TIMES)
        if COW_TIMES == 10:  Thread(target=playsound,args=("./assets/1.mp3",)).start()
        if COW_TIMES == 20:  Thread(target=playsound,args=("./assets/2.mp3",)).start()
        if COW_TIMES > 30: Thread(target=playsound,args=("./assets/3.mp3",)).start()
        cv2.putText(Processed_Frames, "Cow_Times: {}".format(COW_TIMES), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('MEW~~~~',Processed_Frames)
        frame_num +=1
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  

def ProcessData(Body_Poistion,frame,pngs,COW_TIMES):

    def Process_Head(frame,scale):
        head_position = Body_Poistion[0][0]
        head_scale = 5*scale
        _head = cv2.resize(pngs["Cow"],(int(pngs["Cow"].shape[1]*head_scale),int(pngs["Cow"].shape[0]*head_scale))) #缩放Cow.png 
        head_x,head_y = int(head_position[:2][0] - _head.shape[1]//2) ,int(head_position[:2][1] - _head.shape[0]//2) #计算位置
        head_x2 , head_y2 = int(head_x + _head.shape[1]),int(head_y + _head.shape[0])
        return merge_img(frame,_head,head_y, head_y2, head_x, head_x2) # 头部处理完成之后的图片
    
    def Process_Left_Hand(frame,scale):
        left_hand_position = Body_Poistion[0][7]
        hammer_scale = scale
        _hammer =  cv2.resize(pngs["Hammer"],(int(pngs["Hammer"].shape[1]*hammer_scale),int(pngs["Hammer"].shape[0]*hammer_scale))) #缩放Hammer.png
        left_hand_x,left_hand_y = int(left_hand_position[:2][0] - _hammer.shape[1]//2) ,int(left_hand_position[:2][1] - _hammer.shape[0]) #计算位置
        left_hand_x2 , left_hand_y2 = int(left_hand_x + _hammer.shape[1]),int(left_hand_y + _hammer.shape[0])
        return merge_img(frame,_hammer,left_hand_y, left_hand_y2, left_hand_x, left_hand_x2),left_hand_x # 左手处理完成之后的图片

    def Process_Right_Hand(frame,scale):
        right_hand_position = Body_Poistion[0][4]
        Pan_Scale = scale
        _pan =  cv2.resize(pngs["Pan"],(int(pngs["Pan"].shape[1]*Pan_Scale),int(pngs["Pan"].shape[0]*Pan_Scale))) #缩放Hammer.png
        right_hand_x,right_hand_y = int(right_hand_position[:2][0] - _pan.shape[1]//2) ,int(right_hand_position[:2][1] - _pan.shape[0]) #计算位置
        right_hand_x2 , right_hand_y2 = int(right_hand_x + _pan.shape[1]),int(right_hand_y + _pan.shape[0])
        return merge_img(frame,_pan,right_hand_y, right_hand_y2, right_hand_x, right_hand_x2),right_hand_x # 左手处理完成之后的图片

    try:
        scale = (abs(Body_Poistion[0][2][0] - Body_Poistion[0][5][0]))/frame.shape[1] # 检测第一个是X 第二个是Y 但是cv2 shape 第一个是Y 第二个是X
        if str(Body_Poistion[0][0]): frame = Process_Head(frame,scale)
        if str(Body_Poistion[0][7]): 
            frame,hammer_x = Process_Left_Hand(frame,scale)
        if str(Body_Poistion[0][4]): 
            frame,pan_x = Process_Right_Hand(frame,scale)
        try:
            hands_dis = abs(hammer_x - pan_x) / frame.shape[1]
            if  hands_dis< 0.4 and hands_dis > 0.01: COW_TIMES+=1
            #else: COW_TIMES = 0
        except:...
        return frame,COW_TIMES
    except Exception as e: print("Cannnot Detect Nose Nor Hands {}".format(e))

def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """
    b_channel, g_channel, r_channel = cv2.split(img) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道

    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    return img_new
 
def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    """ 将png透明图像与jpg图像叠加 
        y1,y2,x1,x2为叠加位置坐标值
    """
    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
    
    '''
    当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
    这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
    '''
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]
 
    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]
 
    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[yy1:yy2,xx1:xx2,3] / 255.0
    alpha_jpg = 1 - alpha_png
    # 开始叠加
    for c in range(0,3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg*jpg_img[y1:y2,x1:x2,c]) + (alpha_png*png_img[yy1:yy2,xx1:xx2,c]))
    return jpg_img

if __name__ == '__main__':
    datum = op.Datum()
    opWrapper = LoadModel()
    Flow(datum,opWrapper)