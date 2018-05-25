#coding=utf-8

import argparse

#图片解码
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
#web服务器接口
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

import utils

#初始化server
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

#设置赛车最大、最小车速
MAX_SPEED = 25 
MIN_SPEED = 10
speed_limit = MAX_SPEED

#在服务器端创建一个活动处理器
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        #当前车辆转角
        steering_angle = float(data["steering_angle"])
        #当前车辆油门开度
        throttle = float(data["throttle"])
        #当前车辆速度
        speed = float(data["speed"])
        # 车辆中间摄像头的照片(这里只能获取中间相机的照片)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)       
            image = utils.preprocess(image) #预处理，ROI、Resize、颜色区间转换
            image = np.array([image])       # model需要4D array

            # 预测转角
            steering_angle = float(model.predict(image, batch_size=1))
            #使车速保持在合理范围内
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
	    #发送指令
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

     
    else:
        
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


#发送指令
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Autonomous Driving')
    parser.add_argument(
        'model',
        type=str
    )    
    args = parser.parse_args()

    #载入模型
    model = load_model(args.model)

    app = socketio.Middleware(sio, app)

    # 部署WSGI server，监听端口
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
