# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:28:05 2022

@author: User
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy
from PIL import Image

from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

from keras.models import load_model

# Model của mình-----------------------------------
model = load_model('my_model.h5')

#dictionary to label all traffic signs class.
classes = { 1:'Tốc độ giới hạn (20km/h)',
            2:'Tốc độ giới hạn (30km/h)', 
            3:'Tốc độ giới hạn (50km/h)', 
            4:'Tốc độ giới hạn (60km/h)', 
            5:'Tốc độ giới hạn (70km/h)', 
            6:'Tốc độ giới hạn (80km/h)', 
            7:'Kết thúc tốc độ giới hạn (80km/h)', 
            8:'Tốc độ giới hạn (100km/h)', 
            9:'Tốc độ giới hạn (120km/h)', 
            10:'Cấm vượt', 
            11:'Xe 3,5 tấn trở lên không được vượt', 
            12:'Đường ưu tiên', 
            13:'Đường ưu tiên', 
            14:'Nhường đường', 
            15:'Dừng', 
            16:'Đường cấm', 
            17:'Cấm xe trên 3.5 tấn', 
            18:'Cấm đi ngược chiều', 
            19:'Nguy hiểm', 
            20:'Đường cong nguy hiểm bên trái', 
            21:'Đường cong nguy hiểm bên phải', 
            22:'2 đường cong nối tiếp ở trước', 
            23:'Đường không bằng phẳng', 
            24:'Đường trơn', 
            25:'Đường hẹp bên phải', 
            26:'Đường đang thi công', 
            27:'Đèn giao thông', 
            28:'Đường cho người đi bộ', 
            29:'Trẻ em qua đường', 
            30:'Xe đạp qua đường', 
            31:'Cẩn thận băng/tuyết',
            32:'Động vật hoang dã qua đường', 
            33:'Hết giới hạn tốc độ', 
            34:'Rẽ phải ở phía trước', 
            35:'Rẽ trái ở phía trước', 
            36:'Chỉ được đi thẳng', 
            37:'Đi thẳng hoặc rẽ phải', 
            38:'Đi thẳng hoặc rẽ trái', 
            39:'Tiếp tục rẽ phải', 
            40:'Tiếp tục rẽ trái', 
            41:'Găp vòng xuyến', 
            42:'Kết thúc cấm vượt', 
            43:'Kết thúc cấm vượt cho xe trên 3,5 tấn' }

pretrained_model = model
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = classes

def preprocess(image):
  image = Image.open(image)
  image = image.resize((32,32))
  image = numpy.expand_dims(image, axis=0)
  #image = numpy.array(image)
  image = numpy.array(image).astype(numpy.float32)
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = tf.convert_to_tensor(image)
  return image
  
# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)
    #print(loss)
  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad


# Tạo giao diện và thao tác với giao diện

# Tạo một cửa sổ mới
top = Tk()

# Thêm tiêu đề cho cửa sổ
top.title('Menu')

# Đặt kích thước cửa sổ
top.geometry('800x600')

# Thêm cái heading
type1 = Label(top, text = "Mô hình nhận diện biển báo giao thông", font = ("Arial Bold", 20))
type1.pack(pady = 20)

# Hàm phân loại và hiển thị nhãn ảnh sạch
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    pred = model.predict(image)
    pred = np.argmax(pred)
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
    
# Hàm hiện nút classify gọi hàm phân loại nhận vào ảnh tải lên    
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(x = 120, y = 400)
    
# Hàm hiện nút classify cho ảnh nhiễu
def show_adver_classify_button(img):
    classify_adver_b=Button(top,text="Classify Image",command=lambda: classify_adver(img),padx=10,pady=5)
    classify_adver_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_adver_b.place(x = 500, y = 400)

def classify_adver(img):
    image_probs = pretrained_model.predict(img)
    pred = numpy.argmax(image_probs)
    label = classes[pred + 1]
    adver_label.configure(text = label)


# Hàm hiện ảnh adver và hiện nút classify adver_img
def generate(file_path):
    epsilon = float(eps.get())
    image = preprocess(file_path)
    image_probs = pretrained_model.predict(image)
    
    pred = numpy.argmax(image_probs)
    image_class = classes[pred + 1]
 
    # Get the input label of the image.
    img_index = pred
    label = tf.one_hot(img_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(image, label)
    
    adv_x = image + epsilon*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    
    show_adver_classify_button(adv_x)

    # Show ra ảnh
    adv_x = adv_x[0] * 0.5 + 0.5
    #adv_x.thumbnail((224,224))
    adv_x = tf.image.resize(adv_x, (224, 224))
    adv_x = adv_x.numpy()
    im = ImageTk.PhotoImage(image = Image.fromarray((adv_x*255).astype(np.uint8)))
    
    #im=ImageTk.PhotoImage(adv_x)
    adver_image.configure(image=im)
    adver_image.image=im
    
    print(epsilon)

def show_adver_button(file_path):
    adver_button = Button(top, text = "Generate adverImage", padx = 5, pady = 5, command=lambda: generate(file_path))
    adver_button.place(x = 600, y = 80)
    adver_button.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    
# Hàm được gọi để up ảnh, gọi hàm sẽ hiện nút classify nhận vào link ảnh vừa up lên
# Link ảnh chỉ xuất hiện trong này
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded = uploaded.resize((32, 32))
        uploaded = uploaded.resize((224, 224))
        #uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        uploaded.thumbnail((224,224))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
        show_adver_button(file_path)
    except:
        pass
# Nút upload
upload=Button(top,text="Upload image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.place(x = 120, y = 80)    

# Hiện ảnh sạch và nhãn ảnh sạch 
label=Label(top, font=('arial',15,'bold'))
label.place(x=120,y=120)

sign_image = Label(top)
sign_image.place(x = 80, y= 150)

# Thêm textbox
epsLabel = Label(top, text = "Epsilon: ", font = ("Arial Bold", 12))
epsLabel.place(x = 400, y=80)

eps = Entry(top, width = 10)
eps.place(x = 480, y=80)

# Hiện ảnh sau khi tấn công và nhãn sau khi tấn công
adver_label=Label(top, font=('arial',15,'bold'))
adver_label.place(x=500,y=120)

adver_image = Label(top)
adver_image.place(x = 450, y= 150)


# Hiển thị cửa sổ
top.mainloop()
