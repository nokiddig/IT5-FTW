# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:23:29 2022

@author: User
"""

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy
from PIL import Image

import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import tensorflow as tf
import numpy
#load mô hình đã train trước
from keras.models import load_model
model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')

# Model MobilenetV2------------------------------
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

# Tạo nhiễu và in ra nhiễu 
loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

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
type1 = Label(top, text = "    Mô hình MobilenetV2   ", font = ("Arial Bold", 20))
type1.pack(pady = 20)

# Hàm phân loại và hiển thị nhãn ảnh sạch
def classify(file_path):
    global label_packed
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image)

    image = tf.cast(image, tf.float32)
    
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    
    image_probs = model.predict(image)
    _, image_class, class_confidence = get_imagenet_label(image_probs)
    print(image_class)
    label.configure(foreground='#011638', text=image_class) 
    
# Hàm hiện nút classify gọi hàm phân loại nhận vào ảnh tải lên    
def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(x = 120, y = 400)

def classify_adver(img):
    _, label, confidence = get_imagenet_label(pretrained_model.predict(img))
#    image_probs = pretrained_model.predict(img)
#    pred = numpy.argmax(image_probs)
#    label = classes[pred + 1]
    adver_label.configure(text = label)    

# Hàm hiện nút classify cho ảnh nhiễu
def show_adver_classify_button(img):
    classify_adver_b=Button(top,text="Classify Image",command=lambda: classify_adver(img),padx=10,pady=5)
    classify_adver_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_adver_b.place(x = 500, y = 400)


    
# Hàm hiện ảnh adver và hiện nút classify adver_img
def generate(file_path):
    epsilon = float(eps.get())
    
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image)
    
    image = preprocess(image)
    
    image_probs = pretrained_model.predict(image)
    
    pred = numpy.argmax(image_probs)
    #image_class = classes[pred + 1]
 
    # Get the input label of the image.
    img_index = pred
    print(pred)
    label = tf.one_hot(img_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(image, label)
    
    adv_x = image + epsilon*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    
    show_adver_classify_button(adv_x)
    
    # Show ra ảnh
    adv_x = adv_x[0] * 0.5 + 0.5
    #adv_x.thumbnail((224,224))
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
