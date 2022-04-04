# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 20:33:17 2022

@author: User
"""
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import ttk
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

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import numpy
from PIL import Image

from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import argparse

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

model.trainable = False

def preprocess(image):
  image = Image.open(image)
  image = image.resize((32,32))
  image = numpy.expand_dims(image, axis=0)
  #image = numpy.array(image)
  image = numpy.array(image).astype(numpy.float32)
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = tf.convert_to_tensor(image)
  return image
  
sccLoss = SparseCategoricalCrossentropy()

# Creating an adversarial example

def clip_eps(tensor, eps):
	# clip the values of the tensor to a given range and return it
	return tf.clip_by_value(tensor, clip_value_min=-eps,
		clip_value_max=eps)

EPS = 20 / 255.0
LR = 0.1
    #EPS = float(eps.get())
    #LR = float(lr.get())
    #Step = int(step.get())
    
optimizer = Adam(learning_rate=LR)

# => def create_adversarial_pattern
def generate_adversaries(model, baseImage, delta, classIdx, steps=50):
	# iterate over the number of steps
	for step in range(0, steps):
		# record our gradients
		with tf.GradientTape() as tape:
			# explicitly indicate that our perturbation vector should 
            # be tracked for gradient updates
			tape.watch(delta) 

            # add our perturbation vector to the base image and
			# preprocess the resulting image
			# baseImage = baseImage*0.5 + 0.5
			adversary = preprocess_input(baseImage + delta)
			# run this newly constructed image tensor through our
			# model and calculate the loss with respect to the
			# *original* class index
			predictions = model(adversary, training=False)
			loss = -sccLoss(tf.convert_to_tensor([classIdx]), predictions)
			
            # check to see if we are logging the loss value, and if
			# so, display it to our terminal
			if step % 5 == 0:
				print("step: {}, loss: {}...".format(step, loss.numpy()))
		  
        # calculate the gradients of loss with respect to the
		# perturbation vector
		gradients = tape.gradient(loss, delta)
	
      	# update the weights, clip the perturbation vector, and
		# update its value
		optimizer.apply_gradients([(gradients, delta)])
		delta.assign_add(clip_eps(delta, eps=EPS))

	# return the perturbation vector
	return delta

EPS = 20 / 255.0
LR = 5e-3
def generate_target_adversaries(model, baseImage, delta, classIdx, advIdx, steps=200):
	# iterate over the number of steps
	for step in range(0, steps):
		# record our gradients
		with tf.GradientTape() as tape:
			# explicitly indicate that our perturbation vector should 
            # be tracked for gradient updates
			tape.watch(delta) 

            # add our perturbation vector to the base image and
			# preprocess the resulting image
			# baseImage = baseImage*0.5 + 0.5
			adversary = preprocess_input(baseImage + delta)
			# run this newly constructed image tensor through our
			# model and calculate the loss with respect to the
			# *original* class index
			predictions = model(adversary, training=False)
			loss = -sccLoss(tf.convert_to_tensor([classIdx]), predictions) + sccLoss(tf.convert_to_tensor([advIdx]), predictions)
            
            # check to see if we are logging the loss value, and if
			# so, display it to our terminal
			if step % 5 == 0:
				print("step: {}, loss: {}...".format(step, loss.numpy()))
		  
        # calculate the gradients of loss with respect to the
		# perturbation vector
		gradients = tape.gradient(loss, delta)
	
      	# update the weights, clip the perturbation vector, and
		# update its value
		optimizer.apply_gradients([(gradients, delta)])
		delta.assign_add(clip_eps(delta, eps=EPS))

	# return the perturbation vector
	return delta

# Tạo giao diện và thao tác với giao diện

# Tạo một cửa sổ mới
top = Tk()

# Thêm tiêu đề cho cửa sổ
top.title('Menu')

# Đặt kích thước cửa sổ
top.geometry('850x500')

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
    classify_b.place(x = 90, y = 400)
    
# Hàm hiện nút classify cho ảnh nhiễu
def show_adver_classify_button(img):
    classify_adver_b=Button(top,text="Classify Image",command=lambda: classify_adver(img),padx=10,pady=5)
    classify_adver_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_adver_b.place(x = 500, y = 400)

def classify_adver(img):
    image_probs = model.predict(img)
    pred = numpy.argmax(image_probs)
    label = classes[pred + 1]
    adver_label.configure(text = label)


# Hàm hiện ảnh adver và hiện nút classify adver_img

def generate(file_path):
    #epsilon = float(eps.get())
    # define the epsilon and learning rate constants
    
    # Untarget
    #EPS = 20 / 255.0
    #LR = 0.1
    
    # Target
    #EPS = 20 / 255.0
    #LR = 5e-3
    
    EPS = float(eps.get())
    LR = float(lr.get())
    Step = int(step.get())
    
    optimizer = Adam(learning_rate=LR)  
    
    baseImage = Image.open(file_path)
    baseImage = baseImage.resize((32,32))
    baseImage = numpy.expand_dims(baseImage, axis=0)
    baseImage = numpy.array(baseImage).astype(numpy.float32)

    baseImage = tf.constant(baseImage, dtype=tf.float32)
    delta = tf.Variable(tf.zeros_like(baseImage, dtype = 'float32'), trainable=True) 
    
    BaseImage2 = preprocess(file_path)
    pred = model(BaseImage2, training=False)
    pred = numpy.argmax(pred)
    
    print("[INFO] generating perturbation...")
    
    targetIdx = [-1, 5, 9, 38, 17]
    #target['values'] = ("untarget", "Speed limit (80km/h)", "No passing", "Keep right")
    if target.current() == 0: deltaUpdated = generate_adversaries(model, baseImage, delta, pred, steps = Step)
    else: deltaUpdated = generate_target_adversaries(model, baseImage, delta, pred, targetIdx[target.current()], steps = Step)
    
    # create the adversarial example, swap color channels, and save the
    # output image to disk
    print("[INFO] creating adversarial example...")
    #adverImage = (baseImage + deltaUpdated).numpy().squeeze()
    adverImage = (baseImage + deltaUpdated)
    #adverImage = np.clip(adverImage, 0, 255).astype('uint8')
    
    print("[INFO] running inference on the adversarial example...")
    preprocessedImage = preprocess_input(baseImage + deltaUpdated)
    
    show_adver_classify_button(preprocessedImage)
    
    
    
    # Show ra ảnh
    #adv_x.thumbnail((224,224))

    #adv_x.thumbnail((224,224))
    adverImage = tf.image.resize(adverImage, (224, 224))
    adverImage = adverImage.numpy().squeeze()
    adverImage = np.clip(adverImage, 0, 255).astype('uint8')
    im = ImageTk.PhotoImage(image = Image.fromarray(adverImage))
    
    #adverImage = adverImage.numpy()
    #adverImage = adverImage[0]*0.5 + 0.5
    #adverImage = adverImage.resize((224, 224))
    #im = ImageTk.PhotoImage(image = Image.fromarray(adverImage))
    #im = ImageTk.PhotoImage(image = adverImage)
    #im=ImageTk.PhotoImage(adv_x) 
    
    adver_image.configure(image=im)
    adver_image.image=im
    #print(epsilon)

def show_adver_button(file_path):
    adver_button = Button(top, text = "Generate adverImage", padx = 5, pady = 5, command=lambda: generate(file_path))
    adver_button.place(x = 680, y = 80)
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
upload.place(x = 90, y = 80)    

# Hiện ảnh sạch và nhãn ảnh sạch 
label=Label(top, font=('arial',15,'bold'))
label.place(x= 100,y=120)

sign_image = Label(top)
sign_image.place(x = 40, y= 150)

# Thêm textbox epsilon
epsLabel = Label(top, text = "Epsilon: ", font = ("Arial Bold", 12))
epsLabel.place(x = 300, y=70)

eps = Entry(top, width = 10)
eps.place(x = 302, y=90)

# Thêm textbox learning rate
lrLabel = Label(top, text = "Learing rate: ", font = ("Arial Bold", 12))
lrLabel.place(x = 380, y=70)

lr = Entry(top, width = 10)
lr.place(x = 395, y=90)

# Thêm textbox step
stepLabel = Label(top, text = "Step: ", font = ("Arial Bold", 12))
stepLabel.place(x = 490, y=70)

step = Entry(top, width = 8)
step.place(x = 495, y=90)

# Thêm textbox target
targetLabel = Label(top, text = "Target: ", font = ("Arial Bold", 12))
targetLabel.place(x = 560, y=70)

target = ttk.Combobox(top, width=15)
target['values'] = ("untarget", "Speed limit(80km/h)", "No passing", "Keep right", "No entry")
target.current(0)
target.place(x = 565, y=90)

# Hiện ảnh sau khi tấn công và nhãn sau khi tấn công
adver_label=Label(top, font=('arial',15,'bold'))
adver_label.place(x=500,y=120)

adver_image = Label(top)
adver_image.place(x = 450, y= 150)


# Hiển thị cửa sổ
top.mainloop()
