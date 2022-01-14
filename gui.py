import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
#load mô hình đã train trước
from keras.models import load_model
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

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict_classes(image)[0]
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#011638', text=sign) 

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Đọc biển báo giao thông",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()