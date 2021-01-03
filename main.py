from tkinter import *
from PIL import Image
from PIL import ImageTk
import utils
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tkinter import filedialog
import cv2
from Kmean_color import Kmeans
import matplotlib.pyplot as plt
def select_image():
	# grab a reference to the image panels
	global panelA, panelB
	path = filedialog.askopenfilename()
	n_clusters=int(e.get())
	if len(path) > 0:
			# load the image from disk, convert it to grayscale, and detect
		# edges in it
		image = cv2.imread(path)
		image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#dinh dang hinh anh thanh 1 danh sach pixel ma tran Mxn,voi 3 mau RGB
		image1 = image2.reshape((image2.shape[0] * image2.shape[1], 3))
		#chinh kich thuoc anh ve 640x380
		image=cv2.resize(image,(640,380))
		image3 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		km  = Kmeans(n_clusters)
		km.fit(image1)
		labels = km.labels
		centroids = km.centroids
		
		#init_centers=kmeans.kmeans_init_centers(image1,n_clusters)
		#init_labels=np.zeros(image1.shape[0])		
		#centroids,labels=kmeans.kmeans(init_centers,init_labels,image1,n_clusters)
		#sap xep danh sach cac pixel		
		hist = utils.centroid_histogram(labels)
		edged = utils.plot_colors(hist, centroids)		
		#tao hinh anh dung de ve tren plt
		image = Image.fromarray(image3)
		edged = Image.fromarray(edged)
		
		image = ImageTk.PhotoImage(image)
		edged = ImageTk.PhotoImage(edged)
	if panelA is None or panelB is None:
			
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)
			
			panelB = Label(image=edged)
			panelB.image = edged
			panelB.pack(side="right", padx=10, pady=10)
			
	else:
			
			panelA.configure(image=image)
			panelB.configure(image=edged)
			panelA.image = image
			panelB.image = edged
from SSE import figure_SSE
def button_SSE():
    figure_SSE()
	#path = filedialog.askopenfilename()
	#n_clusters=int(e.get())
	#if len(path) > 0:
			# load the image from disk, convert it to grayscale, and detect
		# edges in it
		#image = cv2.imread(path)
		#image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#dinh dang hinh anh thanh 1 danh sach pixel ma tran Mxn,voi 3 mau RGB
		#image1 = image2.reshape((image2.shape[0] * image2.shape[1], 3))
		#chinh kich thuoc anh ve 640x380
		#image=cv2.resize(image,(640,380))
		#sse = []
		#list_k = list(range(0,10))
		#for k in list_k:
		#	km= Kmeans(n_clusters=k)
		#	km.fit(image1)
		#	sse.append(km.compute_sse(image1,labels=km.labels,centroids=km.centroids))
		#plt.figure(figsize=(6,6))
		#plt.plot(list_k, sse, '-o')
		#plt.xlabel(r'Number of clusters *k*')
		#plt.xlabel('Sum of squared distance')
		#plt.show()
        #plt.ylabel('Sum of squared distance');

        #for k in list_k:
        #   km = Kmeans(n_clusters=k)
        #  km.fit(image1)
           #sse.append(km.compute_sse(X,labels=km.labels,centroids=km.centroids))

root=Tk()
panelA = None
panelB = None
#tao giao dien 720x640
root.minsize(720,640)
a=Label(root,text="cluster_centers")
a.pack()
e=Entry(root,width=50)
e.pack()
btn2=Button(root,text="figure of color")
btn1 =Button(root, text="SSE of image",command=button_SSE)
btn = Button(root, text="Select an image", command=select_image)
btn2.pack(side="top", fill="both", expand="no", padx="10", pady="10")
btn1.pack(side="top", fill="both", expand="no", padx="10", pady="10")
btn.pack(side="top", fill="both", expand="no", padx="10", pady="10")
root.mainloop();