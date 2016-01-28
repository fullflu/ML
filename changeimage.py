#coding:utf-8
# let you have two directories, "pos" and "neg" 
   # ("pos" contains positive images, and "neg" contains negative images)

# you can do 5 tasks
# 1. you can make two directories "train" and "val" so that "train" contains training images and "val" contains validation images
# 2. you can make two text files "train.txt" and "val.txt" to implement caffe/chainer-imagenet training
  # each line of "train.txt" is '[path_to_training_image] {1 or 0}' , and of "val.txt" is '[path_to_valdation_image] {1 or 0}' 
  # for simplicity, this script is written for two-class classification, but you can modify for multiclass classification
# 3. you can change a size of each image , from any size to (256,256), keeping its aspect ratio
  # this is necessary to implement chainer
# 4. you can change color, from RGB to gray scale
# 5. you can make mean images of gray_scale training images "mean_gray.npy", which is necessary to implement chainer/examples/imagenet
  # the shape of "mean_gray.npy" is ((1,256,256))

import numpy as np
import six.moves.cPickle as pickle
from PIL import Image
import os

pos="pos"
neg="neg"
now=os.getcwd()
posd=os.path.join(now,pos)
negd=os.path.join(now,neg)
posl = os.listdir(posd)
negl = os.listdir(negd)

posperm = np.random.permutation(len(posl))
negperm = np.random.permutation(len(negl))

#training ratio
ratio = .8


sum_image = np.zeros((256,256))
#positive training images
f = open("train.txt","w")
os.mkdir("train")
for i in posperm[:int(len(posl)*ratio)]:
	path = os.path.join(posd,posl[i])
	d = Image.open(path)
	if d.size[0] > d.size[1]:
		cr = 256.0 / d.size[0]
		d = d.convert("L")
		d2 = d.resize((256,int(cr*d.size[1])))
		d3 = np.array(d2)
		start = (256 - d2.size[1])/2
		d4 = np.ones((256,256))*16
		for j in range(d2.size[1]):
			d4[start+j,:] = d3[j,:]
		sum_image += d4
		d = Image.fromarray(np.uint8(d4))
		name = os.path.join(now,"train","pos_"+posl[i])
		d.save(name)
		f.write(name)
		f.write(" 1")
		f.write("\n")
	else:
		cr = 256.0 / d.size[1]
		d = d.convert("L")
		d2 = d.resize((int(cr*d.size[1]),256))
		d3 = np.array(d2)
		start = (256 - d2.size[0])/2
		d4 = np.ones((256,256))*16
		for j in range(d2.size[0]):
			d4[:,start+j] = d3[:,j]
		sum_image += d4
		d = Image.fromarray(np.uint8(d4))
		name = os.path.join(now,"train","pos_"+posl[i])
		d.save(name)
		f.write(name)
		f.write(" 1")
		f.write("\n")

#negative training images
for i in negperm[:int(len(negl)*ratio)]:
	path = os.path.join(negd,negl[i])
	d = Image.open(path)
	if d.size[0] > d.size[1]:
		cr = 256.0 / d.size[0]
		d = d.convert("L")
		d2 = d.resize((256,int(cr*d.size[1])))
		d3 = np.array(d2)
		start = (256 - d2.size[1])/2
		d4 = np.ones((256,256))*16
		for j in range(d2.size[1]):
			d4[start+j,:] = d3[j,:]
		d = Image.fromarray(np.uint8(d4))
		name = os.path.join(now,"train","neg_"+negl[i])
		d.save(name)
		f.write(name)
		f.write(" 0")
		f.write("\n")
	else:
		cr = 256.0 / d.size[1]
		d = d.convert("L")
		d2 = d.resize((int(cr*d.size[1]),256))
		d3 = np.array(d2)
		start = (256 - d2.size[0])/2
		d4 = np.ones((256,256))*16
		for j in range(d2.size[0]):
			d4[:,start+j] = d3[:,j]
		d = Image.fromarray(np.uint8(d4))
		name = os.path.join(now,"train","neg_"+negl[i])
		d.save(name)
		f.write(name)
		f.write(" 0")
		f.write("\n")

f.close()

mean_image = sum_image / (posperm.shape[0]+negperm.shape[0])
mean_image = mean_image.reshape((1,256,256))
pickle.dump(mean_image, open("mean_gray.npy", 'wb'), -1)


#positive validation images
f = open("val.txt","w")
os.mkdir("val")
for i in posperm[int(len(posl)*ratio):]:
	path = os.path.join(posd,posl[i])
	d = Image.open(path)
	if d.size[0] > d.size[1]:
		cr = 256.0 / d.size[0]
		d = d.convert("L")
		d2 = d.resize((256,int(cr*d.size[1])))
		d3 = np.array(d2)
		start = (256 - d2.size[1])/2
		d4 = np.ones((256,256))*16
		for j in range(d2.size[1]):
			d4[start+j,:] = d3[j,:]
		d = Image.fromarray(np.uint8(d4))
		name = os.path.join(now,"val","pos_"+posl[i])
		d.save(name)
		f.write(name)
		f.write(" 1")
		f.write("\n")
	else:
		cr = 256.0 / d.size[1]
		d = d.convert("L")
		d2 = d.resize((int(cr*d.size[1]),256))
		d3 = np.array(d2)
		start = (256 - d2.size[0])/2
		d4 = np.ones((256,256))*16
		for j in range(d2.size[0]):
			d4[:,start+j] = d3[:,j]
		d = Image.fromarray(np.uint8(d4))
		name = os.path.join(now,"val","pos_"+posl[i])
		d.save(name)
		f.write(name)
		f.write(" 1")
		f.write("\n")

#negative validation image
for i in negperm[int(len(negl)*ratio):]:
	path = os.path.join(negd,negl[i])
	d = Image.open(path)
	if d.size[0] > d.size[1]:
		cr = 256.0 / d.size[0]
		d = d.convert("L")
		d2 = d.resize((256,int(cr*d.size[1])))
		d3 = np.array(d2)
		start = (256 - d2.size[1])/2
		d4 = np.ones((256,256))*16
		for j in range(d2.size[1]):
			d4[start+j,:] = d3[j,:]
		d = Image.fromarray(np.uint8(d4))
		name = os.path.join(now,"val","neg_"+negl[i])
		d.save(name)
		f.write(name)
		f.write(" 0")
		f.write("\n")
	else:
		cr = 256.0 / d.size[1]
		d = d.convert("L")
		d2 = d.resize((int(cr*d.size[1]),256))
		d3 = np.array(d2)
		start = (256 - d2.size[0])/2
		d4 = np.ones((256,256))*16
		for j in range(d2.size[0]):
			d4[:,start+j] = d3[:,j]
		d = Image.fromarray(np.uint8(d4))
		name = os.path.join(now,"val","neg_"+negl[i])
		d.save(name)
		f.write(name)
		f.write(" 0")
		f.write("\n")
f.close()






