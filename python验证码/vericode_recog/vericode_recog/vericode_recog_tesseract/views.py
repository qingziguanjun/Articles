#-*- coding:utf-8 -*-
from django.http import HttpResponse
from django.http import request
import base64
from django.views.decorators.csrf import csrf_exempt 
from PIL import Image
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import multiprocessing
import cv2
import requests
import time


CHAR_SET_LEN = 128

class Type(object):
	def __init__(self):
		dic={}
		with open('/home/user/vericode_recog/vericode_recog/vericode_recog_tesseract/type.txt', 'r') as f:
			content=f.read()
		types=content.split(';')
		for str in types:
			temp=str.split('=')
			dic[temp[0]]=temp[1]
		self.dic = dic
typePair = Type()


def name2vec(name,maxCaptcha):
    vector = np.zeros(maxCaptcha*CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i * CHAR_SET_LEN + ord(c)
        vector[idx] = 1
    return vector



def vec2name(vec):
    name = []
    for i in vec:
        a = chr(i)
        name.append(a)
    return "".join(name)

def get_name_and_image(num, file):
    all_image = os.listdir(file)
    random_file = random.randint(0, num-1)
    base = os.path.basename(file + all_image[random_file])
    name = os.path.splitext(base)[0]
    image = Image.open(file + all_image[random_file])
    image = np.array(image)
    return name, image
	
def get_next_batch(test_image_folder,maxCaptcha,totalNum,height,width,batch_size=64):
    batch_x = np.zeros([batch_size, height*width])
    batch_y = np.zeros([batch_size, maxCaptcha*CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image(totalNum, test_image_folder)
        batch_x[i, :] = 1*(image.flatten())
        batch_y[i, :] = name2vec(name,maxCaptcha)
    return batch_x, batch_y
def ceil_divide(divisor,dividend):
	truncResult=int(divisor/dividend)
	if divisor%dividend > 0:
		truncResult += 1
	return truncResult

	

####################################################

def train_crack_captcha_cnn(test_image_folder,model_save_folder,height,width,maxCaptcha,totalNum,w_alpha=0.01, b_alpha=0.1):
	with tf.Graph().as_default(): 
		filterSize = 3
		if height*width > 10000:
			filterSize = 5
		X = tf.placeholder(tf.float32, [None, height*width])
		Y = tf.placeholder(tf.float32, [None, maxCaptcha*CHAR_SET_LEN])
		keep_prob = tf.placeholder(tf.float32)
		x = tf.reshape(X, shape=[-1, height, width, 1])
		# 3 conv layer
		w_c1 = tf.Variable(w_alpha * tf.random_normal([filterSize, filterSize, 1, 32]))
		b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
		conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
		conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv1 = tf.nn.dropout(conv1, keep_prob)

		w_c2 = tf.Variable(w_alpha * tf.random_normal([filterSize, filterSize, 32, 64]))
		b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
		conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
		conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv2 = tf.nn.dropout(conv2, keep_prob)

		w_c3 = tf.Variable(w_alpha * tf.random_normal([filterSize, filterSize, 64, 64]))
		b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
		conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
		conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv3 = tf.nn.dropout(conv3, keep_prob)

		# Fully connected layer
		truncWidth = ceil_divide(width, 8)
		truncHeight = ceil_divide(height, 8)
		w_d = tf.Variable(w_alpha * tf.random_normal([truncWidth * truncHeight * 64, 1024]))
		b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
		dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
		dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
		dense = tf.nn.dropout(dense, keep_prob)

		w_out = tf.Variable(w_alpha * tf.random_normal([1024, maxCaptcha * CHAR_SET_LEN]))
		b_out = tf.Variable(b_alpha * tf.random_normal([maxCaptcha * CHAR_SET_LEN]))
		output = tf.add(tf.matmul(dense, w_out), b_out)
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

		predict = tf.reshape(output, [-1, maxCaptcha, CHAR_SET_LEN])
		max_idx_p = tf.argmax(predict, 2)
		max_idx_l = tf.argmax(tf.reshape(Y, [-1, maxCaptcha, CHAR_SET_LEN]), 2)
		correct_pred = tf.equal(max_idx_p, max_idx_l)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			step = 0
			while True:
				batch_x, batch_y = get_next_batch(test_image_folder,maxCaptcha,totalNum,height,width)
				_, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.})
				print(step, loss_)
				if step % 100 == 0:
					batch_x_test, batch_y_test = get_next_batch(test_image_folder,maxCaptcha,totalNum,height,width)
					acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
					print(step, acc)
					if acc > 0.99:
						saver.save(sess, model_save_folder, global_step=step)
						break
				step += 1



def vericode_to_string(modelFolder,file,height,width,maxCaptcha,w_alpha=0.01, b_alpha=0.1):
	with tf.Graph().as_default():
		filterSize = 3
		if height*width > 10000:
			filterSize = 5
		
		X = tf.placeholder(tf.float32, [None, height*width])
		Y = tf.placeholder(tf.float32, [None, maxCaptcha*CHAR_SET_LEN])
		keep_prob = tf.placeholder(tf.float32)
		x = tf.reshape(X, shape=[-1, height, width, 1])
		# 3 conv layer
		w_c1 = tf.Variable(w_alpha * tf.random_normal([filterSize, filterSize, 1, 32]))
		b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
		conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
		conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv1 = tf.nn.dropout(conv1, keep_prob)

		w_c2 = tf.Variable(w_alpha * tf.random_normal([filterSize, filterSize, 32, 64]))
		b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
		conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
		conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv2 = tf.nn.dropout(conv2, keep_prob)

		w_c3 = tf.Variable(w_alpha * tf.random_normal([filterSize, filterSize, 64, 64]))
		b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
		conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
		conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		conv3 = tf.nn.dropout(conv3, keep_prob)

		# Fully connected layer
		truncWidth = ceil_divide(width, 8)
		truncHeight = ceil_divide(height, 8)
		w_d = tf.Variable(w_alpha * tf.random_normal([truncWidth * truncHeight * 64, 1024]))
		b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
		dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
		dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
		dense = tf.nn.dropout(dense, keep_prob)

		w_out = tf.Variable(w_alpha * tf.random_normal([1024, maxCaptcha * CHAR_SET_LEN]))
		b_out = tf.Variable(b_alpha * tf.random_normal([maxCaptcha * CHAR_SET_LEN]))
		output = tf.add(tf.matmul(dense, w_out), b_out)
	
		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess, tf.train.latest_checkpoint(modelFolder))
			image = Image.open(file)
			image = np.array(image)
			image = 1 * (image.flatten())
			predict = tf.argmax(tf.reshape(output, [-1, maxCaptcha, CHAR_SET_LEN]), 2)
			text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
			vec = text_list[0].tolist()
			predict_text = vec2name(vec)
			return predict_text
###############
def otsu_binary(srcFolder,saveFolder):
    for filename in os.listdir(srcFolder):
		image = cv2.imread(srcFolder+filename) 
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
		ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		cv2.imwrite(saveFolder+filename,th2)

def binary(srcFolder,saveFolder,threshold):
	for filename in os.listdir(srcFolder):
		image = cv2.imread(srcFolder+filename) 
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
		ret2,th2 = cv2.threshold(image,threshold,255,cv2.THRESH_BINARY)
		cv2.imwrite(saveFolder+filename,th2)

@csrf_exempt 
def binary_image(request):
	if request.method == 'GET':
		srcFolder=request.GET['srcFolder']
		saveFolder=request.GET['saveFolder']
		type=request.GET['type']
		if type == 'otsu':
			otsu_binary(srcFolder,saveFolder)
		if type == 'binary':
			threshold=int(request.GET['threshold'])
			binary(srcFolder,saveFolder,threshold)
		return HttpResponse("done")

############
def do_clear_contour(cuttingWidth,cuttingHeight,srcFolder,saveFolder):
  for filename in os.listdir(srcFolder):
		image = cv2.imread(srcFolder + filename)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
		height = np.size(image, 0)
		width = np.size(image, 1)
		for x in range(0,cuttingWidth):
			for y in range(height):
				image[y][x] = 255
		for x in range(width-cuttingWidth,width):
			for y in range(height):
				image[y][x] = 255
		
		
		for y in range(0,cuttingHeight):
			for x in range(width):
				image[y][x] = 255
		for y in range(height-cuttingHeight,height):
			for x in range(width):
				image[y][x] = 255
		cv2.imwrite(saveFolder+filename,image)
@csrf_exempt 
def clear_contour(request):
	if request.method == 'GET':
		srcFolder=request.GET['srcFolder']
		saveFolder=request.GET['saveFolder']
		cuttingWidth=int(request.GET['cuttingWidth'])
		cuttingHeight=int(request.GET['cuttingHeight'])
		do_clear_contour(cuttingWidth,cuttingHeight,srcFolder,saveFolder)			
		return HttpResponse("done")
		
@csrf_exempt 
def vericode_recog_view(request):
	if request.method == 'POST':
		text=""
		if request.body == "":
			text = 'No post data.'
		else:
			params = request.body.split(',')
			data=typePair.dic[params[0]].split(',')
			height = int(data[0])
			width = int(data[1])
			maxCaptcha = int(data[2])
			modelFolder = data[3]
			type=data[4]
			threshold=int(data[5])
			image_raw_data = params[1]
			image_data = base64.b64decode(image_raw_data)
			file_name = 'temp.jpg'
			file = open(file_name,'wb')
			file.write(image_data)
			file.flush()
			file.close()
			binary_file_name='binary.jpg'
			image = cv2.imread(file_name) 
			image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
			if type == 'otsu':
				ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)	
			if type == 'binary':
				ret2,th2 = cv2.threshold(image,threshold,255,cv2.THRESH_BINARY)
			cv2.imwrite(binary_file_name,th2)
			text = vericode_to_string(modelFolder,binary_file_name,height,width,maxCaptcha)
		return HttpResponse(text)
	

@csrf_exempt 
def train_data(request):
	if request.method == 'GET':
		width=int(request.GET['width'])
		height=int(request.GET['height'])
		trainFolder=request.GET['trainFolder']
		modelSaveFolder=request.GET['modelSaveFolder']
		charNum=request.GET['charNum']
		totalNumStr=request.GET['totalNum']
		maxCaptcha = int(charNum)
		totalNum = int(totalNumStr)
		train_crack_captcha_cnn(trainFolder,modelSaveFolder,height,width,maxCaptcha,totalNum)
		return HttpResponse('done')
	
def doCrawl(url,folder,start,end):
    for i in range(start, end):
        r = requests.get(url,verify=False)
        if r.status_code != 200:
            time.sleep(100)
            continue
        f = open(folder + str(i) + '.jpg', 'wb')
        f.write(r.content)
        f.close()		
@csrf_exempt 
def crawl_pic(request):
	if request.method == 'GET':
		picUrl=request.GET['picUrl']
		print(picUrl)
		saveFolder=request.GET['saveFolder']
		start=int(request.GET['start'])
		end=int(request.GET['end'])
		doCrawl(picUrl,saveFolder,start,end)
		return HttpResponse('done')
		
@csrf_exempt 
def verify_model(request):
	if request.method == 'GET':
		width=int(request.GET['width'])
		height=int(request.GET['height'])
		testFolder=request.GET['testFolder']
		modelSaveFolder=request.GET['modelFolder']
		charNum=request.GET['charNum']
		totalNumStr=request.GET['totalNum']
		type=request.GET['type']
		maxCaptcha = int(charNum)
		totalNum = int(totalNumStr)
		
		right=0
		n = 1
		while n < totalNum:
			all_image = os.listdir(testFolder)
			base = os.path.basename(testFolder+all_image[n-1])
			file_name = testFolder+base
			binary_file_name='binary.jpg'
			image = cv2.imread(file_name)
			image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
			if type == 'otsu':
				ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			if type == 'binary':
				ret2, th2 = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
			cv2.imwrite(binary_file_name, th2)
			predict_text = vericode_to_string(modelSaveFolder,binary_file_name,height,width,maxCaptcha)
			text=base.split(".")[0]
			compare=(text == predict_text)
			if compare :
				right+=1
			n=n+1  
		return HttpResponse(str(right) + "/" + totalNumStr)
	

