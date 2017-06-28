import numpy as np 
import csv
import matplotlib.pyplot as plt
import time

"""
Load raw data from  Kaggle Dataset and convert to numpy array

@return	- numpy array[emotions, image pixels as numpy array]

"""
def load_raw_data_from_file():

	#Kaggle Dataset
	f = open('../Datasets/fer2013/fer2013.csv','r')
	file_reader = csv.reader(f)

	i = -1
	#A list of emotions and image pixel as numpy array
	d = []
	for row in file_reader:
		i += 1

		#Text- Emotion , Pixel in first(0th) row
		if i==0:
			continue
		#row[0] - emotion between 0-6 as string
		emotion = int(row[0])
		#row[1] - pixels as string in row major order, size - 48*48
		pixel_list = row[1].split()
		#convert pixel to numpy array as 48*48 matrix
		image = np.array(pixel_list,dtype='float64').reshape(48,48)

		d.append([emotion,image])

	#Changing to numpy array
	dataset = np.array(d,dtype=object)

	return dataset

#Save preprocessed numpy array "dataset" to file "file_name" in binary format
def save_to_file(file_name,dataset):

	np.save(file_name,dataset)


#Load preprocessed Numpy array from file file_name
def load_from_file(file_name):

	dataset = np.load(file_name)

	return dataset


"""
Show image for 2 seconds
@image_array - Numpy array as image

"""
def show_image(image_array):

	plt.imshow(image_array)
	plt.show(block=False)
	time.sleep(2)
	plt.close()

