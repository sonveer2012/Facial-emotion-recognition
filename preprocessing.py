import numpy as np

"""
Preprocess the image

@dataset : Dataset as numpy array[emotions, image pixels as numpy array]

@return : Preprocessed Input in the same format

"""
def preprocess(dataset):

	images = dataset[:,1]
	#Find mean image
	#Only 1 color channel in input
	mean_image = np.mean(images)
	mean_images = [np.subtract(i,mean_image) for i in images]

	#Change to numpy array
	#mean_images = np.array(mean_images)

	#Images already 48*48 with pixel values 0-255
	#So, normalization not necessary

	#PCA not required in CNN

	#DataAugmentation can be done such as adding flipped images into datasets

	dataset[:,1] = mean_images

	return dataset




