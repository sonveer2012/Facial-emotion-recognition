import numpy as np
import matplotlib.pyplot as plt
import util
import constants as  cnst


# Make sure that caffe is on the python path:
caffe_root = '~/Desktop/Projects/deep_learning/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/Sharad/Desktop/Projects/Major/models/caffe_model_1/deploy.prototxt'
PRETRAINED = '/home/Sharad/Desktop/Projects/Major/models/caffe_model_1/snapshot/_iter_5000.caffemodel'

testing_dataset = util.load_from_file('/home/Sharad/Desktop/Projects/Major/Datasets/Kaggle_preprocessed/Testingset.npy')

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       image_dims=(cnst.IMAGE_WIDTH, cnst.IMAGE_HEIGHT))

cnt_correct = 0
cnt_total = testing_dataset.shape[0]

#Can be predicted without for-loop
for i in xrange(testing_dataset.shape[0]):

	label = testing_dataset[i][0]
	image = testing_dataset[i][1]

	#Show image
	#util.show_image(image)

	#image = W*H, required = H*W*K
	image = np.resize(image,(image.shape[0],image.shape[1],1))
	image = np.rollaxis(image,1,0)

	prediction = net.predict([image])

	if prediction[0].argmax()==label:
		cnt_correct += 1
		print i,cnt_correct

print "Testing Accuracy " + str((cnt_correct*1.0)/cnt_total)


''''
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)

prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
print 'prediction shape:', prediction[0].shape
plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
plt.show()
'''
