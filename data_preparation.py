import csv
from preprocessing import preprocess
import numpy as np
import util
import math

def main():
	
	#Numpy array
	dataset = util.load_raw_data_from_file()

	total_size = dataset.shape[0]
	#ratio is 60:20:20
	training_set_start = 0
	training_set_end = training_set_start + int(math.ceil(total_size*0.6))
	validation_set_start = training_set_end
	validation_set_end = validation_set_start + int(math.ceil(total_size*0.2))
	testing_set_start = validation_set_end
	testing_set_end = total_size

	#Preprocess Training set and save to file in binary format
	training_set = preprocess(dataset[training_set_start:training_set_end])
	util.save_to_file("../Datasets/Kaggle_preprocessed/Trainingset.npy",training_set)

	validation_set = preprocess(dataset[validation_set_start:validation_set_end])
	util.save_to_file("../Datasets/Kaggle_preprocessed/Validationset.npy",validation_set)

	testing_set = preprocess(dataset[testing_set_start:testing_set_end])
	util.save_to_file("../Datasets/Kaggle_preprocessed/Testingset.npy",testing_set)
	



if __name__ == '__main__':
	main()