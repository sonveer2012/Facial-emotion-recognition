import numpy as np
import collections
import os
import Constants as cnst

Datasets = collections.namedtuple("Datasets", ["Train", "Test", "Validate"])

class DataSet(object):

    def  __init__(self, images, labels):
        self._images = images
        self._labels = labels

    #Images in the dataset.Should be a numpy array
    @property
    def images(self):
        return self._images

    #Image labels corresponding to images in the dataset.
    @property
    def labels(self):
        return self._labels

    #return a random subset from the dataset
    #batch_size : cardinality of subset
    def next_batch(self,batch_size):
        if len(self._images) < batch_size:
            print("Batch size greater than dataset size\n")
        else:
            perm = []
            for _ in range(batch_size):
                rand_index = (int)(np.random.rand()*len(self._images))
                perm.append(rand_index)
            perm = np.array(perm)
            return self._images[perm], self._labels[perm]

#Read the data into training, testing and validation. Returns a Datasets object.
#dir_path : directory path where dataset is stored. Dataset to be stored in .npy format with file names: train.npy, test.npy, validate.npy
def read_data(dir_path):
    train_path = os.path.join(dir_path, "train.npy")
    train_images, train_labels = extract_data(train_path)
    Train = DataSet(train_images, train_labels)
    test_path = os.path.join(dir_path,"test.npy")
    test_images, test_labels = extract_data(test_path)
    Test = DataSet(test_images, test_labels)
    validate_path = os.path.join(dir_path ,"validate.npy")
    validate_images, validate_labels = extract_data(validate_path)
    Validate = DataSet(validate_images, validate_labels)
    return Datasets(Train = Train, Test = Test, Validate = Validate)

#Extract image and label data from file.
#file_path : path of file to extract data from.
def extract_data(file_path):
    data = np.load(file_path,encoding = 'latin1')
    total_size = data.shape[0]
    images = data[:,1]
    labels = np.zeros(shape=(total_size ,cnst.classes))
    for i in range(total_size):
        labels[i,data[i,0]] = 1
        images[i] = images[i].flatten()
    lst_images = images.tolist()
    images = np.array(lst_images)
    return images, labels