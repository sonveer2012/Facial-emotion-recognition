import Input
import FilePaths
import numpy as np
import matplotlib.pyplot as plt
total_data = [0, 0, 0, 0, 0, 0, 0]
train_data = [0, 0, 0, 0, 0, 0, 0]
test_data = [0, 0, 0, 0, 0, 0, 0]
validate_data = [0, 0, 0, 0, 0, 0, 0]
# (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess():
    data = Input.read_data(FilePaths.inp_dir_path)
    print("data_is_loaded")
    train_lable = data.Train.labels
    for i in train_lable:
        for j in range(7):
            if i[j]==1.0:
                train_data[j] = train_data[j] + 1
                total_data[j] = total_data[j] + 1
    test_lable = data.Test.labels
    for i in test_lable:
        for j in range(7):
            if i[j]==1.0:
                test_data[j] = test_data[j] + 1
                total_data[j] = total_data[j] + 1
    validate_label = data.Validate.labels

    for i in validate_label:
        for j in range(7):
            if i[j]==1.0:
                validate_data[j] = validate_data[j] + 1
                total_data[j] = total_data[j] + 1
def draw_barchart_test(data_list):
    print(data_list)
    y_pos = np.arange(len(emotion))
    plt.bar(y_pos, data_list, align='center', alpha=0.5)
    plt.xticks(y_pos, emotion)
    plt.ylabel('Emotion_count')
    plt.title('Testing Dataset')
    plt.show()
def draw_barchart_validate(data_list):
    print(data_list)
    y_pos = np.arange(len(emotion))
    plt.bar(y_pos, data_list, align='center', alpha=0.5)
    plt.xticks(y_pos, emotion)
    plt.ylabel('Emotion_count')
    plt.title('Validate Dataset')
    plt.show()
def draw_barchart_test_val(data_list):
    for i in range(7):
        data_list[i] = data_list[i] + validate_data[i]
    y_pos = np.arange(len(emotion))
    plt.bar(y_pos, data_list, align='center', alpha=0.5)
    plt.xticks(y_pos, emotion)
    plt.ylabel('Emotion_count')
    plt.title('Validate+Testing Dataset')
    plt.show()

preprocess()
print(total_data)
print(train_data)
print(test_data)
print(validate_data)
draw_barchart_test(test_data)
draw_barchart_validate(validate_data)
draw_barchart_test(test_data)
draw_barchart_test_val(test_data)