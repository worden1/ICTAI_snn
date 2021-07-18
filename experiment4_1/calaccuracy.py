import numpy as np
import numba as nb



from mnist import MNIST
mndata = MNIST('./MNIST/')
GrayLevels = 255  # Image GrayLevels
tmax = 256  # Simulatin time
cats = [4, 1, 0, 7, 9, 2, 3, 5, 8, 6]  # Reordering the categories
images = []  # To keep training images
labels = []  # To keep training labels
images_test = []  # To keep test images
labels_test = []  # To keep test labels

Images, Labels = mndata.load_training()
Images=np.array(Images)    
for i in range(len(Labels)):
    if Labels[i] in cats:
        images.append(np.floor(Images[i].reshape(28,28)).astype(int))
        #images.append(np.floor((GrayLevels-Images[i].reshape(28,28))*tmax/GrayLevels).astype(int))
        labels.append(cats.index(Labels[i]))
  
Images, Labels = mndata.load_testing()
Images=np.array(Images)
for i in range(len(Labels)):
    if Labels[i] in cats:
        images_test.append(np.floor(Images[i].reshape(28,28)).astype(int))           
        #images_test.append(np.floor((GrayLevels-Images[i].reshape(28,28))*tmax/GrayLevels).astype(int)) 
        labels_test.append(cats.index(Labels[i]))
                        
del Images,Labels

images = np.transpose(np.asarray(images), (1, 2, 0))
labels = np.asarray(labels)
images_test = np.transpose(np.asarray(images_test), (1, 2, 0))
labels_test = np.asarray(labels_test)

def cal_accuracy_noonehot(labels, readouts, rates):
    temp = np.around(readouts / rates)
    for j in range(len(readouts)): 
        if temp[j] > 9:
            temp[j] = 9
        elif temp[j] < 0:
            temp[j] = 0
        else:
            pass
    num = 0
    for i in range(len(labels)):
        if labels_test[i] == temp[i]:
            num += 1
    test_accu = num / len(labels)
    return test_accu


testlabels = np.load('~/data/mnist_2_100_100_testlabels.npz')
average_readouts_fit = testlabels['average_readouts_fit']
average_readouts_fit_2 = testlabels['average_readouts_fit_2']
test_accuracy = cal_accuracy_noonehot(labels_test, average_readouts_fit, 100)
test_accuracy_2 = cal_accuracy_noonehot(labels_test, average_readouts_fit_2, 100)
print('test accuracy=',test_accuracy,' test accuracy 2=',test_accuracy_2)