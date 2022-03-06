# coding: utf-8
#----------------------------------------------------------------
# Name: SVM
"""
支持向量机
用于区分手写字母abc
"""

from scipy.misc import imread
import cv2
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def boundaries(binarized, axis): # 2值化后的大数组 坐标轴

    rows = np.sum(binarized, axis=[1, 0][axis]) > 0 # 在这个坐标轴上求和 1*3013
    rows[1:] = np.logical_xor(rows[1:], rows[:-1]) # 边缘检测？当前和前一个
    change = np.nonzero(rows)[0] # 记录第几列[28,60,169,202,297,326……] 有1 3 5 7 没2 4 6 8
    ymin = change[::2]
    ymax = change[1::2]
    height = ymax-ymin # 两个边缘的距离
    too_small = 10 # 10像素以下不考虑了
    ymin = ymin[height > too_small] # 正常数据
    ymax = ymax[height > too_small] # 正常数据
    return zip(ymin, ymax)


def separate(abc_list):
    orig_img = abc_list.copy()
    pure_white = 255.
    white = np.max(abc_list)
    black = np.min(abc_list)
    thresh = (white+black)/2.0 # （最大值+最小值）/2.0 ？？？
    binarized = abc_list < thresh # 用中值区分 二值化的大数组
    row_bounds = boundaries(binarized, axis=0)
    cropped = []
    for top, bottom in row_bounds: # 打包的ymin ymax每次取一对
        abc_list = binarized[top:bottom,:] # 上下夹
        left, right = zip(*boundaries(abc_list, axis=1)) # 左右夹
        rects = [top, bottom, left[0], right[0]] # 上下左右 在大数组中的位置
        # print('上下左右=', top, bottom, left[0], right[0])
        cropped.append(np.array(orig_img[rects[0]:rects[1], rects[2]:rects[3]] / pure_white))
    return cropped


def partition(data_set, label_set, training_ratio):

    label_num = len(np.unique(label_set)) # 这里3
    sum_num = int(len(data_set)/label_num)
    train_num = int(sum_num * training_ratio)
    test_num = int(sum_num - train_num)

    # _train_label = np.chararray(int(label_num*train_num))
    # _test_label = np.chararray(int(label_num*test_num))
    _train_label = np.empty(int(label_num * train_num), dtype=label_set.dtype)
    _test_label = np.empty(int(label_num*test_num), dtype=label_set.dtype)

    _train_data = np.empty([len(_train_label), len(data_set[1])])
    _test_data = np.empty([len(_test_label), len(data_set[1])])
    for _target in np.arange(label_num): # 对每组标号
        for _i in np.arange(sum_num): # 对每组中的每个数据
            if _i < train_num: # 在约定的训练集大小之内
                _train_label[_i+(_target*train_num-1)] = label_set[_target * sum_num]
                _train_data[_i+(_target*train_num-1)] = data_set[_i+(_target*sum_num-1)]
            else: # 在约定的测试集大小之内
                _test_label[(_target*test_num)+sum_num-_i-1] = label_set[_target * sum_num]
                _test_data[(_target*test_num)+sum_num-_i-1] = data_set[_i+(_target*sum_num-1)]
    return _train_data, _train_label, _test_data, _test_label

def main():

    """主程序 参数"""
    visualize = True
    training_percent =.15
    resized_w = 5
    resized_h = 5

    letters = list()
    for i in range(ord('a'),ord('z')+1):
        letters.append(chr(i))
    """读入手写图像"""
    original_pic=list()
    for i in range(3):
        original_pic.append(imread(letters[i]+'.png',flatten=True))
    """格式化图片"""
    # 分割图像 每个字符上下左右夹后重整到10*10
    seperated_pic = list()
    resized_pic = list()
    data=resized_pic
    for i in range(len(original_pic)):
        seperated_pic.append(original_pic[i])
        for img in seperated_pic[i]:
            resized_pic.append(cv2.resize(img,(resized_w,resized_h)))
    """整理数据和标号"""
    data=np.asarray(data)
    data=data.reshape((-1,resized_w*resized_h))

    #全数据数组 data[第几个图片，10，10]
    label=np.empty(len(data),dtype=int)
    for i in range(len(data)):
        if i <=22:
            label[i]=ord(letters[0])
        elif 23<= i <=45:
            label[i]=ord(letters[1])
        elif 46<= i <=68:
            label[i]=ord(letters[2])

    train_data,train_label,test_data,test_label=partition(data,label,training_percent)

    """训练分类器 和 测试其正确率"""
    # clf=svm.LinearSVC()
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    clf=svm.SVC(C=1.0,kernel='rbf',gamma='auto')
    clf.fit(train_data,train_label)
    predict=clf.predict(test_data)
    correct=len(predict)
    for i in range(len(predict)):
        if predict[i]!=test_label[i]:
            correct-=1
    """打印结果"""
    for i in range(len(predict)):
        print(i+1,chr(test_label[i]),'->',chr(predict[i]))
        print("总数据量: %d类标号* %d个样本" % (len(np.unique(label)), int(len(data) / len(np.unique(label)))))
        train_data_num = int(int(len(data) / len(np.unique(label))) * training_percent)
        test_data_num = int(int(len(data) / len(np.unique(label))) - train_data_num)
        print("训练数据集%d, " % train_data_num, end='')
        print("测试数据集%d" % test_data_num, end='')
        print("(训练数据占总数据{0}%)".format(training_percent * 100))
        print("测试集正确率{0}%".format(100 * correct / len(predict)))

        """ 可视化"""
        if visualize:
            test_data = test_data.reshape(-1, resized_h, resized_w)
        images_and_labels = list(zip(test_data, predict))
        for index, (image, label) in enumerate(images_and_labels[::2]):
            plt.subplot(6, 5, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap='gray', interpolation='nearest')
        plt.title('predict:%s' % chr(label))
        plt.show()

if __name__ == '__main__':
    main()