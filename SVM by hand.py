import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score

np.random.seed(12)
num_observations = 50

# todo:check the np.random.multivariate normal function
# np.hstack

x1=np.random.multivariate_normal([0,0],[[1,0.75],[0.75,1]],num_observations)
x2=np.random.multivariate_normal([1,4],[[1,0.75],[0.75,1]],num_observations)

X=np.vstack((x1,x2)).astype(np.float32)
y=np.hstack((np.zeros(num_observations),np.ones(num_observations)))
y=np.where(y<=0,-1,1)

#Display the output
plt.figure(figsize=(12,8))
plt.scatter(X[:,0],X[:,1],c=y,alpha=.4)
plt.show()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf=SVC(kernel='linear',C=1e5)  #创建分类器对象
clf.fit(X,y)  #用训练数据拟合分类器模型
y_pred=clf.predict(X)   #用训练好的分类器去预测X数据的标签
print(accuracy_score(y,y_pred))  #计算准确率



def Lagrangian(w,alpha):
    first_part=np.sum(alpha)
    second_part=np.sum(np.dot(alpha*alpha*y*y*X.T,X))
    res=first_part-second_part*0.5
    return res

def gradient_(w,X,y,b,lr):
    for i in range(2000):
        for idx,x_i in enumerate(X):
            y_i=y[idx]

            cond=y_i*(np.dot(w,x_i)-b)>=1
            if cond:
                w-=lr*2*w
            else:
                w-=lr*(2*w-np.dot(x_i,y_i))
                b-=lr*y_i
    return w,b

w,b,lr=np.random.random(X.shape[1]),0,0.0001
w,b=gradient_(w,X,y,b,lr)
def predict(w,b,X):
    return np.sign(np.dot(X,w)-b)

svm_predictions=predict(w,b,X)
print(accuracy_score(y,svm_predictions))


def gradient_(w,X,y,lr):
    for i in range(2000):
        for idx,x_i in enumerate(X):
            y_i=y[idx]

            cond=y_i*(np.dot(w,x_i)-b)>=1
            if cond:
                w-=lr*2*w
            else:
                w-=lr*(2*w-np.dot(x_i,y_i))
    return w

w,lr=np.random.random(X.shape[1]),0.0001
w=gradient_(w,X,y,lr)


def predict(X,w):
    pred=np.dot(X,w)
    return np.sign(pred)

svm_pred=predict(X,w)

print(accuracy_score(y,svm_pred))
