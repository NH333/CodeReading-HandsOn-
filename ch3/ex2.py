from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

mnist = fetch_mldata('MNIST original')
X = mnist["data"]
y = mnist["target"]

test = X[36000].reshape(28,28)
# plt.matshow(test.reshape(28,28),cmap=plt.cm.gray)
# plt.show()

# print("矩阵行数�?%d | 矩阵列数�?%d " % (test.shape[0],test.shape[1]))
ZuoYou = 0
ShangXia = 0



ZuoYou = int(input('left - right: '))
ShangXia = int(input('up - down: '))
while ShangXia > 28 or ShangXia < -28:
    print("please reinput number: \n")
    ShangXia = int(input('up - down: '))
while ZuoYou > 28 or ZuoYou < -28:
    print("please reinput number: \n")
    ZuoYou = int(input('left - right: '))

# tmp = []
# result_r1 = []
# shape_row = 28
# result = np.array(0)
print(type(ShangXia))
#先实现向上和向下
#负数表示向下，�?�数表示向上
def move_ShangXia(ShangXia,Image):
    tmp = []
    result_r1 = []
    shape_row = 28
    if ShangXia >= 0:
        for i in range(28):
            if i < ShangXia:
                tmp.append(Image[i].tolist()) #把array�?换成list
            else:
                result_r1 = Image[i:].tolist()
                
                for z in range(ShangXia):
                    result_r1.append(tmp[z])                  #append貌似一次只能压入一�?�?list�?嵌�?�的一组list
                result_r1 = [j for r in result_r1 for j in r] #把嵌套的list合并成一�?大的list

                result = np.array(result_r1)
                break
    else :
        for i in range(28):
            if i < -ShangXia:
                result_r1.append(Image[shape_row-i-1].tolist())
                
            else:
                result_r1.reverse()
                tmp = Image[:shape_row-i].tolist()

                for z in range(len(tmp)):
                    result_r1.append(tmp[z])
                result_r1 = [j for r in result_r1 for j in r]

                result = np.array(result_r1)
                break

    return result.reshape(28,28)

#实现左右移动
#正数表示向左，负数表示向�?
def move_ZuoYou(ZuoYou,Image):
    tmp = []
    result_r1 = []
    shape_row = 28
    Image = Image.T
    if ZuoYou >= 0:
        for i in range(28):
            if i < ZuoYou:
                tmp.append(Image[i].tolist())
            else:
                result_r1 = Image[i:].tolist()
                
                for z in range(ZuoYou):
                    result_r1.append(tmp[z])
                result_r1 = [j for r in result_r1 for j in r]

                result = np.array(result_r1)
                break
    else :
        for i in range(28):
            if i < -ZuoYou:
                result_r1.append(Image[shape_row-i-1].tolist())
                
            else:
                result_r1.reverse()
                tmp = Image[:shape_row-i].tolist()

                for z in range(len(tmp)):
                    result_r1.append(tmp[z])
                result_r1 = [j for r in result_r1 for j in r]

                result = np.array(result_r1)
                break
    result = result.reshape(28,28)
    
    return result.T

if ShangXia:
    a = move_ShangXia(ShangXia,test)
if ZuoYou:
    b = move_ZuoYou(ZuoYou,a)

plt.matshow(b,cmap=plt.cm.gray)
plt.show()
