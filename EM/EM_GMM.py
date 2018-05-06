# -*- coding:utf-8 -*-

import numpy as np

tol=1e-2

# a=[1,2,3,4]
# b=[2,4,2,6]
# d=list(map(lambda x: x[0]-x[1], zip(a, b)))
# print(d)
# d = np.abs(d)
# if max(d) < 1:
#     print("yes")
# else:
#     print("No")
# a=[1.3,-0.00004]
# print(np.abs(a))


def generateData(k,mu,sigma,dataNum):
    '''
    产生混合高斯模型的数据
    :param k: 比例系数
    :param mu: 均值
    :param sigma: 标准差
    :param dataNum:数据个数
    :return: 生成的数据
    '''
    # 初始化数据
    dataArray = np.zeros(dataNum,dtype=np.float32)
    # 逐个依据概率产生数据
    # 高斯分布个数
    n = len(k)
    for i in range(dataNum):
        # 产生[0,1]之间的随机数
        rand = np.random.random()
        Sum = 0
        index = 0
        while(index < n):
            Sum += k[index]
            if(rand < Sum):
                dataArray[i] = np.random.normal(mu[index],sigma[index])
                break
            else:
                index += 1
    return dataArray

def normPdf(x,mu,sigma):
    '''
    计算均值为mu，标准差为sigma的正态分布函数的密度函数值
    :param x: x值
    :param mu: 均值
    :param sigma: 标准差
    :return: x处的密度函数值
    '''
    return (1./np.sqrt(2*np.pi))*(np.exp(-(x-mu)**2/(2*sigma**2)))



def em(dataArray,k,mu,sigma,step = 10):
    '''
    em算法估计高斯混合模型
    :param dataNum: 已知数据个数
    :param k: 每个高斯分布的估计系数
    :param mu: 每个高斯分布的估计均值
    :param sigma: 每个高斯分布的估计标准差
    :param step:迭代次数
    :return: em 估计迭代结束估计的参数值[k,mu,sigma]
    '''
    # 高斯分布个数
    n = len(k)
    # 数据个数
    dataNum = dataArray.size
    # 初始化gama数组
    gamaArray = np.zeros((n,dataNum))
    tmp_mu=[0 for n in range(n)]
    tmp_sigma=[0 for n in range(n)]
    tmp_k =[0 for n in range(n)]
    for s in range(step):
        for i in range(n):
            for j in range(dataNum):
                Sum = sum([k[t]*normPdf(dataArray[j],mu[t],sigma[t]) for t in range(n)])
                gamaArray[i][j] = k[i]*normPdf(dataArray[j],mu[i],sigma[i])/float(Sum)
        # 更新 mu
        for i in range(n):
            mu[i] = np.sum(gamaArray[i]*dataArray)/np.sum(gamaArray[i])

        # 更新 sigma
        for i in range(n):
            sigma[i] = np.sqrt(np.sum(gamaArray[i]*(dataArray - mu[i])**2)/np.sum(gamaArray[i]))


        # 更新系数k
        for i in range(n):
            k[i] = np.sum(gamaArray[i])/dataNum


        #在误差范围内，稳定
        delta_change_mu = list(map(lambda x: x[0] - x[1], zip(tmp_mu, mu)))
        delta_change_mu = np.absolute(delta_change_mu)

        delta_change_sigma = list(map(lambda x: x[0] - x[1], zip(tmp_sigma, sigma)))
        delta_change_sigma = np.absolute(delta_change_sigma)

        delta_change_k = list(map(lambda x: x[0] - x[1], zip(tmp_k, k)))
        delta_change_k = np.absolute(delta_change_k)

        if max(delta_change_mu) < tol and max(delta_change_sigma) < tol and max(delta_change_k)< tol:
            print("comes stable,step num=", s)
            break
        #记录旧的值
        tmp_mu = mu.copy()
        tmp_sigma = sigma.copy()
        tmp_k = k.copy()
        #默认是指针方式，不会真正开辟新空间的
        if s == step-1:
            print("maybe more steps")
        # print("delta_change_mu=",delta_change_mu)
        # print("max delta_change_mu=", max(delta_change_mu))
        # print("mu=", mu)
    return [k,mu,sigma]





if __name__ == '__main__':
    # 参数的准确值
    k = [0.3,0.4,0.3]
    mu = [2,4,3]
    sigma = [1,1,4]
    # 样本数
    dataNum = 5000
    # 产生数据
    dataArray = generateData(k,mu,sigma,dataNum)
    # 参数的初始值
    # 注意em算法对于参数的初始值是十分敏感的
    k0 = [0.3,0.3,0.4]
    mu0 = [1,3,2]
    sigma0 = [1,1,1]
    step = 100
    # 使用em算法估计参数
    k1,mu1,sigma1 = em(dataArray,k0,mu0,sigma0,step)
    # 输出参数的值
    print("参数实际值:")
    print("k:",k)
    print("mu:",mu)
    print("sigma:",sigma)
    print("参数估计值:")
    print("k1:",k1)
    print("mu1:",mu1)
    print("sigma1:",sigma1)

'''
参数实际值:
k: [0.3, 0.4, 0.3]
mu: [2, 4, 3]
sigma: [1, 1, 4]
参数估计值:
k1: [0.47705113093778984, 0.22412094388380474, 0.29882792517840545]
mu1: [2.8741058325624653, 3.3334232653871783, 3.3334232653871783]
sigma1: [3.3459918996502322, 1.4827735604822387, 1.4827735604822365]
'''