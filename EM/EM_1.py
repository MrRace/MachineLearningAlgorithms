#-*- coding: utf-8 -*-
import numpy
import scipy
import scipy.stats

#硬币投掷结果
observations = numpy.array([[1,0,0,0,1,1,0,1,0,1],
                        [1,1,1,1,0,1,1,1,1,1],
                        [1,0,1,1,1,1,1,0,1,1],
                        [1,0,1,0,0,0,1,1,0,0],
                        [0,1,1,1,0,1,1,1,0,1]])

def em_single(priors,observations):

    """
    EM算法的单次迭代
    Arguments
    ------------
    priors:[theta_A,theta_B]
    observation:[m X n matrix]

    Returns
    ---------------
    new_priors:[new_theta_A,new_theta_B]
    :param priors:
    :param observations:
    :return:
    """
    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
    theta_A = priors[0]
    theta_B = priors[1]
    #E step
    for observation in observations:#遍历观测数据
        len_observation = len(observation)#每次的观测数据，10
        num_heads = observation.sum()
        num_tails = len_observation-num_heads
        #二项分布求解公式
        #stats.binom.pmf计算每次观测的概率分布函数。
        #试验结果为1的次数，试验次数，单次出现1的概率。
        #如果第一个参数是一个list，则返回也是一个list
        #例如：
        #stats.binom.pmf(range(6), 5, 1/6.0)
        #运行结果是：
        #array([  4.01877572e-01,   4.01877572e-01,   1.60751029e-01,
        #3.21502058e-02,   3.21502058e-03,   1.28600823e-04])
        #由结果可知：出现0或1次6点的概率为40.2%，而出现3次6点的概率为3.215%。
        contribution_A = scipy.stats.binom.pmf(num_heads,len_observation,theta_A)
        contribution_B = scipy.stats.binom.pmf(num_heads,len_observation,theta_B)

        #将两个概率正规化，得到数据来自硬币A，B的概率：
        weight_A = contribution_A / (contribution_A + contribution_B)
        weight_B = contribution_B / (contribution_A + contribution_B)
        #更新在当前参数下A，B硬币产生的正反面次数
        counts['A']['H'] += weight_A * num_heads
        counts['A']['T'] += weight_A * num_tails
        counts['B']['H'] += weight_B * num_heads
        counts['B']['T'] += weight_B * num_tails

    # M step
    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
    return [new_theta_A,new_theta_B]

def em(observations,prior,tol = 1e-6,iterations=10000):
    """
    EM算法
    ：param observations :观测数据
    ：param prior：模型初值
    ：param tol：迭代结束阈值
    ：param iterations：最大迭代次数
    ：return：局部最优的模型参数
    """
    iteration = 0;
    while iteration < iterations:
        new_prior = em_single(prior,observations)
        delta_change = numpy.abs(prior[0]-new_prior[0])
        if delta_change < tol:
            break
        else:
            prior = new_prior
            iteration +=1
    return [new_prior,iteration]

#假设A硬币向上的概率是0.6，B硬币向上的概率是0.5
rsp = em(observations,[0.6,0.5])#迭代14次就返回了
#[[0.79678875938310978, 0.51958393567528027], 14]，与理论计算结果非常接近，可以再调整误差范围
print(rsp)