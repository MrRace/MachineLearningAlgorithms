# -*- coding:utf-8 -*-
'''
三模型问题
'''
import math

[pi,p,q]=[0.4,0.6,0.7]

y=[1,1,0,1,0,0,1,0,1,1]

#书中公式
def cal_u(pi1,p1,q1,yi):
    return pi1*math.pow(p1,yi)*math.pow(1-p1,1-yi)/float(pi1*math.pow(p1,yi)*math.pow(1-p1,1-yi)+(1-pi1)*math.pow(q1,yi)*math.pow(1-q1,1-yi))

def e_step(pi1,p1,q1,y):
    '''
    E步就是计算u
    :param pi1:
    :param p1:
    :param q1:
    :param y:
    :return:
    '''
    return [cal_u(pi1,p1,q1,yi) for yi in y]

def m_step(u,y):
    pi1=sum(u)/len(u)
    p1=sum([u[i]*y[i] for i in range(len(u))]) / sum(u)
    q1=sum([(1-u[i])*y[i] for i in range(len(u))]) / sum([1-u[i] for i in range(len(u))])
    return [pi1,p1,q1]

def run(start_y,start_pi,start_p,start_q,iter_num):
    for i in range(iter_num):
        u = e_step(start_pi,start_p,start_q,y)
        print("len(u)=",len(u))
        print(i,[start_pi,start_p,start_q])
        if [start_pi,start_p,start_q]==m_step(u,y):
            break
        else:
            [start_pi,start_p,start_q]=m_step(u,y)

run(y,pi,p,q,100)