# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:07:25 2021

@author: jlliu
"""
import numpy as np

# plt.figure(figsize=(16,16))

# xtest=np.zeros([100,64])
# for i,data in enumerate(xtest[:16]):
#      plt.subplot(4,4,i+1)
#      plt.imshow(data.reshape(8,8))
#      plt.tight_layout()
#      plt.axis('off')
timeelapsed=np.load('res/nlcsg_tcf_mnist_1bit/mnist_1bit_time_elapsed.npz')
time_elapsed_1bit=timeelapsed['time_elapsed']
print(time_elapsed_1bit.shape)
# print(time_elapsed_1bit[:,0,0,:])
print(time_elapsed_1bit[3,0,0,:]/10)

timeelapsed=np.load('res/nlcsg_tcf_mnist_cubic/mnist_cubic_time_elapsed.npz')
time_elapsed_cubic=timeelapsed['time_elapsed']
print(time_elapsed_cubic.shape)
# print(time_elapsed_cubic[:,0,0,:])
print(time_elapsed_cubic[3,0,0,:]/10)



timeelapsed=np.load('res/nlcsg_celebA_1bit/celebA_1bit_time_elapsed_m_4000.npz')
time_elapsed_1bit=timeelapsed['time_elapsed']
print(time_elapsed_cubic.shape)
# print(time_elapsed_cubic[:,0,0,:])
print(time_elapsed_1bit[0,0,0,0,:])



timeelapsed=np.load('res/nlcsg_celebA_cubic/celebA_cubic_time_elapsed_m_4000.npz')
time_elapsed_cubic=timeelapsed['time_elapsed']
print(time_elapsed_cubic.shape)
# print(time_elapsed_cubic[:,0,0,:])
print(time_elapsed_cubic[0,0,0,0,:])