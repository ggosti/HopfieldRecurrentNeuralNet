
import hrnn
import time
import numpy as np
import getCycles

net = [[0.0000000E+0000, - 0.5171256E+0000, -0.2989531E-0001, -0.7808704E+0000, 0.7945437E+0000],
      [-0.5171256E+0000, 0.0000000E+0000, -0.2580485E+0000, -0.6977754E+0000, -0.8953681E+0000],
      [-0.2989531E-0001, -0.2580485E+0000, 0.0000000E+0000, 0.8754015E+0000, 0.9808731E+0000],
      [-0.7808704E+0000, -0.6977754E+0000, 0.8754015E+0000, 0.0000000E+0000, -0.1136637E+0000],
      [0.7945437E+0000, -0.8953681E+0000, 0.9808731E+0000, -0.1136637E+0000, 0.0000000E+0000]]

net = np.array(net).T
#net = getCycles.get_connectivity_matrix(9999, 14,0.0,0.0)
print(net)
print(net.shape)
nets = np.array([net.T])
print(nets.shape)

t1 = time.time()
cycles = getCycles.getCyclesNX(net, net.shape[0], 0, 0)
t2 = time.time()
print('time pyton',t2-t1)
print('cycles python',cycles[0])



t1 = time.time()
#nets = np.ones((4,14,14))
print(nets)
dictMes = hrnn.runRHNNwithNets(0.0,nets)
t2 = time.time()
print(dictMes)
print('time cpp',t2-t1)

print('cycles python',len(cycles[0]), 'cycles cpp',int(dictMes['NClu']))
print('Test result ', len(cycles[0]) == int(dictMes['NClu']))
