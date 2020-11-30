
import hrnn
import time
import numpy as np
import matplotlib.pyplot as plt
import getCycles
from multiprocessing import Pool

def runRHNN(nets):
  dictMes = hrnn.runRHNNwithNets(0.0,nets)
  return dictMes['NClu'],dictMes['Dist']

if __name__ == '__main__':
  p = Pool(6)
  N = 14
  runNum = 1000
  seed = 8889#int(np.random.rand() * 10000)
  np.random.seed(seed)
  rhos = np.linspace(0.0, 1.0, num=100) 

  t1 = time.time()

  rhosNets = []

  for rho in rhos:
    print('rho',rho)
    numLoops = []
    nets = []
    for run in range(runNum):
      #print 'seed',seed
      #C = get_connectivity_matrix(seed,N=N,rho=rho,epsilon=1.0)
      net = getCycles.get_connectivity_matrix_fortranStyle(N=N,rho=rho,epsilon=1.0)
      nets.append(net.T)
    nets = np.array(nets)
    rhosNets.append(nets)
  t2 = time.time()


  CsDs = p.map(runRHNN,rhosNets)
  t3 = time.time()
  print('time hrnn run ',t3-t2)

  C = []
  D = []
  for c,d in CsDs:
    C.append(c)
    D.append(d)
  t4 = time.time() 
  print('time generate nets',t2-t1)
  print('time hrnn run ',t3-t2)
  print('time format data ',t4-t3)
  
  plt.figure()
  plt.plot(rhos,C)

  plt.figure()
  plt.plot(rhos,D)
