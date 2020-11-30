
import hrnn
import time

t1 = time.time()
dictMes = hrnn.runRHNN(0.0,0.0,2)#1000)
t2 = time.time()
print(dictMes)
print(t2-t1)
