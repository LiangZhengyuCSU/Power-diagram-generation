import numpy as np
import LDgenerator as LD
import time
E = np.loadtxt(r'.\test_points.txt')
wts = np.random.uniform(1,1,(E.shape[0],1))
start = time.time()
LD = LD.LaguerreDiagram(E, wts)
end = time.time()
print('complete, duration: %fs' %(end-start))
LD.showLD()