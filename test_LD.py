import numpy as np
import LDgenerator as LD
import matplotlib.pyplot as plt
import time
from statsmodels.distributions.empirical_distribution import ECDF
E = np.random.uniform(-1000,1000,(1000,2))
wts = np.random.uniform(1,1,(E.shape[0],1))
start = time.time()
LD = LD.LaguerreDiagram(E,wts,bounded=True)
Ls,Ss,LAs = LD.cell_aixs(downsampling=True)
areas = LD.cell_areas(downsampling=True)
end = time.time()
print('complete, duration: %fs' %(end-start))
# show LD
LD.showLD()
# CDF of cell areas
ecdf_A = ECDF(areas)
plt.plot(ecdf_A.x,ecdf_A.y)
plt.show()
# CDF of long cell axis
ecdf_Ls = ECDF(Ls)
plt.plot(ecdf_Ls.x,ecdf_Ls.y)
plt.show()
# CDF of shortest cell axis
ecdf_Ss = ECDF(Ss)
plt.plot(ecdf_Ss.x,ecdf_Ss.y)
plt.show()
