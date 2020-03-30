#!/usr/bin/python
"""
Analyze benchmarks made in cascading hash branch of code.
"""
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt

def follow_your_nose(x,y,seed_idx):
    """
    get top envelope (part of convex hull) of set of 2d pts
    """
    pt = np.asarray(list(zip(x,y)))
    p0 = pt[seed_idx]

    def next_step(cp):
        """
        sort by angle, find max angle as next step
        """
        diff = pt - cp
        cpx = diff[:,0] + 1j*diff[:,1]
        angles = np.angle(cpx)
        imax = np.argmax(angles)
        return imax, angles[imax]

    ret = [seed_idx,]
    cp = p0
    while True:
        imax, ma = next_step(cp)
        if ma <= 0:
            break
        ret.append(imax)
        cp = pt[imax]

    return np.asarray(ret)

if len(sys.argv) < 2:
    print 'usage: ', sys.argv[0], ' BENCH_DICT_PICKLE_FILE '
    sys.exit(0)
benchf = sys.argv[1]

with open(benchf,'r') as f:
    bench = pickle.load(f)

# get basetime
basetime = bench[-1,-1,-1][0]
del bench[-1,-1,-1]

t_cutoff = 5.

# plot full (top-2) performance (both first and second neighbours) vs time
perf = np.asarray([ [x[2], (x[0] / basetime), k] for k,x in bench.iteritems() if x[0] / basetime <= t_cutoff ])
border_idx = follow_your_nose(perf[:,1],perf[:,0],np.argmin(perf[:,1]))
print "TOP-2 PERFORMANCE RESULTS:\n", perf[border_idx,:]
plt.figure(); plt.title("top-2-performance vs time (normalized)")
plt.plot(perf[:,1],perf[:,0],'rx')
plt.xlabel("time (normalized)")
plt.ylabel("performance (normalized to ground-truth)")
plt.plot(perf[border_idx,1],perf[border_idx,0],'bo')


# plot partial (top-1) performance (both first and second neighbours) vs time
perf = np.asarray([ [x[1], (x[0] / basetime), k] for k,x in bench.iteritems() if x[0] / basetime <= t_cutoff ])
border_idx = follow_your_nose(perf[:,1],perf[:,0],np.argmin(perf[:,1]))
print "TOP-1 PERFORMANCE RESULTS:\n", perf[border_idx,:]
plt.figure(); plt.title("top-1-performance vs time (normalized)")
plt.plot(perf[:,1],perf[:,0],'rx')
plt.xlabel("time (normalized)")
plt.ylabel("performance (normalized to ground-truth)")
plt.plot(perf[border_idx,1],perf[border_idx,0],'bo')

plt.show(block=True)
