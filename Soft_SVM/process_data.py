#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:22:03 2019

@author: superzhy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:13:33 2019

@author: superzhy
"""

from csv import reader
 
## Load a CSV file
def load(files):
    fh = open("pcdata.txt", 'w')
    for filename in files:
        f = open(filename, "r")
        #f.next()
        for line in f:
            d = line.split()
            if len(d)>10:
                for i in range(3):
                    fh.write(d[i]+ " " )
                
                for i in range(5,len(d)-1):
                    fh.write(d[i]+ " " )
                    
                if (int(d[4]) == 1004):
                    label = 0 #veg
                elif (int(d[4]) == 1100):
                    label = 1 #wire
                elif (int(d[4]) == 1103):
                    label = 2 #pole
                elif (int(d[4]) == 1200):
                    label = 3 #ground
                elif (int(d[4]) == 1400):
                    label = 4 #facade
                    
                fh.write(str(label) + "\n" )    
#
#from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import random
#
#def load_data(filename):
#    f = open(filename, 'r')
#    x,y,z = list(),list(),list()
#    for line in f:
#        d = line.split()
#        x.append(float(d[0]))
#        y.append(float(d[1]))
#        z.append(float(d[2]))
#    return x,y,z
#
#x,y,z = load_data('processed.txt')
#fig = plt.figure()
#ax = Axes3D(fig)
#
#
#
#ax.scatter(x, y, z, s =0.2)
#plt.show()
















