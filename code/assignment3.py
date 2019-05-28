from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

matches = np.loadtxt('../data/library/library_matches.txt')
#print(matches[:,3])
#print(matches[0])
#print(matches[1])
print(matches)
print("\n")
x_xprime = matches[:,0]* matches[:,2]
x_yprime = matches[:,0] * matches[:,3]
x = matches[:,0]
xprime_y = matches[:,2] * matches[:,1]
y_yprime = matches[:,1] * matches[:,3]
y = matches[:,1]
xprime = matches[:,2]
yprime = matches[:,3]

  
A_matrix = np.column_stack((x_xprime,x_yprime,x,xprime_y,y_yprime,y,xprime,yprime,np.ones((309,1))))

vectors,values,vh = np.linalg.svd(A_matrix)
min_vector = vh[np.argmin(values)]
#print(np.argmin(values)) 

