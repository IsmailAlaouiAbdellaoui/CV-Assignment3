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

  
A_matrix = np.column_stack((x_xprime,x_yprime,x,xprime_y,y_yprime,y,xprime,yprime,np.ones((len(matches),1))))

#getting SVD of A
u, s, vh = np.linalg.svd(A_matrix)

#finding the values of F
f = vh[np.argmin(s)]

#calculating the SVD of F
uf, sf, vhf = np.linalg.svd(f.reshape(3,3).T)

#setting the smallest singular favue to 0
sf[np.argmin(sf)] = 0

#recalculate F values
smat = np.diag(sf)
fundamental_matrix = np.linalg.multi_dot([uf, smat , vhf])

    
pt1 =  np.array([matches[0,0], matches[0,1], 1])
    
pt2 =  np.array([matches[0,2], matches[0,3], 1])
    
dist = np.linalg.norm(pt2.T.dot(fundamental_matrix).dot(pt1))
print(dist)
