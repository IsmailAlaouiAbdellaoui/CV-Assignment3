from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from sklearn.metrics import mean_squared_error
import math
import random
import part1_sample_code_python as p1

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
    
norm = np.linalg.norm(pt2.T.dot(fundamental_matrix).dot(pt1))
print(norm)
x = matches[:,0]
y = matches[:,1]
x_p = matches[:,2]
y_p = matches[:,3]

def calculate_mse():
    mse = 0
    for i in range(309):
        x_difference = matches[i,2] - matches[i,0]
        y_difference = matches[i,3] - matches[i,1]
        squared = x_difference**2 + y_difference**2
        result = math.sqrt(squared)
        mse += result
    mse /= 309
    return mse
print(calculate_mse())

def get_random_points(number_points):
    points = np.zeros((number_points,4),dtype=float)
    for i in range(number_points):
        points[i] = matches[random.randint(0,308)]
        
    return points

#test = get_random_points(10)
#print(test)
#threshold = 1
while(True):
    points = get_random_points(10)
    F = p1.fit_fundamental_matrix(points)
    random_point = matches[random.randint(0,308)]
    x = np.append(random_point[0:2],1)
#    x = np.column_stack((random_point[0:2]))
    y = np.append(random_point[2:4],1)
#    print(x)
    threshold = np.linalg.multi_dot([x.T, F, y])
    if threshold ==0:
        break
print(F)
    
#print(x)
#print(y)
    
#    threshold = 
    
    
    
        
    
    

