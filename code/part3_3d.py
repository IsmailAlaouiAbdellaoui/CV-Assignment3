from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from sklearn.metrics import mean_squared_error
import math

matches_library = np.loadtxt('../data/library/library_matches.txt')
camera_1_library = np.loadtxt('../data/library/library1_camera.txt')
camera_2_library = np.loadtxt('../data/library/library2_camera.txt')
matches_house = np.loadtxt('../data/house/house_matches.txt')
camera_1_house = np.loadtxt('../data/house/house1_camera.txt')
camera_2_house = np.loadtxt('../data/house/house2_camera.txt')

#print(camera_1_library[2,1])
#linear traingulation - estimate a 3D point X
matches_librar=matches_library[:,1]
A_matrix_row1 = np.array([np.dot(matches_library[:,1],camera_1_library[2,0])-camera_1_library[1,0],
                           np.dot(matches_library[:,1],camera_1_library[2,1])-camera_1_library[1,1],
                           np.dot(matches_library[:,1],camera_1_library[2,2])-camera_1_library[1,2],
                           np.dot(matches_library[:,1],camera_1_library[2,3])-camera_1_library[1,3]])

A_matrix_row2 = np.array([np.dot(matches_library[:,0],camera_1_library[2,0])-camera_1_library[0,0],
                           np.dot(matches_library[:,0],camera_1_library[2,1])-camera_1_library[0,1],
                           np.dot(matches_library[:,0],camera_1_library[2,2])-camera_1_library[0,2],
                           np.dot(matches_library[:,0],camera_1_library[2,3])-camera_1_library[0,3]])

A_matrix_row3 = np.array([np.dot(matches_library[:,3],camera_2_library[2,0])-camera_2_library[1,0],
                           np.dot(matches_library[:,3],camera_2_library[2,1])-camera_2_library[1,1],
                           np.dot(matches_library[:,3],camera_2_library[2,2])-camera_2_library[1,2],
                           np.dot(matches_library[:,3],camera_2_library[2,3])-camera_2_library[1,3]])

A_matrix_row4 = np.array([np.dot(matches_library[:,2],camera_2_library[2,0])-camera_2_library[0,0],
                           np.dot(matches_library[:,2],camera_2_library[2,1])-camera_2_library[0,1],
                           np.dot(matches_library[:,2],camera_2_library[2,2])-camera_2_library[0,2],
                           np.dot(matches_library[:,2],camera_2_library[2,3])-camera_2_library[0,3]])

test = np.stack((A_matrix_row1,A_matrix_row2,A_matrix_row3,A_matrix_row4))
print(test)
#for i in len(camera_1_library[i,:])
#u = np.dot(camera_1_library,matches_library)
#u_prime = np.dot(camera_2_library,matches_library)

#minimal 3D error - X midpoint between back projected image points


#minimal algebraic error- P determined by SVD
uc1l, sc1l, vc1l = np.linalg.svd(test)
#uc2l, sc2l, vc2l = np.linalg.svd(camera_2_library)
#finding the values of F
fc1l = vc1l[np.argmin(sc1l)]
#calculating the SVD of F
ufc1l, sfc1l, vhfc1l = np.linalg.svd(fc1l.reshape(2,2).T)

#setting the smallest singular favue to 0
sfc1l[np.argmin(sfc1l)] = 0

#recalculate F values
smat = np.diag(sfc1l)
quasi_f = np.linalg.multi_dot([ufc1l, smat , vhfc1l])
#minimal reprojection error - determine epipolar plane
