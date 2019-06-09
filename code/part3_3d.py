from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from sklearn.metrics import mean_squared_error
import math



def null(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T

matches_library = np.loadtxt('../data/library/library_matches.txt')
camera_1_library = np.loadtxt('../data/library/library1_camera.txt')
camera_2_library = np.loadtxt('../data/library/library2_camera.txt')
matches_house = np.loadtxt('../data/house/house_matches.txt')
camera_1_house = np.loadtxt('../data/house/house1_camera.txt')
camera_2_house = np.loadtxt('../data/house/house2_camera.txt')

#print(camera_1_library[2,1])
#linear traingulation - estimate a 3D point X
matches_librar=matches_library[0,1]
#print(matches_library[0,1])
points_3d = np.zeros((len(matches_library),4),dtype=float)
for i in range(len(matches_library)):

    A_matrix_row1 = np.array([matches_library[i,1]*camera_1_library[2,0]-camera_1_library[1,0],
                               matches_library[i,1]*camera_1_library[2,1]-camera_1_library[1,1],
                               matches_library[i,1]*camera_1_library[2,2]-camera_1_library[1,2],
                               matches_library[i,1]*camera_1_library[2,3]-camera_1_library[1,3]])
    
    A_matrix_row2 = np.array([matches_library[i,0]*camera_1_library[2,0]-camera_1_library[0,0],
                               matches_library[i,0]*camera_1_library[2,1]-camera_1_library[0,1],
                              matches_library[i,0]*camera_1_library[2,2]-camera_1_library[0,2],
                               matches_library[i,0]*camera_1_library[2,3]-camera_1_library[0,3]])
    
    A_matrix_row3 = np.array([matches_library[i,3]*camera_2_library[2,0]-camera_2_library[1,0],
                               matches_library[i,3]*camera_2_library[2,1]-camera_2_library[1,1],
                               matches_library[i,3]*camera_2_library[2,2]-camera_2_library[1,2],
                               matches_library[i,3]*camera_2_library[2,3]-camera_2_library[1,3]])
    
    A_matrix_row4 = np.array([matches_library[i,2]*camera_2_library[2,0]-camera_2_library[0,0],
                               matches_library[i,2]*camera_2_library[2,1]-camera_2_library[0,1],
                              matches_library[i,2]*camera_2_library[2,2]-camera_2_library[0,2],
                               matches_library[i,2]*camera_2_library[2,3]-camera_2_library[0,3]])
    


    test = np.vstack((A_matrix_row1,A_matrix_row2,A_matrix_row3,A_matrix_row4))
    #print(test)
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
    points_3d[i] = quasi_f.T.reshape(1,4)
#minimal reprojection error - determine epipolar plane

#test = null(test)
#print(test)
solution = np.zeros((4,1))
#test2 = np.linalg.lstsq(test,solution)
#test3 = np.arange(16).reshape(4,4)
#test4 = np.arange(4)
#test5 = np.dot(test3,test4)
#import cv2
#
#test3 = cv2.triangulatePoints(camera_1_library, camera_2_library, matches_library[0,0:2], matches_library[0,2:4])

def compute_camera_center(camera_projection):
    ucam1, scam1, vcam1 = np.linalg.svd(camera_projection)
    print(vcam1[-1])
    
camera_center1 = compute_camera_center(camera_1_library)
camera_center2 = compute_camera_center(camera_2_library)














