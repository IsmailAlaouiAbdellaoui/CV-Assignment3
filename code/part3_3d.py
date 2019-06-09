from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import cv2

matches_library = np.loadtxt('../data/library/library_matches.txt')
camera_1_library = np.loadtxt('../data/library/library1_camera.txt')
camera_2_library = np.loadtxt('../data/library/library2_camera.txt')
matches_house = np.loadtxt('../data/house/house_matches.txt')
camera_1_house = np.loadtxt('../data/house/house1_camera.txt')
camera_2_house = np.loadtxt('../data/house/house2_camera.txt')

def compute_camera_center(camera_projection):
    ucam1, scam1, vcam1 = np.linalg.svd(camera_projection)
    return vcam1[:,-1]


def triangulation_plotting_library(matches_library,camera1_library,camera2_library):
    points_3d_library = np.zeros((len(matches_library),4),dtype=float)
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
    
        #minimal 3D error - X midpoint between back projected image points
        
        
        #minimal algebraic error- P determined by SVD
        uc1l, sc1l, vc1l = np.linalg.svd(test)
#
#        #finding the values of F
#        fc1l = vc1l[np.argmin(sc1l)]
#        #calculating the SVD of F
#        ufc1l, sfc1l, vhfc1l = np.linalg.svd(fc1l.reshape(2,2).T)
#        
#        #setting the smallest singular favue to 0
#        sfc1l[np.argmin(sfc1l)] = 0
#        
#        #recalculate F values
#        smat = np.diag(sfc1l)
#        quasi_f = np.linalg.multi_dot([ufc1l, smat , vhfc1l])
        
        temp = (-1) * vc1l[-1]  
        temp = temp / temp[-1]  
        
        
        
        points_3d_library[i] = temp.T.reshape(1,4)
        
    camera_center1 = compute_camera_center(camera_1_library)
    camera_center1 = camera_center1/camera_center1[-1]
    
    camera_center2 = compute_camera_center(camera_2_library)
    camera_center2 = camera_center2/camera_center2[-1]
    
    fig = plt.figure()
    fig.suptitle('3D reconstruction of the Library', fontsize=16)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(camera_center1[0],camera_center1[1],camera_center1[2],c='r',marker='+',label="Camera 1 center")
    ax.scatter(camera_center2[0],camera_center2[1],camera_center2[2],c='g',marker='+',label="Camera 2 center")
    for i in range(len(points_3d_library)):
        ax.scatter(points_3d_library[i][0],points_3d_library[i][1],points_3d_library[i][2],c='b')
    plt.legend()
    plt.axis('equal')
    plt.xlabel("x axis")
    plt.ylabel("y axis")
#    plt.zlabel("z axis")
    return points_3d_library




def triangulation_plotting_house(matches_house,camera_1_house,camera_2_house):
    points_3d_house = np.zeros((len(matches_house),4),dtype=float)
    for i in range(len(matches_house)):
    
        A_matrix_row1 = np.array([matches_house[i,1]*camera_1_house[2,0]-camera_1_house[1,0],
                                   matches_house[i,1]*camera_1_house[2,1]-camera_1_house[1,1],
                                   matches_house[i,1]*camera_1_house[2,2]-camera_1_house[1,2],
                                   matches_house[i,1]*camera_1_house[2,3]-camera_1_house[1,3]])
        
        A_matrix_row2 = np.array([matches_house[i,0]*camera_1_house[2,0]-camera_1_house[0,0],
                                   matches_house[i,0]*camera_1_house[2,1]-camera_1_house[0,1],
                                  matches_house[i,0]*camera_1_house[2,2]-camera_1_house[0,2],
                                   matches_house[i,0]*camera_1_house[2,3]-camera_1_house[0,3]])
        
        A_matrix_row3 = np.array([matches_house[i,3]*camera_2_house[2,0]-camera_2_house[1,0],
                                   matches_house[i,3]*camera_2_house[2,1]-camera_2_house[1,1],
                                   matches_house[i,3]*camera_2_house[2,2]-camera_2_house[1,2],
                                   matches_house[i,3]*camera_2_house[2,3]-camera_2_house[1,3]])
        
        A_matrix_row4 = np.array([matches_house[i,2]*camera_2_house[2,0]-camera_2_house[0,0],
                                   matches_house[i,2]*camera_2_house[2,1]-camera_2_house[0,1],
                                  matches_house[i,2]*camera_2_house[2,2]-camera_2_house[0,2],
                                   matches_house[i,2]*camera_2_house[2,3]-camera_2_house[0,3]])
        
    
    
        test = np.vstack((A_matrix_row1,A_matrix_row2,A_matrix_row3,A_matrix_row4))
    
        #minimal 3D error - X midpoint between back projected image points
        
        
        #minimal algebraic error- P determined by SVD
        uc1l, sc1l, vc1l = np.linalg.svd(test)
        #uc2l, sc2l, vc2l = np.linalg.svd(camera_2_library)
        #finding the values of F
#        fc1l = vc1l[np.argmin(sc1l)]
#        #calculating the SVD of F
#        ufc1l, sfc1l, vhfc1l = np.linalg.svd(fc1l.reshape(2,2).T)
#        
#        #setting the smallest singular favue to 0
#        sfc1l[np.argmin(sfc1l)] = 0
#        
#        #recalculate F values
#        smat = np.diag(sfc1l)
#        quasi_f = np.linalg.multi_dot([ufc1l, smat , vhfc1l])
        
        temp = (-1) * vc1l[-1]  
        temp = temp / temp[-1]  
#        
        
        
        points_3d_house[i] = temp.T.reshape(1,4)
        
    camera_center1 = compute_camera_center(camera_1_house)
    camera_center1 = camera_center1/camera_center1[-1]
    
    camera_center2 = compute_camera_center(camera_2_house)
    camera_center2 = camera_center2/camera_center2[-1]
    
    fig = plt.figure()
    fig.suptitle('3D reconstruction of the house', fontsize=16)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(camera_center1[0],camera_center1[1],camera_center1[2],c='r',marker='+',label="Camera 1 center")
    ax.scatter(camera_center2[0],camera_center2[1],camera_center2[2],c='g',marker='+',label="Camera 2 center")
    for i in range(len(points_3d_house)):
        ax.scatter(points_3d_house[i][0],points_3d_house[i][1],points_3d_house[i][2],c='b')
    plt.legend()
    plt.axis('equal')
    plt.xlabel("x axis")
    plt.ylabel("y axis")
#    plt.zlabel("z axis")
    return points_3d_house









def test_library_triangulation(matches_library,camera_1_library,camera_2_library):
    points_3d_library = np.zeros((len(matches_library),4),dtype=float)
    for i in range(len(matches_library)):
#        print(matches_house[i,2:4].shape)
        points_3d_library[i] = cv2.triangulatePoints(camera_1_library, camera_2_library, matches_library[i,0:2], matches_library[i,2:4]).reshape(1,4)
        points_3d_library[i] = points_3d_library[i]/points_3d_library[i][-1]
#        print(point)
    camera_center1 = compute_camera_center(camera_1_library)
    camera_center1 = camera_center1/camera_center1[-1]
    
    camera_center2 = compute_camera_center(camera_2_library)
    camera_center2 = camera_center2/camera_center2[-1]
    
    fig = plt.figure()
    fig.suptitle('3D reconstruction of the Library using OpenCV', fontsize=16)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(camera_center1[0],camera_center1[1],camera_center1[2],c='r',marker='+',label="Camera 1 center")
    ax.scatter(camera_center2[0],camera_center2[1],camera_center2[2],c='g',marker='+',label="Camera 2 center")
    for i in range(len(points_3d_library)):
        ax.scatter(points_3d_library[i][0],points_3d_library[i][1],points_3d_library[i][2],c='b')
    plt.legend()
    plt.axis('equal')
    plt.xlabel("x axis")
    plt.ylabel("y axis")
#    plt.zlabel("z axis")
    return points_3d_library
    
#open_cv_method = test_library_triangulation(matches_library,camera_1_library,camera_2_library)
our_method = triangulation_plotting_library(matches_library,camera_1_library,camera_2_library)

def test_house_triangulation(matches_house,camera_1_house,camera_2_house):
    points_3d_house = np.zeros((len(matches_house),4),dtype=float)
    for i in range(len(matches_house)):
#        print(matches_house[i,2:4].shape)
        points_3d_house[i] = cv2.triangulatePoints(camera_1_house, camera_2_house, matches_house[i,0:2], matches_house[i,2:4]).reshape(1,4)
    camera_center1 = compute_camera_center(camera_1_house)
    camera_center1 = camera_center1/camera_center1[-1]
    
    camera_center2 = compute_camera_center(camera_2_house)
    camera_center2 = camera_center2/camera_center2[-1]
    
    fig = plt.figure()
    fig.suptitle('3D reconstruction of the house using OpenCV', fontsize=16)
    ax = fig.add_subplot(111, projection="3d")
#    ax.scatter(camera_center1[0],camera_center1[1],camera_center1[2],c='r',marker='+',label="Camera 1 center")
#    ax.scatter(camera_center2[0],camera_center2[1],camera_center2[2],c='g',marker='+',label="Camera 2 center")
    for i in range(len(points_3d_house)):
        ax.scatter(points_3d_house[i][0],points_3d_house[i][1],points_3d_house[i][2],c='b')
    plt.legend()
    plt.axis('equal')
    plt.xlabel("x axis")
    plt.ylabel("y axis")
#    plt.zlabel("z axis")
    return points_3d_house

#opencv_house = test_house_triangulation(matches_house,camera_1_house,camera_2_house)
#our_house = triangulation_plotting_house(matches_house,camera_1_house,camera_2_house)



    
    
    















