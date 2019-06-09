import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy

matches_library = np.loadtxt('../data/library/library_matches.txt')
camera_1_library = np.loadtxt('../data/library/library1_camera.txt')
camera_2_library = np.loadtxt('../data/library/library2_camera.txt')
matches_house = np.loadtxt('../data/house/house_matches.txt')
camera_1_house = np.loadtxt('../data/house/house1_camera.txt')
camera_2_house = np.loadtxt('../data/house/house2_camera.txt')

#This function returns the null space of a matrix (needed for the camera center)
def compute_camera_center(camera_projection):
    return scipy.linalg.null_space(camera_projection)

#This function plots the 3D reconstruction points of the library data points
def triangulation_plotting_library(matches_library,camera1_library,camera2_library):
    points_3d_library = np.zeros((len(matches_library),4),dtype=float)
    #constructing the A matrix as defined in the triangulation slides
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
        
    
        #stacking each row to get a 4 by 4 matrix for each point
        A = np.vstack((A_matrix_row1,A_matrix_row2,A_matrix_row3,A_matrix_row4))
    
        #SVD decomposition of the matrix
        uc1l, sc1l, vc1l = np.linalg.svd(A)
        
        #taking the last element of the vc1 matrix and multiplying by -1
        temp = vc1l[-1]*(-1)
        
        #converting the point into Euclidean space
        point_3d = temp / temp[-1]
        
        
        #Formatting the point in the right shape and adding it to the points_3d_library numpy array
        points_3d_library[i] = point_3d.T.reshape(1,4)
        
    camera_center1 = compute_camera_center(camera_1_library)
    camera_center2 = compute_camera_center(camera_2_library)
    
    #Transforming camera coordinates to Euclidean space
    camera_center1 = camera_center1/camera_center1[-1]
    camera_center2 = camera_center2/camera_center2[-1]


    fig = plt.figure()
    fig.suptitle('3D Reconstruction of the Library', fontsize=16)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(camera_center1[0],camera_center1[1],camera_center1[2],c='r',marker='+',label="Camera 1 center")
    ax.scatter(camera_center2[0],camera_center2[1],camera_center2[2],c='g',marker='+',label="Camera 2 center")
    for i in range(len(points_3d_library)):
        ax.scatter(points_3d_library[i][0],points_3d_library[i][1],points_3d_library[i][2],c='b')
    plt.legend()
    plt.axis('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-15, 15)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    return points_3d_library



#This function plots the 3D reconstruction points of the house data points
def triangulation_plotting_house(matches_house,camera_1_house,camera_2_house):
    points_3d_house = np.zeros((len(matches_house),4),dtype=float)
    #constructing the A matrix as defined in the triangulation slides
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
        
    
        #stacking each row to get a 4 by 4 matrix for each point
        A = np.vstack((A_matrix_row1,A_matrix_row2,A_matrix_row3,A_matrix_row4))
        
        #SVD decomposition of the matrix
        uc1l, sc1l, vc1l = np.linalg.svd(A)

        #taking the last element of the vc1 matrix and multiplying by -1
        temp =vc1l[-1] *  (-1)
        
        #converting the point into Euclidean space
        temp = temp / temp[-1]  

        #Formatting the point in the right shape and adding it to the points_3d_library numpy array
        points_3d_house[i] = temp.T.reshape(1,4)
        
    camera_center1 = compute_camera_center(camera_1_house)
    camera_center2 = compute_camera_center(camera_2_house)
    
    #Transforming camera coordinates to Euclidean space
    camera_center1 = camera_center1/camera_center1[-1]
    camera_center2 = camera_center2/camera_center2[-1]
    
    fig = plt.figure()
    fig.suptitle('3D Reconstruction of the House', fontsize=16)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(camera_center1[0],camera_center1[1],camera_center1[2],c='r',marker='+',label="Camera 1 center")
    ax.scatter(camera_center2[0],camera_center2[1],camera_center2[2],c='g',marker='+',label="Camera 2 center")
    for i in range(len(points_3d_house)):
        ax.scatter(points_3d_house[i][0],points_3d_house[i][1],points_3d_house[i][2],c='b')
    plt.legend()
    plt.axis('equal')
    ax.set_xlim(-5, 2.5)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 0)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    return points_3d_house

  

#triangulation_plotting_library(matches_library,camera_1_library,camera_2_library)

triangulation_plotting_house(matches_house,camera_1_house,camera_2_house)



    
    
    















