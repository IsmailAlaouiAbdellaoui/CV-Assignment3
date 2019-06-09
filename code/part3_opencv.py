import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
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




def test_house_triangulation(matches_house,camera_1_house,camera_2_house):
    points_3d_house = np.zeros((len(matches_house),4),dtype=float)
    for i in range(len(matches_house)):
#        print(matches_house[i,2:4].shape)
        points_3d_house[i] = cv2.triangulatePoints(camera_1_house, camera_2_house, matches_house[i,0:2], matches_house[i,2:4]).reshape(1,4)
        points_3d_house[i] = points_3d_house[i]/points_3d_house[i][-1]
    camera_center1 = compute_camera_center(camera_1_house)
    camera_center1 = camera_center1/camera_center1[-1]
    
    camera_center2 = compute_camera_center(camera_2_house)
    camera_center2 = camera_center2/camera_center2[-1]
    
    fig = plt.figure()
    fig.suptitle('3D Reconstruction of the House using OpenCV', fontsize=16)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(camera_center1[0],camera_center1[1],camera_center1[2],c='r',marker='+',label="Camera 1 center")
    ax.scatter(camera_center2[0],camera_center2[1],camera_center2[2],c='g',marker='+',label="Camera 2 center")
    for i in range(len(points_3d_house)):
        ax.scatter(points_3d_house[i][0],points_3d_house[i][1],points_3d_house[i][2],c='b')
    plt.legend()
    plt.axis('equal')
    plt.xlabel("x axis")
    plt.xlim(-10,5)
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
#    print(camera_center1)
#    print()
    camera_center1 = camera_center1/camera_center1[-1]
    
    camera_center2 = compute_camera_center(camera_2_library)
#    print(camera_center2)
    camera_center2 = camera_center2/camera_center2[-1]
    
    
    fig = plt.figure()
    fig.suptitle('3D reconstruction of the Library using OpenCV', fontsize=16)
#    ax = fig.add_subplot(111, projection="3d")
    ax = fig.gca(projection="3d")
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