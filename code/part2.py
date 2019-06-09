from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cyvlfeat import sift
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
import copy
from part1 import fit_fundamental_matrix 
import random 
import cv2


def find_matching_points(image1, image2, n_levels=3, distance_threshold=300):
    """
    :param image1 and image2 must be RGB images
    :param n_levels: number of scales
    :param distance_threshold: a threshold to accept a given match
    :return: two numpy lists, each with keypoints in [x,y]
    """

    # TODO
    '''
    Important note : you  might need to change the parameters (sift parameters) inside this function to
    have more or better matches
    '''
    matches_1 = []
    matches_2 = []
    image1 = np.array(image1.convert('L'))
    image2 = np.array(image2.convert('L'))
    '''
    Each column of keypoints is a feature frame and has the format [X;Y;S;TH], where X,Y is the (fractional) center of
    the frame, S is the scale and TH is the orientation (in radians).

    AND each column of features is the descriptor of the corresponding frame in F.
    A descriptor is a 128-dimensional vector of class UINT8
    '''
    keypoints_1, features_1 = sift.sift(image1, compute_descriptor=True, n_levels=n_levels)
    keypoints_2, features_2 = sift.sift(image2, compute_descriptor=True, n_levels=n_levels)
    pairwise_dist = cdist(features_1, features_2)  # len(features_1) * len(features_2)
    closest_1_to_2 = np.argmin(pairwise_dist, axis=1)
    for i, idx in enumerate(closest_1_to_2):
        if pairwise_dist[i, idx] <= distance_threshold:
            matches_1.append([keypoints_1[i][1], keypoints_1[i][0]])
            matches_2.append([keypoints_2[idx][1], keypoints_2[idx][0]])
    return np.array(matches_1), np.array(matches_2)


def get_random_points(number_points, matches):
    points = np.zeros((number_points,4),dtype=float)
    for i in range(number_points):
        points[i] = matches[random.randint(0, len(matches)-1)]

    return points

def RANSAC_for_fundamental_matrix(matches):  # this is a function that you should write
    
    #setting the distance threshold and N of iterations
    THRESHOLD = 0.1
    N = 10000
    
    best_distance = 999
    
    test_list = []
    
    for i in range(N):
        
        #get some random feature points >8
        points = get_random_points(10, matches)
        #calculate the fundamental matrix from those random points
        F = fit_fundamental_matrix(points)
        
        #getting fund matrix from cv2 for test purposes
#        F, mask = cv2.findFundamentalMat(points[:,0:2], points[:,2:4], method=cv2.FM_8POINT)
        
        #resetting counters
        inliers = []
        distance_sum = 0
        
        #fit the fundamental matrix to every match
        for i in range(len(matches)):
            
            #get the first and second points in matches and append 1
            points1 = np.append(matches[i, 0:2], 1)
            points2 = np.append(matches[i, 2:4], 1)
            
            #the metric distance. closer to zero, the better
            distance = np.linalg.multi_dot([ points2.T , F, points1])
            
            #keeping only the inliers
            if abs(distance) < THRESHOLD:
                inliers.append(matches[i])
                distance_sum += abs(distance) 
                
        inliers_count = len(inliers)      

        #setting the metric to keep the best set of inliers and best F
        #keep the F and inliers of the minimum  total average distance
        if (inliers_count >30  and distance_sum/inliers_count < best_distance): 
            best_inliers = np.array(inliers)
            best_F = F
            best_distance = distance_sum/inliers_count             
            
    #re-estimate the fundamental matrix using the inliers
    F = fit_fundamental_matrix(best_inliers)    
        
    print("inliers : ", len(best_inliers), "\n" )
    print("best_distance : ", best_distance, "\n" )
#    np.save( "bestin_"+str(len(best_inliers))+ "bestdist_" + str(round(best_distance,5)), np.array(test_list))
    
    return best_F, best_inliers
    


if __name__ == '__main__':
    # load images and match and resize the images
    basewidth = 500
    I1 = Image.open('../data/NotreDame/NotreDame1.jpg')
    wpercent = (basewidth / float(I1.size[0]))
    hsize = int((float(I1.size[1]) * float(wpercent)))
    I1 = I1.resize((basewidth, hsize), Image.ANTIALIAS)

    I2 = Image.open('../data/NotreDame/NotreDame2.jpg')
    wpercent = (basewidth / float(I2.size[0]))
    hsize = int((float(I2.size[1]) * float(wpercent)))
    I2 = I2.resize((basewidth, hsize), Image.ANTIALIAS)

    matchpoints1, matchpoints2 = find_matching_points(I1, I2, n_levels=3, distance_threshold=200)
    matches = np.hstack((matchpoints1, matchpoints2))

    N = len(matches)
    matches_to_plot = copy.deepcopy(matches)

    '''
    Display two images side-by-side with matches
    this code is to help you visualize the matches, you don't need to use it to produce the results for the assignment
    '''
#
#    I3 = np.zeros((I1.size[1], I1.size[0] * 2, 3))
#    I3[:, :I1.size[0], :] = I1;
#    I3[:, I1.size[0]:, :] = I2;
#    matches_to_plot[:, 2] += I2.size[0]  # add to the x-coordinate of second image
#    fig, ax = plt.subplots()
#    ax.set_aspect('equal')
#    ax.imshow(np.array(I3).astype(int))
#    colors = iter(cm.rainbow(np.linspace(0, 1, matches_to_plot.shape[0])))
#
#    [plt.plot([m[0], m[2]], [m[1], m[3]], color=next(colors)) for m in matches_to_plot]
#    plt.show()
#
    # first, find the fundamental matrix to on the unreliable matches using RANSAC
    [F, best_matches] = RANSAC_for_fundamental_matrix(matches)  # this is a function that you should write

    '''
    display second image with epipolar lines reprojected from the first image
    '''
    N = len(best_matches)
    M = np.c_[best_matches[:, 0:2], np.ones((N, 1))].transpose()
    L1 = np.matmul(F, M).transpose()  # transform points from
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
    L = np.divide(L1, np.kron(np.ones((3, 1)), l).transpose())  # rescale the line
    pt_line_dist = np.multiply(L, np.c_[best_matches[:, 2:4], np.ones((N, 1))]).sum(axis=1)
    closest_pt = best_matches[:, 2:4] - np.multiply(L[:, 0:2], np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:, 1], -L[:, 0]] * 10  # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(I2)
    ax.plot(best_matches[:, 2], best_matches[:, 3], '+r')
    ax.plot([best_matches[:, 2], closest_pt[:, 0]], [best_matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')
    plt.show()
    
    
    #find the mean squared error distance
    error_x = np.array([best_matches[:, 2] - closest_pt[:, 0]])
    error_y = np.array([best_matches[:, 3] - closest_pt[:, 1]])
    mean_distance = np.sum(np.sqrt(error_x**2 + error_y**2))/len(error_x[0])
    print("The residual error :", mean_distance)

    ## optional, re-estimate the fundamental matrix using the best matches, similar to part1
    # F = fit_fundamental_matrix(best_matches); # this is a function that you wrote for part1
