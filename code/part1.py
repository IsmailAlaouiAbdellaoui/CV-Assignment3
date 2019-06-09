from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from scipy.spatial import distance


def fit_fundamental_matrix(matches):
#    print("Fitting of the fundamental Matrix takes place here!!!")
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
    
    #Test
    point1 = np.array([matches[0,0], matches[0,1], 1])
    point2 = np.array([matches[0,2], matches[0,3], 1]).T
    test = np.linalg.multi_dot([point1, fundamental_matrix, point2])
#    print(test)
    
    return fundamental_matrix

matches = np.loadtxt('../data/library/library_matches.txt')


if __name__ == '__main__':

    # load images and match files for the first example

    I1 = Image.open('../data/library/library1.jpg')
    I2 = Image.open('../data/library/library1.jpg')
    matches = np.loadtxt('../data/library/library_matches.txt')

    N = len(matches)
    matches_to_plot = copy.deepcopy(matches)

    '''
    Display two images side-by-side with matches
    this code is to help you visualize the matches, you don't need to use it to produce the results for the assignment
    '''

    I3 = np.zeros((I1.size[1], I1.size[0] * 2, 3))
    I3[:, :I1.size[0], :] = I1
    I3[:, I1.size[0]:, :] = I2
    matches_to_plot[:, 2] += I2.size[0]
    I3 = np.uint8(I3)
    I3 = Image.fromarray(I3)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(I3)
    colors = iter(cm.rainbow(np.linspace(0, 1, matches_to_plot.shape[0])))
    [plt.plot([m[0], m[2]], [m[1], m[3]], color=next(colors)) for m in matches_to_plot]
    plt.show()

    # first, fit fundamental matrix to the matches
    F = fit_fundamental_matrix(matches);  # this is a function that you should write
    '''
    display second image with epipolar lines reprojected from the first image
    '''
    M = np.c_[matches[:, 0:2], np.ones((N, 1))].transpose()
    print(F.shape)
    print(M.shape)
    L1 = np.matmul(F, M).transpose()  # transform points from
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
    L = np.divide(L1, np.kron(np.ones((3, 1)), l).transpose())  # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis=1)
    closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2], np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:, 1], -L[:, 0]] * 10  # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(I2)
    ax.plot(matches[:,2], matches[:, 3], '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]], [matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')
    
    
    #find the mean squared error distance
    error_x = np.array([matches[:, 2] - closest_pt[:, 0]])
    error_y = np.array([matches[:, 3] - closest_pt[:, 1]])
    mean_distance = np.sum(np.sqrt(error_x**2 + error_y**2))/len(error_x[0])
    print("The residual error is :", mean_distance)
