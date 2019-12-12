import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from pypcd.pypcd import PointCloud
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from provider import *

# k nearest neighbors
knn = 200

# weighting paras
alpha = 0.05  # penalty on the optimization distance between p_star
beta = 0.5   # penalty on the points that far away from p_tilda
lamda = 0.5  # penalty on the points that does not belong to this plane
miu = 0.0001  # for even distribution

# boundary for paras
bnd1 = (-math.inf,math.inf)
bnd2 = (-math.inf,math.inf)
bnd3 = (-math.inf,math.inf)
bnd4 = (0,math.pi*2)
bnd5 = (0,math.pi*2)
bnd = (bnd1,bnd2,bnd3,bnd4,bnd5)

def knnindices(X,K):
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='kd_tree').fit(X)
    # indices.shape = (M,K)
    distances, indices = nbrs.kneighbors(X)

    return distances, indices

def objective1(x, args):
    pnew_x = x[0]
    pnew_y = x[1]
    pnew_z = x[2]
    phi = x[3]
    theta = x[4]

    px = args[0]
    py = args[1]
    pz = args[2]
    i = args[3]
    data = args[4]
    knn = args[5]
    indices = args[6]  # ith row is the ith data's knn
    p = np.array([px, py, pz])
    pnew = np.array([pnew_x, pnew_y, pnew_z])

    nx = (math.cos(phi)) * (math.cos(theta))
    ny = (math.cos(phi)) * (math.sin(theta))
    nz = math.sin(phi)
    n = np.array([nx, ny, nz])

    A = 0.5 * np.exp(alpha * (np.sum((p - pnew)** 2))).item()
    #Xi_knn = findknn(data, knn, i)  # use knn to find the nearest k neighbours
    Xi_knn = data[indices[i, :], :]
    B = np.exp(-beta * np.sum((Xi_knn - p.reshape(1, 3)) ** 2, axis=1))
    D = np.exp(-lamda * (np.dot(Xi_knn - pnew.reshape(1, 3), n) ** 2))
    F = ((Xi_knn - pnew.reshape(1, 3)).dot(n)) ** 2

    #print(A,B,D,F)

    #G = miu * np.exp(np.sum((Xi_knn - pnew.reshape(1, 3)) ** 2, axis=1))
    G = 1.0/(np.sum((Xi_knn - pnew.reshape(1, 3)) ** 2, axis=1)+ 1e-10)
    G = miu/2 * np.sum(G)

    summation = np.multiply(B, D)
    summation = np.multiply(summation, F)
    summation = np.sum(summation)

    #summation2 = np.sum(G)

    #cost =  1/knn * summation2 + A * summation
    cost = A * summation #+ G

    return cost

def single_object_pca_denoising(data):
    # initialization
    phi = 1.57
    theta = 0.5
    N = data.shape[0]
    # record the process
    step = 0

    distance, indices = knnindices(data,knn)

    new_data = np.empty(data.shape)
    for i in range(N):
        p = data[i, :].reshape(3, )
        pnew_x = p[0]
        pnew_y = p[1]
        pnew_z = p[2]

        args = [p[0], p[1], p[2], i, data, knn, indices]
        x0 = [pnew_x, pnew_y, pnew_z, phi, theta]

        cost1 = objective1(x0, args)
        sol = minimize(objective1, x0, method='SLSQP',bounds=bnd, args=args)

        data[i][0] = np.array([sol.x[0].item()])
        data[i][1] = np.array([sol.x[1].item()])
        data[i][2] = np.array([sol.x[2].item()])
        phi = sol.x[3].item()
        theta = sol.x[4].item()


        '''
        new_data[i][0] = np.array([sol.x[0].item()])
        new_data[i][1] = np.array([sol.x[1].item()])
        new_data[i][2] = np.array([sol.x[2].item()])
        '''

        progress = i * 100 / N

        if i % 1000 == 0:
            print("Progress:", progress, "%\n")
        #print("Cost:", cost1, "\n")

    return data

if __name__ == "__main__":
    rawData = pd.read_csv('array.csv').values
    windowData = rawData[:,2:5]
    print(windowData.shape)
    #export_ply(windowData,'window.ply')

    distance, indices = knnindices(windowData, knn)
    mean_distance = np.mean(distance,axis=1).reshape(windowData.shape[0],)

    threshold = np.percentile(mean_distance,98)
    print("threshold is: ",threshold)

    qualified_data = windowData[mean_distance<threshold]
    print(qualified_data.shape)


    plt.hist(mean_distance, bins=1000)
    plt.xlabel('mean distance')
    plt.ylabel('number')
    plt.xlim(0,0.03)
    plt.show()


    #denoised_data = single_object_pca_denoising(qualified_data)
    #export_ply(denoised_data, 'denoise_window.ply')
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    #ax1.scatter(windowData[:, 0], windowData[:, 1], windowData[:, 2], s=1)
    ax1.scatter(denoised_data[:, 0], denoised_data[:, 1], denoised_data[:, 2], s=1)
    plt.show()
    '''
