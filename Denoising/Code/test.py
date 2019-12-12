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
import trimesh     # you may need pip install trimesh
from provider import *


# k nearest neighbors
knn = 20

'''
# weighting paras
alpha = 5.0  # penalty on the optimization distance between p_star
beta = 1   # penalty on the points that far away from p_tilda
lamda = 2  # penalty on the points that does not belong to this plane
miu = 0.01  # for even distribution
'''
# weighting paras
alpha = 8 # penalty on the optimization distance between p_star
beta = 0.1  # penalty on the points that far away from p_tilda
lamda = 0.3  # penalty on the points that does not belong to this plane
miu = 1e-8  # for even distribution
# boundary for paras
bnd1 = (-math.inf,math.inf)
bnd2 = (-math.inf,math.inf)
bnd3 = (-math.inf,math.inf)
bnd4 = (0,math.pi*2)
bnd5 = (0,math.pi*2)
bnd = (bnd1,bnd2,bnd3,bnd4,bnd5)


# no use in our case
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def rearrange(X):
    N, C = X.shape
    for i in range(N):
        p = X[i, :]
        p = np.array([p])
        p = np.transpose(p)
        # initialize parameters
        pnew_x = p[0, :]
        pnew_y = p[1, :]
        pnew_z = p[2, :]
        phi = 0.5
        theta = 0.5
        params = [pnew_x, pnew_y, pnew_z, phi, theta]
    return X

# to be continued
def pca_denoising(batch_data, alpha, beta, lamda):
    """ Conduct PCA Denoising on each group.
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape

    # loop on every object in the file
    for b in range(B):
        '''X: Nx3, represents a single object '''
        X = batch_data[b,:,:].reshape(N,C)
        '''rearrange every point in X'''
        X = rearrange(X)


### KNN function ###
def findknn(X,K,i):
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(X)
    # indices.shape = (M,K)
    distances, indices = nbrs.kneighbors(X)
    Xi_knn = X[indices[i,:],:]
    return Xi_knn

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

    G = 1.0 / (np.sum((Xi_knn - pnew.reshape(1, 3)) ** 2, axis=1) + 1e-10)
    G = miu / 2 * np.sum(G)

    summation = np.multiply(B, D)
    summation = np.multiply(summation, F)
    summation = np.sum(summation)
    #print("s1is:",A * summation)

    cost = A * summation #+ G

    return cost

def objective2(x, args):
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

    # Xi_knn = findknn(data, knn, i)  # use knn to find the nearest k neighbours
    Xi_knn = data[indices[i, :], :]

    F = ((Xi_knn - pnew.reshape(1, 3)).dot(n)) ** 2


    summation = np.sum(F)
    # print("s1is:",A * summation)

    cost = summation  # + G

    return cost

def single_object_pca_denoising(data):
    # initialization
    phi = 1.57
    theta = 0.5
    N = data.shape[0]
    # record the process
    step = 0

    _, indices = knnindices(data,knn)

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

        step = step + 1
        progress = step * 100 / N

        if step % 100 == 0:
            print("Progress:", progress, "%\n")
        #print("Cost:", cost1, "\n")

    return data

# not gonna to use it
def get_data_from_off(filename):
    dictkwargs = {'progress': 'False'}
    mesh = trimesh.load(filename,**dictkwargs)

    data = mesh.vertices
    return data

if __name__ == "__main__":

    #'''
    #this is for modelnet40 dataset
    #------------------------------------------
    h5_filename = 'ply_data_train1.h5'
    data, label = loadDataFile(h5_filename)
    print(data.shape)

    #data1 = data[4,:,:]  # flower pot
    data1 = data[10, :, :]  # change the first number to switch to another object
    print(data1.shape)
    n, m = data1.shape

    reshape_data = data1.reshape(1,data1.shape[0],data1.shape[1])

    #data1 = pd.read_csv("sample_data.csv").values

    noise_data = jitter_point_cloud(reshape_data,0.02, 0.05)
    noise_data = noise_data.reshape(n,m)

    distance, indices = knnindices(noise_data, knn)
    mean_distance = np.mean(distance, axis=1).reshape(n, )

    threshold = np.percentile(mean_distance, 99)
    print("threshold is: ", threshold)

    qualified_data = noise_data[mean_distance < threshold]
    print(qualified_data.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(data1[:, 2], data1[:, 0], data1[:, 1],s=3)
    plt.xlim(-0.6,0.6)
    plt.ylim(-0.6, 0.6)
    ax1.set_zlim3d(-1, 1)
    #ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([0.5, 0.5, 1, 1]))

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(noise_data[:, 2], noise_data[:, 0], noise_data[:, 1], s=3,c='r')
    #ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), np.diag([0.5, 0.5, 2, 1]))
    plt.xlim(-0.6,0.6)
    plt.ylim(-0.6, 0.6)
    ax2.set_zlim3d(-1, 1)


    denoised_data = single_object_pca_denoising(qualified_data)
    #denoised_data = single_object_pca_denoising(data1)
    #fig = plt.figure()
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(denoised_data[:, 2], denoised_data[:, 0], denoised_data[:, 1], s=3,c='g')
    plt.xlim(-0.6,0.6)
    plt.ylim(-0.6, 0.6)
    ax3.set_zlim3d(-1, 1)
    plt.show()
    #-------------------------------------------------------
    #'''

    '''
    this is for pypcd
    '''
    #pc = PointCloud.from_path('window.pcd')