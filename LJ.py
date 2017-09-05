import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time


def doubleLJ(x, *params):
    """
    Calculates total energy and gradient of N atoms interacting with a
    double Lennard-Johnes potential.
    
    Input:
    x: positions of atoms in form x= [x1,y2,x2,y2,...]
    params: parameters for the Lennard-Johnes potential

    Output:
    E: Total energy
    dE: gradient of total energy
    """
    eps, r0, sigma = params
    N = x.shape[0]*x.shape[1]
    
    Natoms = np.size(x, 0)

    E = 0
    dE = np.zeros(N)
    for i in range(Natoms):
        for j in range(Natoms):
            r = np.sqrt(sp.dot(x[i] - x[j], x[i] - x[j]))
            print(i, j, r)
            if j > i:
                E1 = 1/r**12 - 2/r**6
                E2 = -eps * np.exp(-(r - r0)**2 / (2*sigma**2))
                E += E1 + E2
            if j != i:
                dxij = x[i, 0] - x[j, 0]
                dyij = x[i, 1] - x[j, 1]

                dEx1 = 12*dxij*(-1 / r**14 + 1 / r**8)
                dEx2 = eps*(r-r0)*dxij / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))

                dEy1 = 12*dyij*(-1 / r**14 + 1 / r**8)
                dEy2 = eps*(r-r0)*dyij / (r*sigma**2) * np.exp(-(r - r0)**2 / (2*sigma**2))

                dE[2*i] += dEx1 + dEx2
                dE[2*i + 1] += dEy1 + dEy2
    return E, -dE


if __name__ == "__main__":

    x1 = np.array([0, 0, 0.9, 1.1])
    x2 = np.array([0, 1, 1.2, -0.5])
    X = np.c_[x1, x2]
    print(X)
    
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
    E, F = doubleLJ(X, eps, r0, sigma)
    print(F)
    F = F.reshape((4, 2))
    




    
    plt.scatter(x1, x2)
    plt.show()

    '''
    r = np.linspace(0.8, 2.5, 100)
    x1 = np.array([0, 0])
    x2 = np.c_[r, np.zeros(100)]
    eps, r0, sigma = 1.8, 1.1, np.sqrt(0.02)
                                                                                                                    
    E = np.zeros(100)
    Fx = np.zeros(100)
    for i in range(100):
        X = np.array([x1, x2[i, :]])
        print(X)
        E[i], F = doubleLJ(X, eps, r0, sigma)
        Fx[i] = F[0]

    plt.plot(r, E)
    plt.plot(r, Fx)
    plt.xlim([0.9, 2.5])
    plt.ylim([-10, 10])
    plt.show()
    '''
