import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from scipy import stats

def plot_correlations(m, P, N):
    
    axis1=0
    axis2=1
    
    sig1 = np.sqrt(P[axis1,axis1])
    sig2 = np.sqrt(P[axis2,axis2])
    
    xmin = m[axis1] - 10*sig1
    xmax = m[axis1] + 10*sig1
    ymin = m[axis2] - 10*sig2
    ymax = m[axis2] + 10*sig2
    
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    
    
    xgrid, ygrid = np.meshgrid(x, y)
    z = stats.multivariate_normal(m, P).pdf(np.dstack((xgrid, ygrid)))
    samples = np.random.default_rng().multivariate_normal(m.flatten(), P, N)
    
    
    plt.figure()
    plt.plot(samples[:,0], samples[:,1], 'k.')
    plt.plot([0, 15], [0, 0], 'k-')
    plt.plot([0, 0], [0, 15], 'k-')
    plt.axis('equal')
    plt.axis('off')
    
    zmax = np.max(z)
    levels = np.logspace(-8,-1,8)*zmax
    # plt.figure()
    plt.contour(x, y, z, levels=8)
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    
    savefig('test.png', transparent=True)
    
    plt.show()
    
    
    return


def plot_kalman_gain():
    
    x = np.arange(0, 20, 0.01)
    
    lam_prior = 6.
    sig_prior = 1.1
    
    lam_meas = 10.
    sig_meas = 3.
    
    lam_post = 6.5
    sig_post = 1.
    
    y_prior = (1./(np.sqrt(2*np.pi)*sig_prior))*np.exp(-0.5*((x-lam_prior)/sig_prior)**2.)
    y_meas = (1./(np.sqrt(2*np.pi)*sig_meas))*np.exp(-0.5*((x-lam_meas)/sig_meas)**2.)
    y_post = (1./(np.sqrt(2*np.pi)*sig_post))*np.exp(-0.5*((x-lam_post)/sig_post)**2.)
    
    
    plt.figure()
    plt.plot(x, y_prior, 'b--', label='Prior')
    plt.plot(x, y_meas, 'r--', label='Meas')
    plt.plot(x, y_post, 'm--', label='Post')
    plt.plot([0, 20], [0, 0], 'k', lw=2)
    plt.plot([0, 0], [0., 0.5], 'k', lw=2)
    plt.axis('off')
    plt.legend()
    
    
    plt.show()
    
    
    return


if __name__ == '__main__':
    
    plt.close('all')
    
    
    N = 2000
    m = np.array([0, 0])
    P0 = np.array([[2, 0], [0, 1]])
    P1 = np.array([[2, -1], [-1, 1]])
    
    
    # plot_correlations(m, P1, N)
    
    plot_kalman_gain()