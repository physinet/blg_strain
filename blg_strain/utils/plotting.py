import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_bands_3d(Kx, Ky, M, which=[0,1,2,3], cmap='bwr', **kwargs):
    '''
    Makes a 3d plot of the values of M in each of the four bands.

    which: which bands to plot
    kwargs passed to Axes3D.plot_surface
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for n in which:
        ax.plot_surface(Kx, Ky, M[n], **kwargs)

    return fig, ax


def plot_bands(Kx, Ky, M, contour=True, cmap='bwr'):
    '''
    Plots an 4 x Nkx x Nky matrix M as a colorplot vs Kx and Ky for each value of
    the first index.

    This will plot over the 4 bands generated by a 4x4 Hamiltonian.
    '''
    fig, ax = plt.subplots(1, 4, figsize=(10, 2.5))

    for n in range(4):
        a = ax[n]
        a.pcolormesh(Kx, Ky, M[n], cmap=cmap)
        if contour:
            a.contour(Kx, Ky, M[n], colors='k', linewidths=0.5,
                linestyles='solid')

        a.set_xticks([])
        a.set_yticks([])
        a.set_title('Band %i' %n)

    return fig, ax


def plot_bands_KKprime(Kx, Ky, M, M1, contour=True, cmap='bwr'):
    '''
    Plots colorplots for the same quantity over 4 bands and at K and K'
    '''
    fig, ax = plt.subplots(2, 4, figsize=(10, 5))

    for n in range(4):
        a = ax[0][n]
        a1 = ax[1][n]
        a.pcolormesh(Kx, Ky, M[n], cmap=cmap)
        a1.pcolormesh(Kx, Ky, M1[n], cmap=cmap)
        if contour:
            a.contour(Kx, Ky, M[n], colors='k', linewidths=0.5,
                linestyles='solid')
            a1.contour(Kx, Ky, M1[n], colors='k', linewidths=0.5,
                linestyles='solid')

        a.set_xticks([])
        a.set_yticks([])
        a1.set_xticks([])
        a1.set_yticks([])
        a.set_title('Band %i' %n)

    ax[0,0].set_ylabel('$K$', rotation=0, labelpad=30, fontsize=16, va='center')
    ax[1,0].set_ylabel('$K\'$', rotation=0, labelpad=30, fontsize=16, va='center')

    return fig, ax
