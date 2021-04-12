import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



# 1) Load matrices

te_mat = np.loadtxt("1l1m_ksgTE.txt")
nte_mat = np.loadtxt("1l1m_ksgNTE.txt")

print(te_mat[1, 1])
print(nte_mat[4, 4])


# 2) Visualize matricies

fig, ax = plt.subplots(ncols=2)

divider1 = make_axes_locatable(ax[0])
divider2 = make_axes_locatable(ax[1])

cax1 = divider1.append_axes('right', size='5%', pad=0.05)
cax2 = divider2.append_axes('right', size='5%', pad=0.05)

im1 = ax[0].imshow(te_mat, cmap='Greys')
im2 = ax[1].imshow(te_mat, cmap='Greys')

fig.colorbar(im1, cax=cax1, orientation='vertical')
fig.colorbar(im2, cax=cax2, orientation='vertical')

ax[0].set_title("KSG Transfer Entropy")
ax[1].set_title("KSG Norm Transfer Entropy")

fig.savefig("MatrixPlots.pdf")
