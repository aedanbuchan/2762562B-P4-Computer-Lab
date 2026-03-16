import matplotlib.pyplot as plt

def visualise_params(raw, decomped, num, angle_map = None, amp_map = None, angle_amp = None):

    cmap1 = "twilight"
    cmap2 = "inferno"
    angle_or_amp = [0,1,0,1,0]
    
    if angle_map:
        cmap1 = angle_map

    if amp_map:
        cmap2 = amp_map

    if angle_amp:
        angle_or_amp = angle_amp
    maps = []
    
    for k in range(num):
        if angle_or_amp[k] == 0:
            maps.append(cmap2)
        else:
            maps.append(cmap1)
        
    fig, axes = plt.subplots(num, 2,figsize=(10,20))

    for n in range(num):

    # Raw
        im0 = axes[n, 0].imshow(raw[:, :, n],cmap=maps[n])
        axes[n, 0].set_title(f'Raw Parameter {n}')
        fig.colorbar(im0, ax=axes[n, 0])

    # Decomposed
        im1 = axes[n, 1].imshow(decomped[:, :, n],cmap=maps[n])
        axes[n, 1].set_title(f'Decomposed Parameter {n}')
        fig.colorbar(im1, ax=axes[n, 1])

def visualise_residuals(raw,decomped,num):
    fig, axes = plt.subplots(num, 1,figsize=(10,20))

    for n in range(num):
    # Residuals
        im0 = axes[n].imshow(raw[:, :, n]-decomped[:, :, n],cmap='magma')
        axes[n].set_title(f'Residuals Parameter {n}')
        fig.colorbar(im0, ax=axes[n])
    
    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()
