
def plot_and_save_brightfield():
    #Take out the labels that are 256> in column 1
    detections_l2 = detections_l[detections_l[:,1] > 262]

    plt.figure(figsize=(15, 9))
    plt.subplot(2, 1, 1)
    plt.imshow(data[...,0], cmap="gray")
    plt.gca().add_patch(patches.Rectangle((crop_y0, crop_x0), crop_size, crop_size,
                                        linewidth=2, edgecolor="darkorange",
                                        facecolor="none"))
    plt.scatter(detections_l2[:,1], detections_l2[:,0], s = 100, facecolors='none', edgecolors='r')
    #Add a line at the center of the image
    plt.axvline(data.shape[0]//2, color='black', linestyle='--')
    #Shade the region to the right of the line
    plt.axvspan(data.shape[1]//2, data.shape[1]-1, color='lightblue', alpha=0.2)
    plt.xticks([0, data.shape[1]//2, data.shape[1]-1], ["0", "256", "512"])
    plt.yticks([0, data.shape[1]//2, data.shape[1]-1], ["0", "256", "512"])
    plt.title("Quantitative field data (real part)")

    plt.subplot(2, 1, 2)
    plt.imshow(data[...,1], cmap="gray")
    plt.gca().add_patch(patches.Rectangle((crop_y0, crop_x0), crop_size, crop_size,
                                        linewidth=2, edgecolor="darkorange",
                                        facecolor="none"))
    plt.scatter(detections_l2[:,1], detections_l2[:,0], s = 100, facecolors='none', edgecolors='r')
    #Add a line at the center of the image
    plt.axvline(data.shape[0]//2, color='black', linestyle='--')
    #Shade the region to the right of the line
    plt.axvspan(data.shape[1]//2, data.shape[1]-1, color='lightblue', alpha=0.2)
    plt.xticks([0, data.shape[1]//2, data.shape[1]-1], ["0", "256", "512"])
    plt.yticks([0, data.shape[1]//2, data.shape[1]-1], ["0", "256", "512"])
    plt.title("Quantitative field data (imaginary part)")
    plt.savefig("crop.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_and_save_iscat():
    #Take out the labels that are 256> in column 1
    detections_l2 = detections_l[detections_l[:,1]*factor > 262]*factor

    plt.figure(figsize=(8, 8))
    plt.imshow(data[...,0], cmap="gray")

    for crop_x0, crop_y0 in zip(crop_x0s, crop_y0s):
        crop_x0, crop_y0 = int(crop_x0) + 16, int(crop_y0) + 16
        plt.gca().add_patch(patches.Rectangle((crop_y0, crop_x0), crop_size, crop_size,
                                            linewidth=2, edgecolor="darkorange",
                                            facecolor="none"))
        
    plt.scatter(detections_l2[:,1], detections_l2[:,0], s = 150, facecolors='none', edgecolors='r')
    #Add a line at the center of the image
    plt.axvline(data.shape[0]//2, color='black', linestyle='--')
    #Shade the region to the right of the line
    plt.axvspan(data.shape[1]//2, data.shape[1]-1, color='lightblue', alpha=0.2)
    plt.xticks([0, data.shape[1]//2, data.shape[1]-1], ["0", "256", "512"])
    plt.yticks([0, data.shape[1]//2, data.shape[1]-1], ["0", "256", "512"])
    plt.title("Iscat data", fontsize=16)

    plt.savefig("crop.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def nothing():
    if downsample:
        datav2 = skimage.measure.block_reduce(data, (2, 2, 1))
        torch_image = torch.from_numpy(datav2).permute(2, 0, 1).unsqueeze(0).float().to(DEV)
        factor = 2
    else:
        torch_image = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).float().to(DEV)
        factor = 1

    #Make a prediction    
    prediction = lodestar(torch_image)[0].cpu().detach().numpy()
    x, y, rho = prediction[0]*factor, prediction[1]*factor, prediction[-1]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Darkfield image")
    plt.imshow(data[...,0], cmap="gray", vmax=0.3)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("LodeSTAR prediction")
    plt.imshow(rho, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("LodeSTAR prediction with detections")
    plt.imshow(data[...,0], cmap="gray")
    plt.scatter(y.flatten(), x.flatten(), alpha=rho.flatten() / rho.max(), s=5)
    plt.axis("off")

    plt.show()

if __name__ =='__main__':
    pass