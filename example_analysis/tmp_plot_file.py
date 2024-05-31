
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

if __name__ =='__main__':
    pass