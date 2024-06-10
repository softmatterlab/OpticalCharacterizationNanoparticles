import numpy as np
import skimage
import matplotlib.pyplot as plt

def plot_frame_with_detections(data, positions, s=100, title='Frame', figsize=(6, 6), cmap='gray', save_path=None):
    """
    Plot the frame with the detected particles.
    
    Parameters:
    data (np.ndarray): Input image data.
    positions (np.ndarray): Detected particle positions.
    s (int): Radius of the circle.
    title (str): Title of the plot.
    figsize (tuple): Figure size.
    cmap (str): Colormap of the image.
    save_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(data, cmap=cmap)
    ax.scatter(positions[:, 1], positions[:, 0], s=s, facecolors='none', edgecolors='r')
    ax.set_title(title)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_frame_with_detections_filled(data, positions, values, s=100, title='Frame', figsize=(6, 6), cmap='gray', alpha = 0.75, save_path=None):
    """
    Plot the frame with the detected particles.
    
    Parameters:
    data (np.ndarray): Input image data.
    positions (np.ndarray): Detected particle positions.
    values (np.ndarray): Values associated to the positions.
    s (int): Radius of the circle.
    title (str): Title of the plot.
    figsize (tuple): Figure size.
    cmap (str): Colormap of the image.
    save_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(data, cmap=cmap)
    ax.scatter(positions[:, 1], positions[:, 0], s=s, c=values, facecolors='none', edgecolors='r', alpha=alpha)
    ax.set_title(title)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_labels(labels, bins = 20, figsize = (12,3), color = 'darkblue'):
    """
    Plot labels as histograms.

    Parameters:
    labels (np.ndarray): Labels: (pos_x, pos_y, pos_z, radius, ri)
    bins (int) : Number of bins in histogram.
    figsize (tuple): Figure size.
    color (str): Color of histograms.
    """

    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.hist(labels[:,2], bins=bins, color=color)
    plt.title("Z-position (in pixels)")
    plt.subplot(132)
    plt.hist(labels[:,3]*1e9, bins=bins, color=color, range=(75, 225))
    plt.title("Radius(nm)")
    plt.subplot(133)
    plt.hist(labels[:,4], bins=bins, color=color, range=(1.35, 1.62))
    plt.title("Refractive index")
    plt.show()


def rvt_pipeline(data, rmin=4, rmax=25, th_scale=0.3, min_distance=7, return_detection_map=False):
    """
    Apply the Radial Variance Transform to the image and return the detections.
    
    Parameters:
    data (np.ndarray): Input image data.
    rmin (int): Minimum radius for the transform.
    rmax (int): Maximum radius for the transform.
    th_scale (float): Threshold scale for detections.
    min_distance (int): Minimum distance between peaks.
    return_detection_map (bool): If True, return the detection map.
    
    Returns:
    np.ndarray: Detected points.
    np.ndarray (optional): Detection map if return_detection_map is True.
    """

    import rvt

    # Apply Radial Variance Transform of the image
    det = rvt.rvt(data[..., 0], rmin=rmin, rmax=rmax)

    # Local max of the transformed image
    blobs = skimage.feature.peak_local_max(det, min_distance=min_distance)
    blobs = np.stack([blobs[:, 0], blobs[:, 1]], axis=-1)

    # Extract detections that have a high enough value
    detections = []
    th = np.mean(det) * th_scale
    for blob in blobs:
        if np.mean(det[blob[0]-2:blob[0]+2, blob[1]-2:blob[1]+2]) > th:
            detections.append(blob)

    # Convert to numpy array and rename
    detections_rvt = np.array(detections)

    if return_detection_map:
        return detections_rvt, det
    else:
        return detections_rvt

def add_bin_circles(positions, radius, image):
    """
    Add circles to an image.
    
    Parameters:
    positions (np.ndarray): Array of positions to add circles.
    radius (int): Radius of the circles.
    image (np.ndarray): Input image to modify.
    
    Returns:
    np.ndarray: Image with circles added.
    """
    im = image.copy()
    
    if len(positions) == 0:
        return im
    
    for position in positions:
        rr, cc = skimage.draw.disk(position, radius)
        im[rr, cc] = 1
    return im

def get_rois(data, positions, padsize):
    """
    Retrieve ROIs from data.
    
    Parameters:
    data (np.ndarray): Input image data.
    positions (np.ndarray): Positions of the ROIs.
    padsize (int): Padding size for the ROIs.
    
    Returns:
    np.ndarray: Extracted ROIs.
    """
    rois = []
    for pos in positions:
        # Check if the ROI is out of bounds
        if pos[1]-padsize < 0 or pos[1]+padsize >= data.shape[0] or pos[0]-padsize < 0 or pos[0]+padsize >= data.shape[1]:
            continue

        roi = data[int(pos[0]-padsize):int(pos[0]+padsize), int(pos[1]-padsize):int(pos[1]+padsize), :]
        rois.append(roi)
    
    return np.stack(rois)

def get_F1_score(Pred_Particles, GT_particles):
    """
    Get the F1 score between the predicted particles and the ground truth.
    
    Parameters:
    Pred_Particles (np.ndarray): Predicted particles mask.
    GT_particles (np.ndarray): Ground truth particles mask.
    
    Returns:
    float: F1 score.
    """
    TP = np.sum(Pred_Particles * GT_particles)  # True positives
    FP = np.sum(Pred_Particles) - TP            # False positives
    FN = np.sum(GT_particles) - TP              # False negatives

    F1 = 2 * TP / (2 * TP + FP + FN)
    return F1

def get_polarizability_rr(radius, refractive_index, refractive_index_medium=1.333):
    """
    Calculate the polarizability of a particle.
    Equation: polarizability = Volume * Refractive Index Difference
    
    Parameters:
    radius (float): Radius of the particle.
    refractive_index (float): Refractive index of the particle.
    refractive_index_medium (float): Refractive index of the medium.
    
    Returns:
    np.ndarray: Polarizability.
    """
    return np.array(4/3 * np.pi * radius**3 * (refractive_index - refractive_index_medium))

def get_polarizability(radius, refractive_index, refractive_index_medium=1.333):
    """
    Calculate the polarizability of a particle with given medium refractive index.
    
    Parameters:
    radius (float): Radius of the particle.
    refractive_index (float): Refractive index of the particle.
    refractive_index_medium (float): Refractive index of the medium.
    
    Returns:
    float: Polarizability.
    """
    V = 4/3 * np.pi * radius**3
    return 3/2 * V * (refractive_index**2 - refractive_index_medium**2) / (2 * refractive_index_medium**2 + refractive_index**2)

def form_factor(radius, nm=1.333, wavelength=532):
    """
    Calculate the spherical form factor in nanometers.
    
    Parameters:
    radius (float): Radius of the particle.
    nm (float): Refractive index of the medium.
    wavelength (float): Wavelength in nanometers.
    
    Returns:
    float: Form factor.
    """
    k = (2 * np.pi * nm) / wavelength 
    q = 2 * k * np.sin(np.pi / 2) 
    return 3 / (q * radius)**3 * (np.sin(q * radius) - q * radius * np.cos(q * radius))

def signal_iscat(form_factor, polarizability):
    """
    Calculate the signal in iSCAT.
    
    Parameters:
    form_factor (float): Form factor.
    polarizability (float): Polarizability.
    
    Returns:
    float: Signal in iSCAT.
    """
    return np.abs(form_factor) * polarizability

def pol_range(rad_range, ri_range, w=0.532, nm=1.33):
    """
    Calculates the range of polarizabilities given arrays of possible radius and refractive indeces.

    Parameters:
    rad_range (np.ndarray): Array of radius values.
    ri_range (np.ndarray): Array of refractive index values.
    w: (float): Wavelength in mikrometer.
    nm: (float): Refractive index.

    Returns:
    tuple: min and max values of possible values
    
    """
    dm = lambda rad, ri: get_polarizability(rad*1e6, ri, nm)

    vals = [
        dm(rad_range[0], ri_range[0]), 
        dm(rad_range[1], ri_range[0]), 
        dm(rad_range[0], ri_range[1]), 
        dm(rad_range[1], ri_range[1])
        ]

    return min(vals), max(vals)

def pol_range_mean_std(rad_range, ri_range, w=0.532, nm=1.33):

    """
    Calculates the mean and standard deviation of polarizabilities given arrays of possible radius and refractive indeces.

    Parameters:
    rad_range (np.ndarray): Array of radius values. Assume it is in nanometers
    ri_range (np.ndarray): Array of refractive index values.
    w: (float): Wavelength in mikrometer.
    nm: (float): Refractive index.

    Returns:
    tuple: mean and standard deviation of possible values
    
    """
    dm = lambda rad, ri: get_polarizability(rad*1e6, ri, nm)
   

    rad = np.linspace(rad_range[0], rad_range[1], 1000)
    ri = np.linspace(ri_range[0], ri_range[1], 1000)

    vals = [dm(r, i) for r in rad for i in ri]

    return np.mean(vals), np.std(vals)

def signal_range(rad_range, ri_range, w=0.532, nm=1.33):
    """
    Calculates the range of signals given arrays of possible radius and refractive indeces.

    Parameters:
    rad_range (np.ndarray): Array of radius values. Assume it is in nanometers
    ri_range (np.ndarray): Array of refractive index values.
    w: (float): Wavelength in mikrometer.
    nm: (float): Refractive index.

    Returns:
    tuple: min and max values of possible values
    
    """
    dm = lambda rad, ri: get_polarizability(rad*1e6, ri, nm)

    form_f = lambda rad: form_factor(rad*1e6, nm=nm, wavelength=w)

    vals = [
        dm(rad_range[0], ri_range[0]) * np.abs(form_f(rad_range[0])),
        dm(rad_range[0], ri_range[0]) * np.abs(form_f(rad_range[1])),

        dm(rad_range[1], ri_range[0]) * np.abs(form_f(rad_range[0])),
        dm(rad_range[1], ri_range[0]) * np.abs(form_f(rad_range[1])),

        dm(rad_range[0], ri_range[1]) * np.abs(form_f(rad_range[0])),
        dm(rad_range[0], ri_range[1]) * np.abs(form_f(rad_range[1])),

        dm(rad_range[1], ri_range[1]) * np.abs(form_f(rad_range[0])),
        dm(rad_range[1], ri_range[1]) * np.abs(form_f(rad_range[1]))
        ]

    return min(vals), max(vals)

def signal_range_mean_std(rad_range, ri_range, w=0.532, nm=1.33):
    """
    Calculates the mean and standard deviation of signals given arrays of possible radius and refractive indeces.

    Parameters:
    rad_range (np.ndarray): Array of radius values. Assume it is in nanometers
    ri_range (np.ndarray): Array of refractive index values.
    w: (float): Wavelength in mikrometer.
    nm: (float): Refractive index.

    Returns:
    tuple: mean and standard deviation of possible values
    
    """
    dm = lambda rad, ri: get_polarizability(rad*1e6, ri, nm)

    rad = np.linspace(rad_range[0], rad_range[1], 1000)
    ri = np.linspace(ri_range[0], ri_range[1], 1000)

    vals = [dm(r, i) * np.abs(form_factor(r*1e6, nm=nm, wavelength=w)) for r in rad for i in ri]

    return np.mean(vals), np.std(vals)

def gaussian_fit(input_data, upscale=1, binary_gauss=False, return_integral=False):
    """
    Fit a 2D Gaussian to the input data and return the Gaussian values.
    
    Parameters:
    input_data (np.ndarray): Input data to fit.
    upscale (float): Upscale factor for input data.
    binary_gauss (bool): If True, binarize the Gaussian values.
    return_integral (bool): If True, return the integral of the Gaussian values.
    
    Returns:
    np.ndarray or float: Gaussian values or integral of Gaussian.
    """
    from scipy import optimize

    height, width = input_data.shape

    # Upscale the input data
    input_data = input_data * upscale

    # Create meshgrid for the image coordinates
    x, y = np.meshgrid(np.arange(0, width) - width / 2, np.arange(0, height) - height / 2)
    data = np.column_stack((x.flatten(), y.flatten()))

    # Define 2D Gaussian function
    def fitf(x, a, b, c, d, e):
        return a * np.exp(-((x[:, 0] - b)**2 + (x[:, 1] - c)**2) / (2 * d**2)) + e

    # Fit the Gaussian to the data
    f, _ = optimize.curve_fit(
        fitf,
        data,
        input_data.flatten() - np.mean(input_data.flatten()),
        p0=[1 * upscale, 0, 0, 1, 0],
        bounds=((-1 * upscale, -5, -5, 0.1, -1), (1 * upscale, 5, 5, 100, 1))
    )

    # Calculate the Gaussian values
    gaussian_values = f[0][0] * np.exp(-((x - f[0][1])**2 + (y - f[0][2])**2) / (2 * f[0][3]**2)) + f[0][4]

    # Binarize the Gaussian values
    if binary_gauss:
        gaussian_values = gaussian_values > np.quantile(gaussian_values, 0.95)

    # Return the integral of the Gaussian values
    if return_integral:
        return np.abs(f[0][0] * f[0][3]**2 * np.sqrt(2 * np.pi)) / upscale

    return gaussian_values

def radial_variance_gaussian(input_data, rmin=3, rmax=20, binary_gauss=False, return_integral=False):
    """
    Creates a score map with Radial Variance Transform and fits a 2D Gaussian to the data.
    
    Parameters:
    input_data (np.ndarray): Input data.
    rmin (int): Minimum radius for the transform.
    rmax (int): Maximum radius for the transform.
    binary_gauss (bool): If True, binarize the Gaussian values.
    return_integral (bool): If True, return the integral of the Gaussian.
    
    Returns:
    float or np.ndarray: Weighted sum of the pixel values or the integral of the Gaussian.
    """
    import rvt

    # Apply the Radial Variance Transform to the input data (imaginary part)
    input_data_rvt = rvt.rvt(input_data[..., 0], rmin=rmin, rmax=rmax)

    # Fit a 2D Gaussian distribution to the pixel intensity values
    gaussian_values = gaussian_fit(input_data_rvt, binary_gauss=binary_gauss, return_integral=return_integral)

    # Sum the pixel values weighted by the Gaussian values
    if not return_integral:
        return np.abs(np.sum(input_data[..., 0] * gaussian_values))

    return np.sum(gaussian_values) ** (1 / 2)