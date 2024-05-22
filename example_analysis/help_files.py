import numpy as np
import skimage
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import rvt

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

def get_polarizability_rr(radius, refractive_index):
    """
    Calculate the polarizability of a particle.
    
    Parameters:
    radius (float): Radius of the particle.
    refractive_index (float): Refractive index of the particle.
    
    Returns:
    np.ndarray: Polarizability.
    """
    return np.array(4/3 * np.pi * radius**3 * (refractive_index - 1.333))

def get_polarizability_rr2(radius, refractive_index, nm=1.333):
    """
    Calculate the polarizability of a particle with given medium refractive index.
    
    Parameters:
    radius (float): Radius of the particle.
    refractive_index (float): Refractive index of the particle.
    nm (float): Refractive index of the medium.
    
    Returns:
    float: Polarizability.
    """
    V = 4/3 * np.pi * radius**3
    return 3/2 * V * (refractive_index**2 - nm**2) / (2 * nm**2 + refractive_index**2)

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
    f, _ = scipy.optimize.curve_fit(
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
    # Apply the Radial Variance Transform to the input data (imaginary part)
    input_data_rvt = rvt.rvt(input_data[..., 0], rmin=rmin, rmax=rmax)

    # Fit a 2D Gaussian distribution to the pixel intensity values
    gaussian_values = gaussian_fit(input_data_rvt, binary_gauss=binary_gauss, return_integral=return_integral)

    # Sum the pixel values weighted by the Gaussian values
    if not return_integral:
        return np.abs(np.sum(input_data[..., 0] * gaussian_values))

    return np.sum(gaussian_values) ** (1 / 2)