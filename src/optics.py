from image import select_image_files, load_images, downsample_image
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import numpy as np


def initializer(case, N, M):
    """
    Function that initializes the source plane matrix, for which one can select the initial condition.
    The source plane matrix is an array of size (N, M, 3) for RGB images or (N, M) for grayscale images.

    Parameters
    ----------
    case : str
        Type of source configuration ('single', 'disc', 'image').
    N : int
        Number of pixels in the x direction.
    M : int
        Number of pixels in the y direction.

    Returns
    -------
    Source_matrix : np.ndarray
        3D array of size (N, M, 3) for RGB or 2D for grayscale.
    """

    Source_matrix = np.zeros((N, M, 3))
    
    # Case of a single source pixel at the center of the source plane
    if case == 'single':
        Source_matrix[int(N / 2), int(M / 2), :] = 1.0

    # Case of a disc source
    if case == 'disc':
        for i in range(N):
            for j in range(M):
                r = np.sqrt((i - N / 2) ** 2 + (j - M / 2) ** 2)
                if r < 5:
                    Source_matrix[i, j, :] = 1.0
    
    # Case of an image
    if case == 'image':
        image_paths = select_image_files()
        images = load_images(image_paths)
        if len(images) > 0:
            image = images[0]
            if image.shape[2] == 4:
                image = image[:, :, :3]  # discard the alpha channel
            image_resized = resize(image, (N, M, image.shape[2]), anti_aliasing=True)
            Source_matrix = image_resized
        else:
            raise ValueError("No images were loaded.")

    return Source_matrix


def raycaster(Source_matrix, m_p, sigma_c, sigma, x_s, N, M, extent):
    """
    Function that computes the deflection of the rays from the source plane to the image plane.

    Parameters
    ----------
    Source_matrix : np.ndarray
        3D array of size (N, M, 3) for RGB or 2D for grayscale.
    m_p : float
        Mass of the lens.
    sigma_c : float
        Critical surface mass density.
    sigma : float
        Standard deviation of the gaussian profile.
    x_s : tuple
        Position of the lens in the source plane (x_s[0], x_s[1]).
    N : int
        Number of pixels in the x direction.
    M : int
        Number of pixels in the y direction.
    extent : float
        Extent of the image plane in the x and y directions. (= the "physical size")
    
    Returns
    -------
    Image_matrix : np.ndarray
        3D array of size (N, M, 3) for RGB or 2D for grayscale.
    """

    # Initialization of the image plane matrix
    if Source_matrix.ndim == 3:                     # RGB image
        Image_matrix = np.zeros((N, M, 3))
    else:                                           # grayscale image
        Image_matrix = np.zeros((N, M))

    x = np.linspace(-extent, extent, num=N)
    y = np.linspace(-extent, extent, num=M)         
    delta_x = x[1] - x[0]                           # step size for x
    delta_y = y[1] - y[0]                           # step size for y
    
    # Loop over the source plane matrix to compute the deflection of the rays
    for i in range(N):
        for j in range(M):
            r = np.sqrt((x[i] - x_s[0]) ** 2 + (y[j] - x_s[1]) ** 2) + 1e-6
            A = m_p / (np.pi * sigma_c * r ** 2) * (-1 + np.exp(-r ** 2 / (2 * sigma ** 2)))
            alpha_x = -A * (x[i] - x_s[0])
            alpha_y = -A * (y[j] - x_s[1])
            b_x = x[i] - alpha_x
            b_y = y[j] - alpha_y
            b_x_pixel = int((b_x - np.min(x)) / (delta_x))   # pixel assignment for x
            b_y_pixel = int((b_y - np.min(y)) / (delta_y))   # pixel assignment for y

            if 0 <= b_x_pixel < N and 0 <= b_y_pixel < M:
                if Source_matrix.ndim == 3:         # RGB image
                    Image_matrix[i, j, :] = Source_matrix[b_x_pixel, b_y_pixel, :]
                else:                               # grayscale image
                    Image_matrix[i, j] = Source_matrix[b_x_pixel, b_y_pixel]

    return Image_matrix


def add_noise(image, sigma_noise=0.1):
    """
    Function that adds a random gaussian noise to the image.

    Parameters
    ----------
    image : np.ndarray
        3D array of size (N, M, 3) for RGB or 2D for grayscale.
    sigma_noise : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    noisy_image : np.ndarray
        3D array of size (N, M, 3) for RGB or 2D for grayscale.
    """
    
    noise = np.random.normal(0, sigma_noise, image.shape)
    noisy_image = image + noise
    return noisy_image


def simulate_detector(image, psf=True, noise=True, sampling=True, sigma_psf=1, sigma_noise=0.1, scale_factor=4):
    """
    Function that simulates detector's effects by:
    1) Convolving the image with a gaussian PSF
    2) Adding gaussian noise
    3) Downsampling the image

    The order of operations is important:
    - Downsampling should be done after adding noise to avoid aliasing
    - Convolution with PSF should be done before adding noise to avoid noise amplification


    Parameters
    ----------
    image : np.ndarray
        3D array of size (N, M, 3) for RGB or 2D for grayscale.
    psf : bool
        If True, convolve the image with a gaussian PSF.
    noise : bool
        If True, add noise to the image.
    sampling : bool
        If True, downsample the image.
    sigma_psf : float
        Standard deviation of the gaussian PSF.
    sigma_noise : float
        Standard deviation of the gaussian noise.
    scale_factor : int
        Factor by which to downsample the image.

    Returns
    -------
    simulated_image : np.ndarray
        3D array of size (N, M, 3) for RGB or 2D for grayscale.
    """
    
    if psf:
        simulated_image = gaussian_filter(image, sigma_psf)
    else:
        simulated_image = image

    if noise:
        simulated_image = add_noise(simulated_image, sigma_noise)

    if sampling:
        simulated_image = downsample_image(simulated_image, scale_factor)

    return simulated_image


def magnification_map(m_p, sigma_c, sigma, x_s, N, M, extent):
    x = np.linspace(-extent, extent, num=N)
    y = np.linspace(-extent, extent, num=M)
    mu_critical_inv = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            r = np.sqrt((x[i] - x_s[0])**2 + (y[j] - x_s[1])**2)
            Kappa = m_p / (sigma_c * 2 * np.pi * sigma**2) * np.exp(-r**2 / (2 * sigma**2))
            Gamma = r**2 + 2 * sigma**2 * (1 - np.exp(-r**2 / (2 * sigma**2)))
            mu_critical_inv[i, j] = (1 - Kappa)**2 - Kappa**2 * Gamma**2
    return mu_critical_inv