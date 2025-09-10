from optics import initializer, raycaster, simulate_detector
import matplotlib.pyplot as plt


def main():
    # Number of pixels in the x and y directions (must be like 2^n + 1, with n = positive int)
    N = 1025
    M = 1025
    # Lens parameters
    m_p = 25.0
    sigma_c = 1.0
    sigma = 1.0
    x_s = (0, 0)    # (0, 0) is the center of the image
    # Detector parameters
    extent = 10
    psf = False
    noise = True
    sampling = False
    sigma_psf = 1
    sigma_noise = 0.5
    scale_factor = 10

    # Lensing
    source = initializer('image', N, M)     # can change first arg to 'single', 'disc', or 'image' for testing
    image_matrix = raycaster(source, m_p, sigma_c, sigma, x_s, N, M, extent)
    detected_image = simulate_detector(image_matrix, psf, noise, sampling, sigma_psf, sigma_noise, scale_factor)

    # Building graph title string (pure esthetic, only for maniac scientists)
    ops = []
    if psf:
        ops.append(f"PSF σ={sigma_psf}")
    if noise:
        ops.append(f"noise σ={sigma_noise}")
    if sampling:
        ops.append(f"sampling ×{scale_factor}")

    # Display
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

    ax1.imshow(source)
    ax1.set_title("Source plane")

    ax2.imshow(image_matrix)
    ax2.set_title("Image plane")

    ax3.imshow(detected_image)
    if ops:
        ax3.set_title("Detected image with \n" + ", ".join(ops))
    else:
        ax3.set_title("Detected image")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()