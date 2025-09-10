from PIL import Image
from skimage import transform
from tkinter import Tk, filedialog
import matplotlib.image as im


def select_image_files():
    """ Prompt the user to select one or several image files. """
    root = Tk()
    root.withdraw()     # hide the root window
    root.attributes('-topmost', True)
    file_paths = filedialog.askopenfilenames(
        title="Select Image Files",
        filetypes=[("Image Files", "*.tif *.png *.jpg *.jpeg *.bmp")]
    )
    root.destroy()
    return file_paths


def load_images(image_paths):
    """ Load images from the specified paths. """
    Image.MAX_IMAGE_PIXELS = 300000001      # to avoid DecompressionBombError
    images = [im.imread(image_path) for image_path in image_paths]
    return images


def downsample_image(image, scale_factor=4):
    """ Downsample the image. """
    return transform.downscale_local_mean(image, (scale_factor,scale_factor,1))