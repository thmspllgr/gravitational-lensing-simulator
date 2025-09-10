This is a Python project started as a part of my first year of Physics Master, which is an attempt at simulation of gravitational lensing. It is largely based on [arXiv:astro-ph/0604360](https://arxiv.org/abs/astro-ph/0604360). The idea is to use a gaussian lens model in order to simulate a lensing effect on an image. I aim to improve this project to have more realistic simulations and useful informations. For now, the project handles one lens and can simulate the optic effects of a detector. Next goals are to handle several lenses, and also compute caustic and critical lines in the image.

## Structure

```
src
├── image.py             # Functions for image loading and processing
├── main.py              # Entry point
└── optics.py            # Functions for lens and detector simulation
data
├── data.dat             # Example parameters that can be used for simulations
└── space.jpg            # Sample image for testing
README.md                
LICENCE.md               
```

## Usage

1. Simply run:
    ```
    python src\main.py
    ```

2. Follow the prompt to select an image. The application will then simulate a lensing effect on the image and display the result.

3. You can tweak the parameters inside the `main` function to change the properties of the lens or the caracteristics of the detector.

## Dependencies

- `numpy`
- `matplotlib`
- `PIL`
- `scikit-image`
- `scipy`
- `tkinter`

## Notes

1. Depending on the chosen arguments in the `initializer` call inside main.py you may be asked to select an image stored on your computer to work with. There may be issues with the image loading process that uses the `Tkinter` library (it comes prepackaged with Python on Windows but not Linux or MacOS). If you encounter issues, for simplicity you can replace the first two lines inside the initializer function after `if case == 'image':` with:
    ```
    image_paths = 'C:\\path\\to\\your\\image.png'
    images = [im.imread(image_paths)]
    ```
And also add `import matplotlib.image as im` at the top of the optics.py file for this to work. This way the only needed function from image.py will be `downsample_image`, which you can also disable by putting `sampling=False` inside the main function, and you can work with the same image for each execution (you can of course choose the one provided in the data folder).

2. The JWST documentation used to get realistic parameters for the simulation can found at:
    - for PSF caracteristics: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions
    - for sampling and noise: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-detector-overview

3. This project is licensed under the MIT License - see LICENSE.md for details.