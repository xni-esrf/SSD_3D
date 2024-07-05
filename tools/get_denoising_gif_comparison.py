import PIL.Image
import tifffile
import numpy as np
import os
from argparse import ArgumentParser

import PIL.ImageOps


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)



def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)



def parse_arguments():
    """
     Parses the users command line input
     Returns: NameSpace object containing all needed user inputs
     """
    parse = ArgumentParser(description="Provided two folders got using a denoising script, save gif comparing de obtained denoised data")
    parse.add_argument('denoised_dir1', help="The directory containing the first denoised volume")
    parse.add_argument('denoised_dir2', help='The directory containing the second denoised volume')
    parse.add_argument('out_dir', help='The directory where gif are saved')
    parse.add_argument('--no_ortho_compare', default=False, action='store_const', const=True,  help="Get only gif comparison for one view")
    parse.add_argument('--index_to_compare', default=None, type=int, help="The index of the tiff images to compare in each view. If none, compare middle images")
    parse.add_argument('--image_mark', default=0,  type=int, help="Size of the mark added at the upper left corner of the first gif image to recognize it")

    return parse.parse_args()

# parse arguments
params = parse_arguments()
denoised_dir1 = params.denoised_dir1
denoised_dir2 = params.denoised_dir2
setting1_name = os.path.normpath(denoised_dir1).split('/')[-1]
setting2_name = os.path.normpath(denoised_dir2).split('/')[-1]
out_dir = params.out_dir
image_mark = params.image_mark

no_ortho_compare = params.no_ortho_compare
if no_ortho_compare :
    nb_views = 1
else:
    nb_views = 3

index_to_compare = params.index_to_compare
if index_to_compare == None:
    index_to_compare = len(os.listdir(os.path.join(denoised_dir1, "0")))//2

if os.path.isdir(out_dir) == False:
    os.mkdir(out_dir)


for num_view in range(nb_views):
    # Get full path to the images to be compared
    path1 = os.path.join(denoised_dir1, str(num_view), "output_{}.tif".format(str(index_to_compare).zfill(5)))
    #path1 = os.path.join(denoised_dir1, str(num_view),"output_{}.tif".format(str(index_to_compare+2).zfill(5)))
    path2 = os.path.join(denoised_dir2, str(num_view), "output_{}.tif".format(str(index_to_compare).zfill(5)))
    #path2 = os.path.join(denoised_dir2, str(num_view),"output_{}.tif".format(str(index_to_compare+2).zfill(5)))

    # Load and normalize images to compare
    img1 = tifffile.imread(str(path1))
    #img1 = img1[2:img1.shape[0]-2,2:img1.shape[0]-2]
    img2 = tifffile.imread(str(path2))
    img1 = (img1 - img1.min())/(img1.max()-img1.min())*255
    img2 = (img2 - img2.min())/(img2.max()-img2.min())*255
    #img2 = img2[2:img2.shape[0]-2,2:img2.shape[0]-2]

    img2 = hist_match(img2,img1)

    #img1 = image_histogram_equalization(img1)
    #img2 = image_histogram_equalization(img2)

    # Mark some pixel to recognize the first image in the saved gif
    if image_mark != 0:
        img1[0:image_mark,0:image_mark]=0

    #TODO automatic contrast adaptation

    # Save compared images as gif
    #imgs_tosave = [PIL.Image.fromarray(img1), PIL.Image.fromarray(img2)]
    imgs_tosave = [PIL.ImageOps.autocontrast(PIL.Image.fromarray(img1).convert('L')), PIL.ImageOps.autocontrast(PIL.Image.fromarray(img2).convert('L'))] 
    output_path = os.path.join(out_dir, "{}_vs_{}_view{}.gif".format(setting1_name, setting2_name, num_view))
    imgs_tosave[0].save(output_path, save_all=True, append_images=[imgs_tosave[1]], duration=1000, loop=0)
    