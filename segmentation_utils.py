import numpy as np 
from skimage.segmentation import watershed
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
from math import floor
from pqdm.processes import pqdm
import multiprocessing as mp
import os
import time
import cc3d
import pandas as pd
from tqdm import tqdm
import tifffile as tiff


from scipy.signal import find_peaks, argrelmin

def save_3d_array_as_tiff(array, path):

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {array.shape}")
    
    tiff.imwrite(
        path,
        array.astype(np.uint8),  # or uint8 depending on your data
        photometric='minisblack',  # ensures grayscale appearance
        compression='zlib'         # improves compatibility
    )

def list_tif_npy_files(folder_path):
    """
    Lists all .tif/.tiff or .npy files in the given folder.

    If both .tif/.tiff and .npy files are found, raises a ValueError.

    Parameters:
    - folder_path (str): Path to a directory.

    Returns:
    - List[str]: Sorted list of matching file paths.

    Raises:
    - NotADirectoryError: If the path is not a directory.
    - ValueError: If both .tif/.tiff and .npy files are found.
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Provided path is not a directory: {folder_path}")

    tif_files = []
    npy_files = []

    for fname in os.listdir(folder_path):
        full_path = os.path.join(folder_path, fname)
        if os.path.isfile(full_path):
            _, ext = os.path.splitext(fname)
            ext = ext.lower()
            if ext in [".tif", ".tiff"]:
                tif_files.append(full_path)
            elif ext == ".npy":
                npy_files.append(full_path)

    if tif_files and npy_files:
        raise ValueError("Folder contains both .tif/.tiff and .npy files, which is not allowed. It is not clear which files to segment. Remove files without your preferred extension.")

    return sorted(tif_files + npy_files)

def read_tif_stack(path):
    """
    Reads a 3D .tif stack into a 3D NumPy array.

    Parameters:
    - path (str): Path to the .tif file.

    Returns:
    - np.ndarray: 3D array (depth, height, width).
    """
    array = tiff.imread(path)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D .tif stack, but got shape {array.shape}")
    return array

def identify_path_type(path):
    """
    Identifies the type of a given path.

    Parameters:
    - path (str): A file or folder path.

    Returns:
    - str: One of "folder", "npy", or "tif"

    Raises:
    - FileNotFoundError: If the path does not exist.
    - ValueError: If the path is neither a folder, a .npy file, nor a .tif file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path does not exist: {path}")

    if os.path.isdir(path):
        return "folder"
    elif os.path.isfile(path):
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext == ".npy":
            return "npy"
        elif ext in [".tif", ".tiff"]:
            return "tif"
        else:
            raise ValueError(f"File extension '{ext}' is not supported. Only .npy and .tif/.tiff files are allowed.")
    else:
        raise ValueError(f"The path '{path}' is neither a regular file nor a folder.")


def save_thresholds_to_file(grad_thresh, bg_thresh, gw_thresh, filename):
    
    savename = os.path.splitext(filename)[0]+'_seg_params.txt'
    # Open the file in write mode
    with open(savename, 'w') as file:
        file.write(f"Gradient Threshold: {grad_thresh}\n")
        file.write(f"Black-Grey Threshold: {bg_thresh}\n")
        file.write(f"Grey-White Threshold: {gw_thresh}\n")
    
    print(f"Segmentation parameters saved to {savename}")

def npytotif(img,filepath):
    dim = img.shape
    slice_img = np.zeros(shape=(img.shape[1],img.shape[2]),dtype=int)
    for i in range(dim[0]):
        name = 'slice_' + str(i) + '.tif'
        for j in range(dim[1]):
            for k in range(dim[2]):
                color = 127*(img[i,j,k]-1)
                slice_img[j,k] = color
        cv2.imwrite(os.path.join(filepath,name),slice_img.astype('uint8'))

def sobel_gradients(img):
    sobels = [ndimage.sobel(img, axis=i) for i in range(img.ndim)]
    grad   = np.sqrt(np.sum([s**2 for s in sobels], axis=0))
    return (1/np.amax(grad))*grad

def threshold_volume(img,bg_thresh,gw_thresh):
    seg_img = np.zeros(shape=img.shape,dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i,j,k] <= bg_thresh:
                    seg_img[i,j,k] = 1
                elif img[i,j,k] <= gw_thresh:
                    seg_img[i,j,k] = 2
                else:
                    seg_img[i,j,k] = 3
    return seg_img

def threshold_grad(img,thresh):
    shape = img.shape
    thresh_img = np.zeros(shape=shape,dtype=np.uint8)
    # if len(shape) == 3:
    #     for i in range(shape[0]):
    #         for j in range(shape[1]):
    #             for k in range(shape[2]):
    #                 if img[i,j,k] <= thresh:
    #                     thresh_img[i,j,k] = 255
    # else:
    #     for i in range(shape[0]):
    #         for j in range(shape[1]):
    #             if img[i,j,k] <= thresh:
    #                     thresh_img[i,j,k] = 255

    thresh_img[img<=thresh] = 1

    return thresh_img

def analyze_seg_dist_markers(img,seg_img,markers):
    unique_markers = np.unique(seg_img)

    avg_grey = np.zeros(len(unique_markers))
    size = np.zeros(len(unique_markers))
    marker_size = np.zeros(len(unique_markers))
    for i in range(seg_img.shape[0]):
        for j in range(seg_img.shape[1]):
            for k in range(seg_img.shape[2]):
                size[seg_img[i,j,k]-1] += 1
                if markers[i,j,k] != 0:
                    avg_grey[seg_img[i,j,k]-1] += img[i,j,k]
                    marker_size[seg_img[i,j,k]-1] += 1

    for i in range(len(unique_markers)):
        avg_grey[i] = avg_grey[i]/marker_size[i]

    return avg_grey, marker_size, size 

def plot_avg_gray(seg_img,avg_gray):
    avg_img = np.zeros(shape=seg_img.shape)
    for i in range(seg_img.shape[0]):
        for j in range(seg_img.shape[1]):
            for k in range(seg_img.shape[2]):
                avg_img[i,j,k] = avg_gray[seg_img[i,j,k]-1]

    return avg_img

def watershed_image(img,grad,thresh,plot_results=False):
    
    print("Thresholding gradient image...")

    thresh_grad = threshold_grad(grad,thresh)
            
    thresh_grad = np.uint8(thresh_grad)

    print("Done.")
    print("Performing connected component analysis to identify markers...")

    markers = cc3d.connected_components(thresh_grad,connectivity=6)
            
    print("Done.")
    print("Applying watershed transform...")
    seg_img = watershed(grad,markers,watershed_line=False)
    print("Done.")

    print("Analyzing distribution of marker greyscales...")

    avg_grey, marker_size, size = analyze_seg_dist_markers(img,seg_img,markers)

    avg_img = plot_avg_gray(seg_img,avg_grey)

    print("Done.")
    
    return avg_grey, marker_size,size,seg_img,avg_img

def gradient_threshold_original(img,grad,thresh,plot_results=False):

    ROOT = os.getcwd()
    file1 = open(os.path.join(ROOT,'seg_run_output.txt'),"a")
    string = 'Threshold ' + str(thresh) + ' starting... \n'
    file1.write(string)
    file1.close()

    thresh_grad = threshold_grad(grad,thresh)
            
    thresh_grad = np.uint8(thresh_grad)

    markers = cc3d.connected_components(thresh_grad,connectivity=6)
            
    file1 = open(os.path.join(ROOT,'seg_run_output.txt'),"a")
    string = 'Finished conected component analysis. Starting watershed... \n'
    file1.write(string)
    file1.close()
    seg_img = watershed(grad,markers,watershed_line=False)

    file1 = open(os.path.join(ROOT,'seg_run_output.txt'),"a")
    string = 'Finished watershedding image. Starting marker analysis... \n'
    file1.write(string)
    file1.close()


    avg_grey, marker_size, size = analyze_seg_dist_markers(img,seg_img,markers)


    avg_img = plot_avg_gray(seg_img,avg_grey)


    file1 = open(os.path.join(ROOT,'seg_run_output.txt'),"a")
    string = 'Finished marker analysis. Starting volume threshold... \n'
    file1.write(string)
    file1.close()
    final_img = threshold_volume(avg_img,81,180)

    file1 = open(os.path.join(ROOT,'seg_run_output.txt'),"a")
    string = 'Finished!\n'
    file1.write(string)
    file1.close()


    return final_img,seg_img,avg_grey,size, avg_img

def gradient_threshold(inputs):
    '''
    This function takes a gradient image and thresholds it. 
    Following the threshold, it preformas a connected-component analysis.
    Function return the number of unique markers,
    '''

    grad,thresh = inputs

    thresh_grad = threshold_grad(grad,thresh)
            
    thresh_grad = np.uint8(thresh_grad)

    markers = cc3d.connected_components(thresh_grad,connectivity=6)

    return len(np.unique(markers))

def get_npy_files(filepath):
    # This function looks into a folder and only returns names of
    # .npy files

    files = os.listdir(filepath)
    npyfiles = []

    for file in files:
        if file.endswith('.npy'):
            npyfiles.append(file)

    return sorted(npyfiles,key=len)

def check_gradient(filename,filepath):
    # This function checks if there is a gradient image with the 
    # same file frefix as in filename

    files = get_npy_files(filepath)

    gradient = False
    for file in files:
        if file.endswith('_gradients.npy') and (file[0:-14]==filename[0:-4]):
            gradient = True 

    return gradient



def save_gradient_sweep(thresholds,num_markers,filepath,verbose=True):
    savename = os.path.splitext(filepath)[0]+'_num_marker.csv'
    if os.path.exists(savename):
        savename = savename[0:-4]+'_' + str(np.random.randint(0,1000000)) + '.csv'
    
    d = {
        'threshold' : thresholds,
        'num_markers' : num_markers
    }
    df = pd.DataFrame(d)
    df.to_csv(savename)
    if verbose:
        print("Save markers results as",savename)

def check_threshs(b_thresh,t_thresh,n_thresh):
    # This function checks if inputs for gradient
    # thresholding are sensical 

    if t_thresh < b_thresh:
        print("Uppder bound for the threshold sweep must be higher than the lower bound")
        exit()
    elif t_thresh == b_thresh:
        print("Upper and lower bounds for threshold sweep must not be the same number.")
        exit()
    elif n_thresh <= 0:
        print('Invalid number of steps for gradient threshold sweep.')
        exit()

def get_gradients(filename,img,verbose=True):

    # Creates list of all .npy files in filename directory (used to check if gradient file exists)
    files = get_npy_files(os.path.dirname(filename))

    # Filename for gradients file
    grad_save = os.path.splitext(os.path.basename(filename))[0] + '_gradients.npy'

    # Checks if gradient file exists 
    if check_gradient(os.path.basename(filename),os.path.dirname(filename)):
        if verbose:
            print("A gradient file for",os.path.basename(filename),"has been found. It will be used.")
        grads = np.load(os.path.join(os.path.dirname(filename),grad_save))
    
    # If no gradient file exists, creates one and saves it
    else:
        if verbose:
            print("No gradient file found for",os.path.basename(filename),". Computing gradients...")
        grads = sobel_gradients(img.astype(float))
        np.save(os.path.join(os.path.dirname(filename),grad_save),grads)
        if verbose:
            print("Done. Gradient file saved as",grad_save)

    return grads


def plot_results_grad_thresh_results(threshs,num_markers,filepath):
    '''
    Plots and saves the results from the gradient threshold sweep
    Outputs the threshold that yielded the maximum number of markers. 
    '''

    max_marker_thresh = threshs[np.argmax(num_markers)]

    plt.close('all')
    plt.figure(figsize=(6,5))
    plt.scatter(threshs,num_markers,s=10,color='blue')
    plt.xlabel('Gradient Threshold',fontsize=12)
    plt.ylabel('Number of Markers',fontsize=12)
    plt.gca().set_box_aspect(1.0)
    filename_without_ext = os.path.splitext(os.path.basename(filepath))[0]
    title_text = filename_without_ext + ' Gradient Threshold Sweep'
    plt.title(title_text,fontsize=12)

    savename = os.path.splitext(filepath)[0] + '_thresh_sweep.png'
    plt.savefig(savename,dpi=300)

    return max_marker_thresh

def analyze_post_water_dist(avg_grey,size,filename,verbose=True):


    counts, bin_edges = np.histogram(avg_grey, weights=size, density=True,bins=np.linspace(np.amin(avg_grey),np.amax(avg_grey),255))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        
    # Step 1: Find peaks (local maxima)
    peaks, _ = find_peaks(counts, distance=10)
    peak_xs = bin_centers[peaks]

    # Step 2: Find local minima
    minima_indices = argrelmin(counts)[0]
    minima_xs = bin_centers[minima_indices]

    # Step 3: Keep only the minima *between* the top 3 peaks
    top3_peaks = peaks[np.argsort(counts[peaks])[-3:]]
    top3_peaks.sort()

    minima_between_peaks = []
    for i in range(len(top3_peaks) - 1):
        left = top3_peaks[i]
        right = top3_peaks[i + 1]
        between = minima_indices[(minima_indices > left) & (minima_indices < right)]
        if len(between) > 0:
            min_idx = between[np.argmin(counts[between])]
            minima_between_peaks.append(min_idx)


    # Get x-values
    min1_x = bin_centers[minima_between_peaks[0]]
    min2_x = bin_centers[minima_between_peaks[1]]

    # Plotting
    plt.close('all')
    plt.figure(figsize=(7,5))
    plt.subplot(1,1,1)
    plt.plot(bin_centers, counts,color='black')
    # plt.plot(bin_centers[peaks], counts[peaks], 'ro', label='Peaks')
    plt.axvline(min1_x,color='red')
    plt.axvline(min2_x,color='red')
    plt.ylim(0)
    plt.xlabel('Marker Average Greyscale',fontsize=12)
    plt.ylabel('Frequency',fontsize=12)
    plt.title('Post-watershed Distribution')
    

    savename = os.path.splitext(filename)[0] + '_post-water_dist.png'
    plt.savefig(savename,dpi=300)   
    
    if verbose:
        print("BG Min:",round(min1_x,3))
        print("GW Min:",round(min2_x,3))

    return min1_x,min2_x



def threshold_sweep_main(filename,b_thresh,t_thresh,n_thresh,mode,verbose=True,full=False):
    '''
    This function is the main wrapper for the gradient threshold step 
    of watershed segmentation.
    '''
    
    # Performs basic checks on inputs
    check_threshs(b_thresh,t_thresh,n_thresh)

    # Get file type 
    path_type = identify_path_type(filename)

    # Load microstructure
    if path_type == "npy":
        img = np.load(filename)
    elif path_type == "tif":
        img = read_tif_stack(filename)
    

    # Finds saved gradient file or computes new one
    grads = get_gradients(filename,img,verbose=verbose)

    # Creates an array of thresholds to test
    thresholds = np.linspace(b_thresh,t_thresh,int(n_thresh))

    # Operates sweep in serial mode (one threshold at a time)
    if mode == 'serial':

        # Initializes array for number of markers outpuit
        num_markers= np.zeros(n_thresh)

        if verbose:
            print("Gradient threshold will be swept from",b_thresh,"to",t_thresh)

        # Iterates through each threshold
        for i in range(n_thresh):
            if verbose:
                print("Beginning",thresholds[i])
            num_markers[i] = gradient_threshold((grads,thresholds[i]))

    # Operates sweep in parallel mode (multiple thresholds at once)
    if mode == 'parallel':

        if verbose:
            print("Gradient threshold will be swept from",b_thresh,"to",t_thresh)
            print("Running in parallel mode")

        # Prepares inputs for parallel operation
        inputs = [(grads,threshold) for threshold in thresholds]

        # Gets an array of the number of markers for each threshold
        num_markers = pqdm(inputs,gradient_threshold,n_jobs=mp.cpu_count())

    # Saves results from threshold sweep to .csv file
    save_gradient_sweep(thresholds,num_markers,filename,verbose=verbose)

    if verbose:
        print("Gradient threshold sweep completed")

    if full:
        return thresholds,num_markers

def watershed_main(filename,thresh,verbose=True,full=False):

    # Get file type 
    path_type = identify_path_type(filename)

    # Load microstructure
    if path_type == "npy":
        img = np.load(filename)
    elif path_type == "tif":
        img = read_tif_stack(filename)

    # Finds saved gradient file or computes new one
    grads = get_gradients(filename,img)
    
    if verbose:
        print("Watershedding ",os.path.basename(filename))

    # Watersheds image
    avg_grey, marker_size,size,seg_img,avg_img = watershed_image(img,grads,thresh,plot_results=False)

    # Saves results
    savename = os.path.splitext(os.path.basename(filename))[0]+'_avg_grey.npy'
    np.save(os.path.join(os.path.dirname(filename),savename),avg_grey)
    savename = os.path.splitext(os.path.basename(filename))[0]+'_marker_size.npy'
    np.save(os.path.join(os.path.dirname(filename),savename),marker_size)
    savename = os.path.splitext(os.path.basename(filename))[0]+'_size.npy'
    np.save(os.path.join(os.path.dirname(filename),savename),size)
    savename = os.path.splitext(os.path.basename(filename))[0]+'_seg_img.npy'
    np.save(os.path.join(os.path.dirname(filename),savename),seg_img)
    savename = os.path.splitext(os.path.basename(filename))[0]+'_avg_img.npy'
    np.save(os.path.join(os.path.dirname(filename),savename),avg_img)

    if verbose:
        print("Watershed output files saved in ",os.path.dirname(filename))

    if full:
        return avg_grey, marker_size,size,seg_img,avg_img


def phase_id_main(filename,bg_thresh,gw_thresh,verbose=True,full=False,original_name=None):

    # Loads microstructure
    img = np.load(filename)

    # Phase_id's watershedded image
    seg_img = threshold_volume(img,bg_thresh,gw_thresh)

    if original_name:
        # What type of file
        path_type = identify_path_type(original_name)
        # If tif file, save as tif stack
        if path_type == "tif":
            savename = os.path.splitext(original_name)[0] + '_final_seg.tiff'

            seg_tiff_save = np.zeros(shape=seg_img.shape,dtype=np.uint8)

            seg_tiff_save[seg_img == 1] = 0
            seg_tiff_save[seg_img == 2] = 127
            seg_tiff_save[seg_img == 3] = 255

            save_3d_array_as_tiff(seg_tiff_save, savename)
        # If npy file, save as npy file
        elif path_type == "npy":
            savename = os.path.splitext(original_name)[0] + '_final_seg.npy'
            np.save(savename,seg_img) 
    else:
        savename = os.path.splitext(filename)[0][0:-8] + '_final_seg.npy'
        np.save(savename,seg_img) 

    if verbose:
        print("Segmented microstructure saved at ",savename,'final_seg.npy')

    if full:
        plt.close('all')
        plt.figure(figsize=(10,5))

        # Load original for comparison
        path_type = identify_path_type(original_name)
        # Load microstructure
        if path_type == "npy":
            grey = np.load(original_name)
        elif path_type == "tif":
            grey = read_tif_stack(original_name)

        plt.subplot(1,2,1)
        plt.imshow(grey[0],cmap='gray')
        plt.title('Greyscale Slice 0',fontsize=12)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1,2,2)
        plt.imshow(seg_img[0],cmap='gray')
        plt.title('Segmented Slice 0',fontsize=12)
        plt.xticks([])
        plt.yticks([])

        savename = os.path.splitext(filename)[0][0:-8] + '_final_slice_0.png'
        plt.savefig(savename,dpi=300)   


