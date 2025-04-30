import numpy as np
import multiprocessing as ml
import argparse
import segmentation_utils as seg
import os
import tifffile as tiff


def parseargs():
    p = argparse.ArgumentParser(description="Converts 3D .npy, .tif, and .tiff files to desired format.")
    p.add_argument('filename',help='Name of file for conversion')
    p.add_argument('-format',default=None,choices=[None,'.npy','.tiff','.tif'],help='Desired output format')

    return p.parse_args()
    
def save_3d_array_as_tiff(array, path,dtype):

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {array.shape}")
    
    tiff.imwrite(
        path,
        array.astype(dtype),  # or uint8 depending on your data
        photometric='minisblack',  # ensures grayscale appearance
        compression='zlib'         # improves compatibility
    )

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

def check_formats(infile,outf):

    inf = os.path.splitext(infile)[1]

    if inf==outf:
        print("Input and output formats are the same.")
        exit()

if __name__ == '__main__':

    args = parseargs()

    # Get file type
    path_type = identify_path_type(args.filename)

    # Check if format requested
    if args.format:
        # Check that input and output file formats are different
        check_formats(args.filename,args.format)

        if args.format == '.npy':
            vol = read_tif_stack(args.filename)
            savename = os.path.splitext(args.filename)[0]+'.npy'
            np.save(savename,vol)
        elif args.format == '.tif':
            vol = np.load(args.filename)
            savename = os.path.splitext(args.filename)[0]+'.tif'
            save_3d_array_as_tiff(vol, savename,type(vol[0,0,0]))
        elif args.format == '.tiff':
            vol = np.load(args.filename)
            savename = os.path.splitext(args.filename)[0]+'.tiff'
            save_3d_array_as_tiff(vol, savename,type(vol[0,0,0]))

    else:
        if path_type == "npy":
            vol = np.load(args.filename)
            savename = os.path.splitext(args.filename)[0]+'.tiff'
            save_3d_array_as_tiff(vol, savename,type(vol[0,0,0]))
        elif path_type == "tif":
            vol = read_tif_stack(args.filename)
            savename = os.path.splitext(args.filename)[0]+'.npy'
            np.save(savename,vol)





