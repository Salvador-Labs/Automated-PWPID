# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:18:03 2024

@author: Salvador
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import img_as_float
import multiprocessing as mp
from skimage.restoration import denoise_nl_means, estimate_sigma
import time
import re
from scipy import ndimage
import argparse
from tqdm import tqdm
from pqdm.processes import pqdm
import multiprocessing as mp

def parseargs():
    p = argparse.ArgumentParser(description="Performs NLM filter on single file")
    p.add_argument('filename',help='Name of file for filtering')
    # Which mode to run in
    subp = p.add_subparsers(title='mode',dest='mode',help='Whether to perform sweep or single filtering')
    p_sweep = subp.add_parser('sweep', help='Run sweep of NLM values.')
    p_single = subp.add_parser('single', help='Run sweep of NLM values.')

    p_sweep.add_argument('-b_h',type=float,default=0.0,help='Bottom of h parametric sweep')
    p_sweep.add_argument('-t_h',type=float,default=5.0, help='Top of h parametric sweep')
    p_sweep.add_argument('-comp',choices=['serial','parallel'],default='serial',help='Whether or not to perform the sweep in serial or parallel. Parallel is useful if using a workstation with high RAM capacity')
    p_sweep.add_argument('-nh',type=int,default=10,help='Number of steps between b_thresh and t_thresh')
    p_sweep.add_argument('-save',type=bool,default=False,help='Whether or not to save filtered images from sweep.')

    p_single.add_argument('h',type=float,help='h_val: Patch cut-off distance (grey_levels). The higher the h_val, the more agressive the filtering. ')
    
    return p.parse_args()


def make_unique_directory(base_name, parent_dir='.'):
    """
    Create a unique directory in the form base_name, base_name_1, base_name_2, etc.
    """
    existing = set(os.listdir(parent_dir))
    pattern = re.compile(rf"^{re.escape(base_name)}(?:_(\d+))?$")

    indices = []
    for name in existing:
        match = pattern.fullmatch(name)
        if match:
            if match.group(1) is None:
                indices.append(0)
            else:
                indices.append(int(match.group(1)))

    new_index = max(indices, default=-1) + 1

    new_dir_name = base_name if new_index == 0 else f"{base_name}_{new_index}"
    new_dir_path = os.path.join(parent_dir, new_dir_name)

    # Final sanity check
    while os.path.exists(new_dir_path):
        new_index += 1
        new_dir_name = f"{base_name}_{new_index}"
        new_dir_path = os.path.join(parent_dir, new_dir_name)

    os.makedirs(new_dir_path)
    return new_dir_path


def sobel_gradients(img):
    sobels = [ndimage.sobel(img, axis=i) for i in range(img.ndim)]
    grad   = np.sqrt(np.sum([s**2 for s in sobels], axis=0))
    return (1/np.amax(grad))*grad


def filter_img(inputs):

    filepath,h_val,dirname,save = inputs

    img = np.load(filepath)
    
    # Estimates sigma for image
    sigma_est = np.mean(estimate_sigma(img))
    
    # Patch size for analysis
    patch_kw = dict(patch_size=5, patch_distance=6)
    
    # slow algorithm 
    denoise = denoise_nl_means(img, h=h_val * sigma_est, preserve_range=True,fast_mode=True, **patch_kw)
    
    if dirname == None:
        savename = os.path.basename(filepath)[0:-4]+'_filtered.npy'
        np.save(os.path.join(os.path.dirname(filepath),savename),denoise)  
        print("Filtered image saved at: savename")
    else:
        if save:
            savename = os.path.basename(filepath)[0:-4]+'_filtered_'+str(round(h_val,3))+'.npy'
            np.save(os.path.join(dirname,savename),denoise) 
        
        # Compute graidents 
        grads = sobel_gradients(denoise)

        plt.close('all')
        plt.figure(figsize=(10,5))

        title_text = "h = "+ str(round(h_val,3))
        plt.suptitle(title_text,fontsize=12)

        plt.subplot(1,2,1)
        plt.hist(grads.flatten(),density=True,color='black',bins=np.linspace(0,1,200))
        plt.xlabel('Voxel Gradient',fontsize=12)
        plt.ylabel('Probability Density',fontsize=12)
        plt.gca().set_box_aspect(1.0)
        plt.title('Gradient Distribution',fontsize=12)

        plt.subplot(1,2,2)
        plt.imshow(grads[0],cmap='gray',vmin=0,vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.title('Slice 0 Gradients',fontsize=12)


        savename = os.path.basename(filepath)[0:-4]+'_filtered_'+str(round(h_val,3))+'.png'
        print("Saving figure here:",savename)
        plt.savefig(os.path.join(dirname,savename),dpi=300)

    

if __name__ == '__main__':

    args = parseargs()

    if args.mode == 'single':
        print('Will perform filter on',args.filename)
        # Filter image
        filter_img((args.filename,args.h_val,None,True))

    elif args.mode== 'sweep':

        # Make directory to save sweep results
        dirname = os.path.splitext(args.filename)[0]+'_NLM_SWEEP'
        SAVE_DIR = make_unique_directory(dirname)
        print(f"Will save results of sweep in: {SAVE_DIR}")

        h_vals = np.linspace(args.b_h,args.t_h,args.nh)
        inputs = [(args.filename,h,SAVE_DIR,args.save) for h in h_vals]

        print("Will run sweep from h_val=",args.b_h,"to",args.t_h,"with",args.nh,"h_vals")

        if args.comp == 'serial':
            print("Running in serial mode.")

            for inp in tqdm(inputs):

                filter_img(inp)

        elif args.comp == 'parallel':
            print("Running in parallel mode with",mp.cpu_count(),"cores")

            pqdm(inputs,filter_img,n_jobs=mp.cpu_count())


    
    print("Done!")
	




