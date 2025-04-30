import numpy as np
import multiprocessing as ml
import argparse
import segmentation_utils as seg
import os
from tqdm import tqdm

def parseargs():
    p = argparse.ArgumentParser(description="Performs watershed segmentation on 3D greyscale microstructures.")
    p.add_argument('filename',help='Name of file for segmentation')
    subp = p.add_subparsers(title='step',dest='step',help='Choose step in segmentation process')
    p_auto = subp.add_parser('full', help='Run full automated segmentation.')
    p_thresh = subp.add_parser('thresh', help='Run threshold sweep.')
    p_watershed = subp.add_parser('watershed', help='Watershed image.')
    p_segment = subp.add_parser('phase_id', help="Segment watershedded image.")

    p_auto.add_argument('-b_thresh',type=float,default=0.0,help='Bottom of threshold parametric sweep')
    p_auto.add_argument('-t_thresh',type=float,default=0.3, help='Top of threshold parametric sweep')
    p_auto.add_argument('-comp',choices=['serial','parallel'],default='serial',help='Whether or not to perform the sweep in serial or parallel. Parallel is useful if using a workstation with high RAM capacity')
    p_auto.add_argument('-nthresh',type=int,default=50,help='Number of steps between b_thresh and t_thresh')
    p_auto.add_argument('-verbose',type=bool,default=True,help='Whether or not to print verbose output')


    p_thresh.add_argument('b_thresh',type=float,help='Bottom of threshold parametric sweep')
    p_thresh.add_argument('t_thresh',type=float,help='Top of threshold parametric sweep')
    p_thresh.add_argument('-n_thresh',type=int,default=20,help='Number of steps between b_thresh and t_thresh')
    p_thresh.add_argument('-mode',choices=['serial','parallel'],default='serial',help='Whether or not to perform the sweep in serial or parallel. Paralle is useful if using a workstation with high RAM capacity')
    p_thresh.add_argument('-verbose',type=bool,default=True,help='Whether or not to print verbose output')

    p_watershed.add_argument('thresh',type=float,help='Gradient threshold for marker planting')
    p_watershed.add_argument('-verbose',type=bool,default=True,help='Whether or not to print verbose output')


    p_segment.add_argument('bg_thesh',help='Black-grey greyscale threshold')
    p_segment.add_argument('gw_thesh',help='Grey-white greyscale threshold')
    p_segment.add_argument('-verbose',type=bool,default=True,help='Whether or not to print verbose output')


    return p.parse_args()
   




if __name__ == '__main__':

    args = parseargs()

    # Figure out if operating on single file or folder
    path_type = seg.identify_path_type(args.filename)

    if path_type == "folder":
        files = seg.list_tif_npy_files(args.filename)
        # files = [os.path.join(args.filename,file) for file in files]
    elif path_type in ["tif", ".npy"]:
        files = [args.filename]



    # Manually run gradient thresholding sweep
    if args.step == 'thresh':
        print("Will run gradient threshold sweep on", len(files),"files")
        for i in range(len(files)):
            seg.threshold_sweep_main(files[i],args.b_thresh,args.t_thresh,args.n_thresh,args.mode,verbose=args.verbose)

    # Manually watershed and save post-watershed information
    if args.step == 'watershed':
        print("Will watershed", len(files),"files with gradient threshold",args.thresh)
        for i in range(len(files)):
            seg.watershed_main(files[i],args.thresh,verbose=args.verbose)

    # Manually phase-ID based upon predetermined greyscale thresholds
    if args.step == 'phase_id':
        print("Will phase-id", len(files),"files with black-grey threshold",args.bg_thresh, "ande grey-white threshold",args.gw_thresh)
        for i in range(len(files)):
            seg.phase_id_main(files[i],args.bg_thresh,args.gw_thresh,verbose=args.verbose)

    # Automatically segment data
    if args.step == 'full':
        print("Will run full segmentation on", len(files),"files")
        for i in range(len(files)):
            # Run gradient threshold sweep
            thresholds, num_markers = seg.threshold_sweep_main(files[i],args.b_thresh,args.t_thresh,args.nthresh,args.comp,verbose=args.verbose,full=True)

            # Plot threshold swep results and find maximum marker threshold. 
            max_marker_thresh = seg.plot_results_grad_thresh_results(thresholds,num_markers,files[i])

            if args.verbose:
                print("Maximum marker threshold:",round(max_marker_thresh,3))

            # Watershed image
            avg_grey, marker_size,size,seg_img,avg_img = seg.watershed_main(files[i],max_marker_thresh,verbose=args.verbose,full=True)

            # Find minima
            bg_thresh, gw_thresh = seg.analyze_post_water_dist(avg_grey,size,files[i],verbose=args.verbose)

            # Phase-ID image
            seg.phase_id_main(os.path.splitext(files[i])[0]+'_avg_img.npy',bg_thresh, gw_thresh,verbose=args.verbose,full=True,original_name=files[i])

            seg.save_thresholds_to_file(max_marker_thresh, bg_thresh, gw_thresh, files[i])



