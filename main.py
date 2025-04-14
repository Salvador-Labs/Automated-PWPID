import numpy as np
import multiprocessing as ml
import argparse
import segmentation_utils as seg
import os

def parseargs():
    p = argparse.ArgumentParser(description="Performes watershed segmentation on 3D greyscale microstructures.")
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


    p_thresh.add_argument('b_thresh',type=float,help='Bottom of threshold parametric sweep')
    p_thresh.add_argument('t_thresh',type=float,help='Top of threshold parametric sweep')
    p_thresh.add_argument('-n_thresh',type=int,default=20,help='Number of steps between b_thresh and t_thresh')
    p_thresh.add_argument('-mode',choices=['serial','parallel'],default='serial',help='Whether or not to perform the sweep in serial or parallel. Paralle is useful if using a workstation with high RAM capacity')

    p_watershed.add_argument('thresh',type=float,help='Gradient threshold for marker planting')

    p_segment.add_argument('bg_thesh',help='Black-grey greyscale threshold')
    p_segment.add_argument('gw_thesh',help='Grey-white greyscale threshold')

    return p.parse_args()
   




if __name__ == '__main__':

    args = parseargs()
    print(args)

    # Manually run gradient thresholding sweep
    if args.step == 'thresh':
        seg.threshold_sweep_main(args.filename,args.b_thresh,args.t_thresh,args.n_thresh,args.mode)

    # Manually watershed and save post-watershed information
    if args.step == 'watershed':
        seg.watershed_main(args.filename,args.thresh)

    # Manually phase-ID based upon predetermined greyscale thresholds
    if args.step == 'phase_id':
        seg.phase_id_main(args.filename,args.bg_thresh,args.gw_thresh)

    # Automatically segment data
    if args.step == 'full':

        # Run gradient threshold sweep
        thresholds, num_markers = seg.threshold_sweep_main(args.filename,args.b_thresh,args.t_thresh,args.nthresh,args.comp,full=True)

        # Plot threshold swep results and find maximum marker threshold. 
        max_marker_thresh = seg.plot_results_grad_thresh_results(thresholds,num_markers,args.filename)

        print("Maximum marker threshold:",round(max_marker_thresh,3))

        # Watershed image
        avg_grey, marker_size,size,seg_img,avg_img = seg.watershed_main(args.filename,max_marker_thresh,full=True)

        # Find minima
        bg_thresh, gw_thresh = seg.analyze_post_water_dist(avg_grey,size,args.filename)

        # Phase-ID image
        seg.phase_id_main(os.path.splitext(args.filename)[0]+'_avg_img.npy',bg_thresh, gw_thresh,full=True)

