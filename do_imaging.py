import os
from basic_func import *

########
# Inputs
########
msname = input('Msname: ')
chan_out=input('Number of output channels: ')

def perform_spectral_imaging(msname,nchan,multiscale_scales=[],weight='briggs',robust=0.0,threshold=5,minuv_l=200,ncpu=-1,mem=-1)
	msname=os.path.abspath(msname)
	prefix=os.path.basename(msname).split('.ms')[0]+'_nchan_'+str(nchan)
    workdir=os.path.dirname(os.path.abspath(msname))+'/imagedir_MFS_ch_'+str(chan_out)+'_'+os.path.basename(msname).split('.ms')[0]
	if os.path.exists(workdir)==False:
		os.makedirs(workdir)
	else:
		os.system('rm -rf '+workdir+'/*')
	cwd=os.getcwd()	
	os.chdir(workdir)
    cellsize=calc_cellsize(msname, 3)
    imsize = calc_imsize(msname, 3)
	
	if weight == "briggs":
        weight += " " + str(robust)
    wsclean_args = [
        "-scale " + str(cellsize) + "asec",
        "-size " + str(imsize) + " " + str(imsize),
        "-no-dirty",
        "-weight " + weight,
        "-name " + prefix,
        "-pol " + str(pol),
        "-niter 10000",
        "-mgain 0.85",
        "-nmiter 5",
        "-gain 0.1",
        "-auto-threshold " + str(threshold) + " -auto-mask " + str(threshold + 0.1),
        "-minuv-l 200",
        "-use-wgridder",
        "-channels-out "+str(nchan),
        "-temp-dir "+workdir,
        "-join-channels",
    ]
    if len(multiscale_scales) > 0:
        wsclean_args.append("-multiscale")
        wsclean_args.append("-multiscale-gain 0.1")
        wsclean_args.append("-multiscale-scales " + ",".join(multiscale_scales))
    if ncpu > 0:
        wsclean_args.append("-j " + str(ncpu))
    if mem > 0:
        wsclean_args.append("-abs-mem " + str(mem))
	for pol in ['I','QU','V']:
	    if pol=='QU':
	        wsclean_cmd='wsclean '+' '.join(wsclean_args)+' -join-polarizations -pol '+pol+' '+msname   
	    else:
	        wsclean_cmd='wsclean '+' '.join(wsclean_args)+' -pol '+pol+' '+msname        
	    print (wsclean_cmd)
        os.system(wsclean_cmd + " > tmp_wsclean")
        os.system("rm -rf tmp_wsclean")
    wsclean_images = glob.glob(prefix + "*I*image.fits")
    wsclean_models = glob.glob(prefix + "*model.fits")
    model_flux, rms_DR, neg_DR = calc_dyn_range(
        wsclean_images, wsclean_models, fits_mask
        )




