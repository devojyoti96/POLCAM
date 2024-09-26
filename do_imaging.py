import os
from basic_func import *
from optparse import OptionParser
os.system('rm -rf casa*log')

def perform_spectral_imaging(msname,nchan,multiscale_scales=[],weight='briggs',robust=0.0,threshold=5,minuv_l=75,ncpu=-1,mem=-1):
    """
    Performing spectral imaging
    Parameters
    ----------
    msname : str
        Name of the measurement set
    nchan : int
        Number of spectral channel
    multiscale_scales : list
        Multiscale scales 
    weight : str
        Image weighting
    robust : str
        Robust parameters for briggs weighting
    threshold : float
        Auto-threshold
    minuv_l : float
        Minimum uv-range in lambda
    ncpu : int
        Number of CPU threads to use
    mem : float
        Memory in GB                    
    """  
    pwd=os.getcwd()      
	msname=os.path.abspath(msname)
    workdir=os.path.dirname(os.path.abspath(msname))+'/imagedir_MFS_ch_'+str(chan_out)+'_'+os.path.basename(msname).split('.ms')[0]
	if os.path.exists(workdir)==False:
		os.makedirs(workdir)
	else:
		os.system('rm -rf '+workdir+'/*')
	prefix=workdir+'/'+os.path.basename(msname).split('.ms')[0]+'_nchan_'+str(nchan)	
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
    os.chdir(pwd)    
    return 0,workdir    
    
################################        
def main():
    usage = "Perform spectral imaging"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of the measurement set",
        metavar="String",
    )
    parser.add_option(
        "--nchan",
        dest="nchan",
        default=1,
        help="Number of spectral channels",
        metavar="Integer",
    )
    parser.add_option(
        "--multiscale_scales",
        dest="multiscale_scales",
        default='',
        help="Multiscale scales",
        metavar="String",
    )
    parser.add_option(
        "--weight",
        dest="weight",
        default='natural',
        help="Image weighting",
        metavar="String",
    )
    parser.add_option(
        "--robust",
        dest="robust",
        default=0.0,
        help="Robust parameter for briggs weighting",
        metavar="Float",
    )
    parser.add_option(
        "--threshold",
        dest="threshold",
        default='threshold',
        help="Auto threshold for CLEANing",
        metavar="Float",
    )
    parser.add_option(
        "--minuv_l",
        dest="minuv_l",
        default=75,
        help="Minimum uv-range in lambda",
        metavar="Float",
    )
    parser.add_option(
        "--ncpu",
        dest="ncpu",
        default=-1,
        help="Numbers of CPU threads to be used",
        metavar="Integer",
    )
    parser.add_option(
        "--mem",
        dest="mem",
        default=-1,
        help="Memory in GB to be used",
        metavar="Float",
    )
    if options.msname==None:
        print ('Please provide the measurement set name.\n')
        return 1   
    scales=[int(i) for i in multiscale_scales.split(',')]    
    msg, imagedir = perform_spectral_imaging(options.msname,int(options.nchan),multiscale_scales=scales,weight=options.weight,robust=float(options.robust),\
                                threshold=float(options.threshold),minuv_l=float(options.minuv_l),ncpu=int(options.ncpu),mem=float(options.mem))
    print ('Images are saved in : ',imagedir)
    return msg

if __name__ == "__main__":
    result=main()
    os._exit(result)     
    
    



