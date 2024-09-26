import os, glob
from casatasks import imsubimage, exportfits
from basic_func import *
			
########
# Inputs
########
imagedir = input("imagedir: ")
metafits = input("metafits file: ")
input_image_prefix = input("input image prefix: ")
nthreads = input("Number of threads:")
outdir = imagedir + "/pbcor_images"
iauorder = input("IAU order:")
pbdir = input("Previous PB directory:")
MWA_PB_file=input("MWA PB file:")
sweetspot_file=input("MWA sweetspot file:")
outdir += "_" + iauorder
if os.path.exists(outdir) == False:
    os.makedirs(outdir)
else:
    os.system("rm -rf " + outdir + "/*")
os.chdir(imagedir)

i_images = sorted(glob.glob(input_image_prefix + "-*I-image.fits"))
q_images = sorted(glob.glob(input_image_prefix + "-*Q-image.fits"))
u_images = sorted(glob.glob(input_image_prefix + "-*U-image.fits"))
v_images = sorted(glob.glob(input_image_prefix + "-*V-image.fits"))

for i in range(len(i_images)):
    if "MFS" not in i_images[i]:
        ch = str(
            i_images[i].split(input_image_prefix + "-")[-1].split("-I-image.fits")[0]
        )
        header = fits.getheader(i_images[i])
        freq_MHz = float(header["CRVAL3"]) / 10**6
        coarse_chan = freq_to_MWA_coarse(freq_MHz)
        pbfile = glob.glob(pbdir + "/*-coch-" + str(coarse_chan) + "*.npy")
        outfile = (
            input_image_prefix
            + "-ch-"
            + str(ch)
            + "-coch-"
            + str(coarse_chan)
            + "-iquv"
        )
        wsclean_images = [i_images[i], q_images[i], u_images[i], v_images[i]]
        output_image = wsclean_to_casaimage(
            wsclean_images=wsclean_images,
            casaimage_prefix=outfile,
            imagetype="image",
            keep_wsclean_images=True,
        )
        cmd = (
            "python3 mwapb.py --MWA_PB_file "+str(MWA_PB_file)+" --sweetspot_file "+str(sweetspot_file)+" --imagename "
            + output_image
            + " --outfile "
            + outfile.split(".fits")[0]
            + "_pbcor --metafits "
            + metafits
            + " --IAU_order "
            + str(iauorder)
            + " --num_threads "
            + str(nthreads)
            + " --verbose True"
        )
        if len(pbfile) > 0:
            cmd += " --pb_jones_file " + pbfile[0]
        else:
            cmd += " --save_pb " + outfile.split(".fits")[0] + "_pb"
        print(cmd + "\n")
        os.system(cmd)
        for stokes in ["I", "Q", "U", "V"]:
            imsubimage(
                imagename=outfile.split(".fits")[0] + "_pbcor.image",
                outfile=outfile.split(".fits")[0] + "-" + stokes + "_pbcor.image",
                stokes=stokes,
            )
            exportfits(
                imagename=outfile.split(".fits")[0] + "-" + stokes + "_pbcor.image",
                fitsimage=outfile.split(".fits")[0] + "-" + stokes + "_pbcor.fits",
            )
            os.system(
                "rm -rf " + outfile.split(".fits")[0] + "-" + stokes + "_pbcor.image"
            )
            os.system(
                "mv "
                + outfile.split(".fits")[0]
                + "-"
                + stokes
                + "_pbcor.fits "
                + outdir
            )
        if os.path.exists(outfile.split(".fits")[0] + "_pb.npy"):
            os.system("mv " + outfile.split(".fits")[0] + "_pb.npy " + outdir)
        os.system("rm -rf " + outfile + "*")

print(
    "############################\n PB corrected images are saved at: "
    + outdir
    + "\n##########################\n"
)
