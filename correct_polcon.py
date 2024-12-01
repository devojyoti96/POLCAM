from basic_func import *
from optparse import OptionParser
from astropy.io import fits
import os, gc, traceback


def leakage_surface(imagename, threshold=5, bkg_image="", rms_image=""):
    """
    Make Stokes I to other stokes leakage surface
    Parameters
    ----------
    imagename : str
        Name of the image
    threshold : float
        Threshold to choose sources from Stokes I
    bkg_image : str
        Background image
    rms_image : str
        rms image
    Returns
    -------
    int
        Success message
    str
        Q surface image
    str
        U surface image
    str
        V surface image
    """
    if (
        bkg_image == ""
        or rms_image == ""
        or os.path.exists(bkg_image) == False
        or os.path.exists(rms_image) == False
    ):
        bkg_image, rms_image = make_bkg_rms_image(imagename)
    data = fits.getdata(imagename)
    header = fits.getheader(imagename)

    if header["CTYPE3"] == "STOKES":
        q_surface=make_leakage_surface(data[0,1,...],data[0,0,...])
        u_surface=make_leakage_surface(data[0,2,...],data[0,0,...])
        v_surface=make_leakage_surface(data[0,3,...],data[0,0,...])
    elif header["CTYPE4"] == "STOKES":
        q_surface=make_leakage_surface(data[1,0,...],data[0,0,...])
        u_surface=make_leakage_surface(data[2,0,...],data[0,0,...])
        v_surface=make_leakage_surface(data[3,0,...],data[0,0,...])
    else:
        print("Stokes axis is not present.\n")
        return 1, None, None, None
    
    imagename_split = imagename.split("-")
    index = imagename_split.index("iquv")
    header["BUNIT"] = "FRAC"

    imagename_split[index] = "q_surface"
    q_surface_name = "-".join(imagename_split)
    fits.writeto(q_surface_name, q_surface, header, overwrite=True)

    imagename_split[index] = "u_surface"
    u_surface_name = "-".join(imagename_split)
    fits.writeto(u_surface_name, u_surface, header, overwrite=True)

    imagename_split[index] = "v_surface"
    v_surface_name = "-".join(imagename_split)
    fits.writeto(v_surface_name, v_surface, header, overwrite=True)

    return 0, q_surface_name, u_surface_name, v_surface_name


def main():
    usage = "Determine pol-conversion leakage surfaces"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--imagename",
        dest="imagename",
        default=None,
        help="Name of the image",
        metavar="String",
    )
    parser.add_option(
        "--threshold",
        dest="threshold",
        default=6.0,
        help="Threshold for source detection",
        metavar="Float",
    )
    parser.add_option(
        "--bkg_image",
        dest="bkg_image",
        default="",
        help="Background imagename",
        metavar="String",
    )
    parser.add_option(
        "--rms_image",
        dest="rms_image",
        default="",
        help="rms imagename",
        metavar="String",
    )
    (options, args) = parser.parse_args()
    if options.imagename == None or os.path.exists(options.imagename) == False:
        print("Please provide a valid image name.\n")
        return 1
    else:
        try:
            msg, q_surface, u_surface, v_surface = leakage_surface(
                options.imagename,
                threshold=float(options.threshold),
                bkg_image=options.bkg_image,
                rms_image=options.rms_image,
            )
            if msg == 1:
                print("Could not make leakage surface.\n")
                gc.collect()
                return 1
            else:
                print("###########################")
                print("Leakage surfaces : "+os.path.basename(q_surface)+','+os.path.basename(u_surface)+','+os.path.basename(v_surface)+".")
                print("###########################")
                gc.collect()
                return 0
        except Exception as e:
            traceback.print_exc()
            gc.collect()
            return 1
            

if __name__ == "__main__":
    result = main()
    os._exit(result)            
