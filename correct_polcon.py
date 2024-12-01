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
        q_surface = make_leakage_surface(data[0, 1, ...], data[0, 0, ...])
        u_surface = make_leakage_surface(data[0, 2, ...], data[0, 0, ...])
        v_surface = make_leakage_surface(data[0, 3, ...], data[0, 0, ...])
    elif header["CTYPE4"] == "STOKES":
        q_surface = make_leakage_surface(data[1, 0, ...], data[0, 0, ...])
        u_surface = make_leakage_surface(data[2, 0, ...], data[0, 0, ...])
        v_surface = make_leakage_surface(data[3, 0, ...], data[0, 0, ...])
    else:
        print("Stokes axis is not present.")
        print("Could not make leakage surface.")
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


def correct_leakage_surface(
    imagename,
    q_surface="",
    u_surface="",
    v_surface="",
    bkg_image="",
    rms_image="",
    leakage_cor_threshold=5.0,
    keep_original=False,
):
    """
    Correct Stokes I to other Stokes leakages
    Parameters
    ----------
    imagename : str
        Imagename
    q_surface : str
        User supplied Stokes Q leakage surface
    u_surface : str
        User supplied Stokes U leakage surface
    v_surface : str
        User supplied Stokes V leakage surface
    bkg_image : str
        Background imagename
    rms_image : str
        rms imagename
    leakage_cor_threashold : float
        Threshold for determining leakage surface (If surfaces are provided, it has no use)
    keep_original : bool
        Keep original image and overwrite it or not
    Returns
    -------
    str
        Output imagename
    """
    if (q_surface == "" or u_surface == "" or v_surface == "") or (
        os.path.exists(q_surface) == False
        or os.path.exists(u_surface) == False
        or os.path.exists(v_surface) == False
    ):
        if (bkg_image == "" or rms_image == "") or (
            os.path.exists(bkg_image) == False or os.path.exists(rms_image) == False
        ):
            bkg_image, rms_image = make_bkg_rms_image(imagename)
        print("Estimating residual leakage surfaces ....\n")
        msg, q_surface, u_surface, v_surface = leakage_surface(
            pbcor_image,
            threshold=float(leakage_cor_threshold),
            bkg_image=bkg_image,
            rms_image=rms_image,
        )
    if (
        os.path.exists(q_surface)
        and os.path.exists(u_surface)
        and os.path.exists(v_surface)
    ):
        print("Correcting residual leakage surfaces ....\n")
        q_surface_data = fits.getdata(q_surface)
        u_surface_data = fits.getdata(u_surface)
        v_surface_data = fits.getdata(v_surface)
        data = fits.getdata(imagename)
        header = fits.getheader(imagename)
        if header["CTYPE3"] == "STOKES":
            data[0, 1, ...] = data[0, 1, ...] - (
                q_surface_data[0, 0, ...] * data[0, 0, ...]
            )
            data[0, 2, ...] = data[0, 2, ...] - (
                q_surface_data[0, 0, ...] * data[0, 0, ...]
            )
            data[0, 3, ...] = data[0, 3, ...] - (
                q_surface_data[0, 0, ...] * data[0, 0, ...]
            )
        else:
            data[1, 0, ...] = data[1, 0, ...] - (
                q_surface_data[0, 0, ...] * data[0, 0, ...]
            )
            data[2, 0, ...] = data[2, 0, ...] - (
                q_surface_data[0, 0, ...] * data[0, 0, ...]
            )
            data[3, 0, ...] = data[3, 0, ...] - (
                q_surface_data[0, 0, ...] * data[0, 0, ...]
            )
        if keep_original == False:
            fits.writeto(imagename, data, header, overwrite=True)
            return imagename
        else:
            fits.writeto(
                imagename.split(".fits")[0] + "_leakagecor.fits",
                data,
                header,
                overwrite=True,
            )
            return imagename.split(".fits")[0] + "_leakagecor.fits"
    else:
        print("Cound not perform leakage surface correction.")
        return


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
        default=5.0,
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
    parser.add_option(
        "--do_leakagecor",
        dest="do_leakagecor",
        default=True,
        help="Perform leakage correction or only make leakage surfaces",
        metavar="Boolean",
    )
    parser.add_option(
        "--q_surface",
        dest="q_surface",
        default="",
        help="Stokes Q leakage surface",
        metavar="String",
    )
    parser.add_option(
        "--u_surface",
        dest="u_surface",
        default="",
        help="Stokes U leakage surface",
        metavar="String",
    )
    parser.add_option(
        "--v_surface",
        dest="v_surface",
        default="",
        help="Stokes V leakage surface",
        metavar="String",
    )
    parser.add_option(
        "--keep_original",
        dest="keep_original",
        default=True,
        help="Keep original image or overwrite it",
        metavar="Boolean",
    )
    (options, args) = parser.parse_args()
    if options.imagename == None or os.path.exists(options.imagename) == False:
        print("Please provide a valid image name.\n")
        return 1
    else:
        try:
            if eval(str(options.do_leakagecor)):
                outfile = correct_leakage_surface(
                    options.imagename,
                    q_surface=options.q_surface,
                    u_surface=options.u_surface,
                    v_surface=options.v_surface,
                    bkg_image=options.bkg_image,
                    rms_image=options.rms_image,
                    leakage_cor_threshold=float(options.threshold),
                    keep_original=eval(str(options.keep_original)),
                )
                if outfile == None:
                    gc.collect()
                    return 1
                else:
                    print("###########################")
                    print("Leakage corrected image : " + outfile + ".")
                    print("###########################")
                    gc.collect()
                    return 0
            else:
                msg, q_surface, u_surface, v_surface = leakage_surface(
                    options.imagename,
                    threshold=float(options.threshold),
                    bkg_image=options.bkg_image,
                    rms_image=options.rms_image,
                )
                if msg == 1:
                    gc.collect()
                    return 1
                else:
                    print("###########################")
                    print(
                        "Leakage surfaces : "
                        + os.path.basename(q_surface)
                        + ","
                        + os.path.basename(u_surface)
                        + ","
                        + os.path.basename(v_surface)
                        + "."
                    )
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
