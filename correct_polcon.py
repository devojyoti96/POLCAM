from basic_func import *
from optparse import OptionParser
from astropy.io import fits
import os, gc, traceback

def fit_leakage_poly(dataq, datai):
    """
    Fit leakage surface polynominal
    Parameters
    ----------
    dataq : numpy.array
        Polarization data
    datai : numpy.array
        Stokes I data
    Returns
    -------
    numpy.array
        Leakage surface array
    """
    # Calculate q_by_i and filter valid indices
    q_by_i = dataq / datai
    valid_indices = ~np.isnan(q_by_i)
    # Extract x, y, and z from valid indices
    x, y = np.where(valid_indices)
    z = q_by_i[valid_indices]
    z[np.abs(z) > 1] = 1
    # Create the design matrix for the least squares fit
    q_stack = np.column_stack((x, y, z))
    AQ = np.c_[
        np.ones(q_stack.shape[0]),
        q_stack[:, :2],
        np.prod(q_stack[:, :2], axis=1),
        q_stack[:, :2] ** 2,
    ]
    # Solve for coefficients using least squares
    CQ, _, _, _ = scipy.linalg.lstsq(AQ, q_stack[:, 2])
    # Generate the surface values
    k_indices, l_indices = np.meshgrid(
        range(dataq.shape[0]), range(dataq.shape[1]), indexing="ij"
    )
    data_backup = (
        CQ[4] * k_indices**2
        + CQ[5] * l_indices**2
        + CQ[3] * k_indices * l_indices
        + CQ[1] * k_indices
        + CQ[2] * l_indices
        + CQ[0]
    )
    return data_backup
    
def leakage_surface(imagename, outdir="", threshold=5, bkg_image="", rms_image=""):
    """
    Make Stokes I to other stokes leakage surface
    Parameters
    ----------
    imagename : str
        Name of the image
    outdir : str
        Output directory name
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
        Output directory name
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
    if header['NAXIS']!=4 or (data.shape[0]!=4 and data.shape[1]!=4):
        print ("This image: "+imagename+" is not a full stokes image. Please provide full Stokes image.")
        return 1, None, None, None, None
    try:    
        if header["CTYPE3"] == "STOKES":
            q_surface = fit_leakage_poly(data[0, 1, ...], data[0, 0, ...])
            u_surface = fit_leakage_poly(data[0, 2, ...], data[0, 0, ...])
            v_surface = fit_leakage_poly(data[0, 3, ...], data[0, 0, ...])
        elif header["CTYPE4"] == "STOKES":
            q_surface = fit_leakage_poly(data[1, 0, ...], data[0, 0, ...])
            u_surface = fit_leakage_poly(data[2, 0, ...], data[0, 0, ...])
            v_surface = fit_leakage_poly(data[3, 0, ...], data[0, 0, ...])
        else:
            print("Stokes axis is not present.")
            print("Could not make leakage surface.")
            return 1, None, None, None, None

        imagename_split = os.path.basename(imagename).split("-")
        index = imagename_split.index("iquv")
        header["BUNIT"] = "FRAC"

        if outdir == "":
            outdir = os.path.dirname(imagename) + "/leakage_surfaces"
        if os.path.isdir(outdir) == False:
            os.makedirs(outdir)

        imagename_split[index] = "q_surface"
        q_surface_name = "-".join(imagename_split)
        fits.writeto(outdir + "/" + q_surface_name, q_surface, header, overwrite=True)

        imagename_split[index] = "u_surface"
        u_surface_name = "-".join(imagename_split)
        fits.writeto(outdir + "/" + u_surface_name, u_surface, header, overwrite=True)

        imagename_split[index] = "v_surface"
        v_surface_name = "-".join(imagename_split)
        fits.writeto(outdir + "/" + v_surface_name, v_surface, header, overwrite=True)

        return (
            0,
            outdir,
            outdir + "/" + q_surface_name,
            outdir + "/" + u_surface_name,
            outdir + "/" + v_surface_name,
        )
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        print(
            "#####################\nLeakage surface estimation failed.\n#####################\n"
        )
        return 1, None, None, None, None


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
        "--leakage_surface_dir",
        dest="leakage_surface_dir",
        default="",
        help="Leakage surface directory",
        metavar="String",
    )
    parser.add_option(
        "--leakage_cor_dir",
        dest="leakage_cor_dir",
        default="",
        help="Leakage surface corrected image directory",
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
                msg, leakage_surface_dir, q_surface, u_surface, v_surface = (
                    leakage_surface(
                        options.imagename,
                        outdir=options.leakage_surface_dir,
                        threshold=float(options.threshold),
                        bkg_image=options.bkg_image,
                        rms_image=options.rms_image,
                    )
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
                    print("Leakage surface direcory : " + leakage_surface_dir)
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
