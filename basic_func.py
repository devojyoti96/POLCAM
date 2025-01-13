from casatools import table, msmetadata
from casatasks import importfits, exportfits, imsubimage
from astropy.io import fits
from astropy.table import Table
import astropy.units as u, os
from astropy.coordinates import AltAz, EarthLocation, get_sun, SkyCoord, Angle
from astropy.time import Time
import numpy as np, os, psutil, time, glob, gc, scipy, copy, math
from pathlib import Path
from mwapb import get_pb_radec

os.system("rm -rf casa*log")
MWALON = 116.67
MWALAT = -26.7
MWAALT = 377.8


def ra_dec_to_deg(ra_hms, dec_dms):
    """
    Convert RA and Dec from hms and dms format to degrees
    Parameters
    ----------
    ra_hms: str
        Right Ascension in 'hms' format
    dec_dms : str
        Declination in 'dms' format
    Returns
    -------
    tuple
        RA and Dec in degrees
    """
    ra = Angle(ra_hms, unit=u.hourangle)
    dec = Angle(dec_dms, unit=u.deg)
    return ra.deg, dec.deg


def ra_dec_to_hms_dms(ra_deg, dec_deg):
    """
    Convert RA and Dec in degrees to hms and dms format
    Parameters
    ----------
    ra_deg : float
        Right Ascension in degrees.
    dec_deg : float
        Declination in degrees.
    Returns
    -------
    tuple
        RA in h:m:s format, Dec in d:m:s format (e.g., '1h5m0s', '1d5m0s').
    """
    # Convert RA to h:m:s
    if ra_deg < 0:
        ra_deg += 360
    ra = Angle(ra_deg, unit=u.deg)
    ra_hms = ra.to_string(unit=u.hourangle, sep=":").split(":")
    ra_hms = ra_hms[0] + "h" + ra_hms[1] + "m" + ra_hms[2] + "s"
    # Convert Dec to d:m:s
    dec = Angle(dec_deg, unit=u.deg)
    dec_dms = dec.to_string(unit=u.deg, sep=":", alwayssign=True).split(":")
    dec_dms = dec_dms[0] + "d" + dec_dms[1] + "m" + dec_dms[2] + "s"
    return ra_hms, dec_dms


def get_directory_size(directory):
    """
    Calculate the total size of a directory and its subdirectories.
    Parameters
    ----------
    direcotry : str
        Directory name
    Returns
    -------
    float
        Directory size in GB
    """
    dir_size = sum(f.stat().st_size for f in Path(directory).rglob("*") if f.is_file())
    dir_size = dir_size / (1024**3)
    return dir_size


def freq_to_MWA_coarse(freq):
    """
    Frequency to MWA coarse channel conversion

    Parameters
    ----------
    freq : float
            Frequency in MHz
    Returns
    -------
    int
            MWA coarse channel number
    """
    freq = float(freq)
    coarse_chans = [[(i * 1.28) - 0.64, (i * 1.28) + 0.64] for i in range(300)]
    for i in range(len(coarse_chans)):
        ch0 = round(coarse_chans[i][0], 2)
        ch1 = round(coarse_chans[i][1], 2)
        if freq >= ch0 and freq < ch1:
            return i


def get_chans_flags(msname):
    """
    Get channels flagged or not
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    numpy.array
        A boolean array indicating whether the channel is completely flagged or not
    """
    tb = table()
    tb.open(msname)
    flag = tb.getcol("FLAG")
    tb.close()
    chan_flags = np.all(np.all(flag, axis=-1), axis=0)
    return chan_flags


def calc_field_of_view(msname, FWHM=True):
    """
    Calculate optimum field of view in arcsec using maximum dish/aperture size in the array
    Parameters
    ----------
    msname : str
        Name of the measurement set
    FWHM : bool
            Upto FWHM, otherwise upto first null
    Returns
    -------
    float
            Field of view in arcsec
    """
    msmd = msmetadata()
    tb = table()
    msmd.open(msname)
    freq = np.nanmax(msmd.chanfreqs(0))
    wavelength = 299792458.0 / (freq)
    msmd.close()
    tb.open(msname + "/ANTENNA")
    dia = np.nanmin(tb.getcol("DISH_DIAMETER"))
    tb.close()
    if FWHM == True:
        FOV = np.rad2deg((1.22 * wavelength) / dia)
    else:
        FOV = 2 * np.rad2deg((1.02 * wavelength) / dia)
    return FOV * 3600  ### In arcsecs


def calc_psf(msname):
    """
    Function to calculate PSF size in arcsec
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    float
        PSF size in arcsec
    """
    maxuv_m, maxuv_l = calc_maxuv(msname)
    psf = np.rad2deg(1.22 / maxuv_l) * 3600.0  # In arcsec
    return psf


def calc_cellsize(msname, num_pixel_in_psf):
    """
    Calculate pixel size in arcsec
    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : int
        Number of pixels in one PSF
    Returns
    -------
    int
        Pixel size in arcsec
    """
    psf = calc_psf(msname)
    pixel = math.ceil(psf / num_pixel_in_psf) - 1
    return pixel


def calc_imsize(msname, num_pixel_in_psf, max_num_pix = 8192,  FWHM=True):
    """
    Calculate image pixel size
    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : int
        Number of pixels in one PSF
    max_num_pix : int
        Maximum number of pixels 
    FWHM : bool
        Image upto FWHM or first null
    Returns
    -------
    int
        Number of pixels
    """
    cellsize = calc_cellsize(msname, num_pixel_in_psf)
    fov = calc_field_of_view(msname, FWHM=FWHM)
    imsize = int(fov / cellsize)
    pow2 = round(np.log2(imsize / 10.0), 0)
    imsize = int((2**pow2) * 10)
    if imsize>max_num_pix:
        imsize=max_num_pix
    return imsize


def calc_multiscale_scales(msname, num_pixel_in_psf, max_scale=16):
    """
    Calculate multiscale scales
    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : int
        Number of pixels in one PSF
    max_scale : float
        Maximum scale in arcmin
    Returns
    -------
    list
        Multiscale scales in pixel units
    """
    psf = calc_psf(msname)
    multiscale_scales = [0, num_pixel_in_psf]
    max_scale_pixel = int(max_scale * 60 / psf)
    other_scales = np.linspace(3 * num_pixel_in_psf, max_scale_pixel, 3).astype("int")
    for scale in other_scales:
        multiscale_scales.append(scale)
    return multiscale_scales


def calc_maxuv(msname):
    """
    Calculate maximum UV
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    float
        Maximum UV in meter
    float
        Maximum UV in wavelength
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = np.nanmin(msmd.chanfreqs(0))
    wavelength = 299792458.0 / (freq)
    msmd.close()
    tb = table()
    tb.open(msname)
    uvw = tb.getcol("UVW")
    tb.close()
    u, v, w = [uvw[i, :] for i in range(3)]
    maxu = float(np.nanmax(u))
    maxv = float(np.nanmax(v))
    maxuv = np.nanmax([maxu, maxv])
    return maxuv, maxuv / wavelength


def calc_bw_smearing_freqwidth(msname):
    """
    Function to calculate spectral width to procude bandwidth smearing
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    float
        Spectral width in MHz
    """
    R = 0.9
    fov = 3600  # 2 times size of the Sun
    psf = calc_psf(msname)
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0)
    msmd.close()
    delta_nu = np.sqrt((1 / R**2) - 1) * (psf / fov) * freq
    delta_nu /= 10**6
    return round(delta_nu, 2)


def get_calibration_uvrange(msname):
    """
    Calibration baseline range suitable for GLEAM model
    Parameters
    ----------
    msname : str
        Name of the measurement set
    Returns
    -------
    str
        UV-range for the calibration
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0)
    msmd.close()
    wavelength = (3 * 10**8) / freq
    minuv_m = 112
    maxuv_m = 2500
    minuv_l = round(minuv_m / wavelength, 1)
    maxuv_l = round(maxuv_m / wavelength, 1)
    uvrange = str(minuv_l) + "~" + str(maxuv_l) + "lambda"
    return uvrange


def get_phasecenter(msname):
    """
        Get phasecenter of the measurement set
    Parameters
    ----------
    msname : str
        Measurement set name
        Returns
        -------
        tuple
                Phasecenter of the measurement set in ['RA','DEC'] in hh mm ss dd mm ss format
        float
                RA in degree
        float
                DEC in degree
    """
    msmd = msmetadata()
    msmd.open(msname)
    phasecenter = msmd.phasecenter()
    msmd.close()
    radeg = np.rad2deg(phasecenter["m0"]["value"])
    decdeg = np.rad2deg(phasecenter["m1"]["value"])
    radec_str = ra_dec_to_hms_dms(radeg, decdeg)
    return radec_str, radeg, decdeg


def create_batch_script_nonhpc(cmd, basedir, basename):
    """
    Function to make a batch script not non-HPC environment
    Parameters
    ----------
    cmd : str
            Command to run
    basedir : str
            Base directory of the measurement set
    basename : str
            Base name of the batch files
    """
    batch_file = basedir + "/" + basename + ".batch"
    cmd_batch = basedir + "/" + basename + "_cmd.batch"
    if os.path.isdir(basedir + "/logs") == False:
        os.makedirs(basedir + "/logs")
    outputfile = basedir + "/logs/" + basename + ".log"
    pid_file = basedir + "/pids.txt"
    running_touch_file = basedir + "/.Running_" + basename
    finished_touch_file = basedir + "/.Finished_" + basename
    os.system("rm -rf " + finished_touch_file + "*")
    finished_touch_file_error = finished_touch_file + "_error"
    finished_touch_file_success = finished_touch_file + "_0"
    cmd_file_content = f"""touch {running_touch_file}\n{cmd}\nsleep 2\nexit_code=$?\nrm -rf {running_touch_file}\nif [ $? -ne 0 ]\nthen touch {finished_touch_file_error}\nelse touch {finished_touch_file_success}\nfi"""
    batch_file_content = f"""export PYTHONUNBUFFERED=1\nnohup sh {cmd_batch}> {outputfile} 2>&1 &\necho $! >> {pid_file}\nsleep 2\n rm -rf {batch_file}\n rm -rf {cmd_batch}"""
    if os.path.exists(cmd_batch):
        os.system("rm -rf " + cmd_batch)
    if os.path.exists(batch_file):
        os.system("rm -rf " + batch_file)
    with open(cmd_batch, "w") as cmd_batch_file:
        cmd_batch_file.write(cmd_file_content)
    with open(batch_file, "w") as b_file:
        b_file.write(batch_file_content)
    os.system("chmod a+rwx " + batch_file)
    os.system("chmod a+rwx " + cmd_batch)
    del cmd
    return basedir + "/" + basename + ".batch"


def get_column_size(msname, colname):
    """
    Get a column size in GB
    Parameters
    ----------
    msname : str
        Name of the ms
    colname : str
        Name of the column
    Returns
    -------
    float
        Size of the column in GB
    """
    tb = table()
    tb.open(msname)
    if colname not in tb.colnames():
        print("No " + colname + " column found in this Measurement Set.")
        tb.close()
        return 0
    # Get the shape of the DATA column and the number of rows
    data_desc = tb.getcolshapestring(colname)[0]
    data_shape_0 = int(
        data_desc.split(",")[0].split("[")[-1]
    )  # shape of each entry (channels, polarization)
    data_shape_1 = int(data_desc.split(", ")[-1].split("]")[0])
    num_rows = tb.nrows()
    bytes_per_element = 16
    # Calculate the estimated size
    estimated_size_bytes = num_rows * data_shape_0 * data_shape_1 * bytes_per_element
    estimated_size_gb = estimated_size_bytes / (1024**3)
    tb.close()
    return estimated_size_gb


def make_stokes_cube(
    wsclean_images,
    outfile_name,
    imagetype="casa",
    keep_wsclean_images=True,
):
    """
    Function to convert WSClean images in Stokes cube image (Stokes modes : 'IQUV', 'XXYY', 'RRLL', 'I', 'QU', 'IV','IQ')
    Parameters
    ----------
    wsclean_images : list
        List of WSClean images
    outfile_name : str
        Name of the output file
    imagetype : str
        'casa' or 'fits' image
    keep_wsclean_images : bool
        Keep the WSClean images or not
    Returns
    -------
    str
        Output imagename
    """
    stokes = []
    wsclean_images = sorted(wsclean_images)
    for i in wsclean_images:
        name_split = os.path.basename(i).split(".fits")[0].split("-")
        if len(name_split) >= 3:
            if name_split[-2] not in stokes:
                stokes.append(name_split[-2])
        else:
            if "I" not in stokes:
                stokes.append("I")
    stokes = sorted(stokes)
    imagename_prefix = "temp_" + os.path.basename(wsclean_images[0]).split("-I")[0]
    imagename = imagename_prefix + ".image"
    if (
        stokes != ["I", "Q", "U", "V"]
        and stokes != ["XX", "YY"]
        and stokes != ["LL", "RR"]
        and stokes != ["I", "V"]
        and stokes != ["Q", "U"]
        and stokes != ["I"]
        and stokes != ["I", "Q"]
    ):
        print("Stokes axes are not in 'IQUV','I','QU','IV','IQ','XX,YY' or 'RR,LL'. \n")
        return
    elif stokes == ["I"]:
        if os.path.isdir(imagename):
            os.system("rm -rf " + imagename)
        importfits(
            fitsimage=wsclean_images[0],
            imagename=imagename,
            defaultaxes=True,
            defaultaxesvalues=["ra", "dec", "stokes", "freq"],
        )
    else:
        if stokes == ["I", "V"]:
            for i in wsclean_images:
                if "-I-" in i:
                    data = fits.getdata(i)
                    header = fits.getheader(i)
                else:
                    data = np.append(data, fits.getdata(i), axis=0)
            header["NAXIS4"] = 2.0
            header["CRVAL4"] = 1.0
            header["CDELT4"] = 3.0
        elif stokes == ["I", "Q", "U", "V"]:
            for i in wsclean_images:
                if "-I-" in i:
                    data = fits.getdata(i)
                    header = fits.getheader(i)
                else:
                    data = np.append(data, fits.getdata(i), axis=0)
            header["NAXIS4"] = 4.0
            header["CRVAL4"] = 1.0
            header["CDELT4"] = 1.0
        elif stokes == ["XX", "YY"]:
            for i in wsclean_images:
                if "-XX-" in i:
                    data = fits.getdata(i)
                    header = fits.getheader(i)
                else:
                    data = np.append(data, fits.getdata(i), axis=0)
            header["NAXIS4"] = 2.0
            header["CRVAL4"] = -5.0
            header["CDELT4"] = -1.0
        elif stokes == ["LL", "RR"]:
            wsclean_images.reverse()
            for i in wsclean_images:
                if "-RR-" in i:
                    data = fits.getdata(i)
                    header = fits.getheader(i)
                else:
                    data = np.append(data, fits.getdata(i), axis=0)
            header["NAXIS4"] = 2.0
            header["CRVAL4"] = -1.0
            header["CDELT4"] = -1.0
        elif stokes == ["Q", "U"]:
            for i in wsclean_images:
                if "-Q-" in i:
                    data = fits.getdata(i)
                    header = fits.getheader(i)
                else:
                    data = np.append(data, fits.getdata(i), axis=0)
            header["NAXIS4"] = 2.0
            header["CRVAL4"] = 2.0
            header["CDELT4"] = 1.0
        elif stokes == ["I", "Q"]:
            for i in wsclean_images:
                if "-I-" in i:
                    data = fits.getdata(i)
                    header = fits.getheader(i)
                else:
                    data = np.append(data, fits.getdata(i), axis=0)
            header["NAXIS4"] = 2.0
            header["CRVAL4"] = 1.0
            header["CDELT4"] = 1.0
        ###############################
        # Final image preparation
        ###############################
        fits.writeto(
            imagename_prefix + ".fits", data=data, header=header, overwrite=True
        )
        if os.path.isdir(imagename):
            os.system("rm -rf " + imagename)
        importfits(
            fitsimage=imagename_prefix + ".fits",
            imagename=imagename,
            defaultaxes=True,
            defaultaxesvalues=["ra", "dec", "stokes", "freq"],
        )
        os.system("rm -rf " + imagename_prefix + ".fits")
    ###############################
    # Final returns
    ###############################
    if keep_wsclean_images == False:
        for i in wsclean_images:
            os.system("rm -rf " + i)
    if os.path.exists(outfile_name):
        os.system("rm -rf " + outfile_name)
    if imagetype == "casa":
        os.system("mv " + imagename + " " + outfile_name)
    else:
        exportfits(
            imagename=imagename,
            fitsimage=outfile_name,
            dropstokes=False,
            dropdeg=False,
            overwrite=True,
        )
        os.system("rm -rf " + imagename)
    gc.collect()
    return outfile_name


def make_bkg_rms_image(imagename):
    """
    Make background and rms image
    Parameters
    ----------
    imagename : str
        Image name
    Returns
    -------
    str
        Background image
    str
        rms image
    """
    image_prefix = imagename.split(".fits")[0]
    if (
        os.path.exists(image_prefix + "_rms.fits") == False
        and os.path.exists(image_prefix + "_bkg.fits") == False
    ):
        if os.path.exists(image_prefix + "_I.image"):
            os.system("rm -rf " + image_prefix + "_I.image")
        imsubimage(
            imagename=imagename,
            outfile=image_prefix + "_I.image",
            stokes="I",
            dropdeg=True,
        )
        if os.path.exists(image_prefix + "_I.fits"):
            os.system("rm -rf " + image_prefix + "_I.fits")
        exportfits(
            imagename=image_prefix + "_I.image",
            fitsimage=image_prefix + "_I.fits",
            dropdeg=True,
        )
        os.system("rm -rf " + image_prefix + "_I.image")
        I_imagename = image_prefix + "_I.fits"
        I_image_prefix = I_imagename.split(".fits")[0]
        print("#########################")
        print("Estimating noise map using BANE...\n")
        bane_cmd = "BANE --noclobber " + I_imagename
        print(bane_cmd + "\n")
        print("#########################")
        os.system(bane_cmd + ">tmp")
        if I_image_prefix != image_prefix:
            os.system(
                "mv " + I_image_prefix + "_bkg.fits " + image_prefix + "_bkg.fits"
            )
            os.system(
                "mv " + I_image_prefix + "_rms.fits " + image_prefix + "_rms.fits"
            )
        os.system("rm -rf " + I_imagename)
    rms_image = image_prefix + "_rms.fits"
    bkg_image = image_prefix + "_bkg.fits"
    return bkg_image, rms_image


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


def cal_field_averaged_polfrac(
    pointing_ra_deg,
    pointing_dec_deg,
    fov=30,
    stokesI_table="GGSM.fits",
    stokesP_table="POGS.fits",
):
    """
    Calculate field averaged polarization fraction
    Parameters
    ----------
    pointing_ra_deg : float
        Pointing RA in degree
    pointing_dec_deg : float
        Pointing DEC in degree
    fov : float
        Field of view in degree
    stokesI_table : str
        Stokes I catalog fits
    stokesP_table : str
        Polarization catalog fits
    Returns
    -------
    float
        Polarization fraction
    """
    print(
        "Searching a "
        + str(fov)
        + " degree field of view centered at RA : "
        + str(pointing_ra_deg)
        + " and DEC: "
        + str(pointing_dec_deg)
        + "degree.\n"
    )
    ###########################
    # Stokes I calculation
    ###########################
    stokes_I = Table.read(stokesI_table)
    ra = np.array(stokes_I["RAJ2000"].tolist())
    dec = np.array(stokes_I["DEJ2000"].tolist())
    I_flux = np.array(stokes_I["S_200"].tolist())
    source_angular_distances = angular_distance(
        pointing_ra_deg, pointing_dec_deg, ra, dec
    )
    pos = np.where(source_angular_distances <= fov)
    I_total_flux = round(np.nansum(I_flux[pos]), 2)

    #############################
    # Linpol calculation
    #############################
    stokes_P = Table.read(stokesP_table)
    ra = np.array(stokes_P["ra"].tolist())
    dec = np.array(stokes_P["dec"].tolist())
    polint = np.array(stokes_P["polint"].tolist())
    source_angular_distances = angular_distance(
        pointing_ra_deg, pointing_dec_deg, ra, dec
    )
    pos = np.where(source_angular_distances <= fov)
    P_total_flux = round(np.nansum(polint[pos]), 2)

    print(
        "Total Stokes I flux: "
        + str(I_total_flux)
        + " and total polarized flux: "
        + str(P_total_flux)
        + " Jy.\n"
    )
    total_P_frac = round(P_total_flux / I_total_flux, 2)
    print("Total polarization fracton: " + str(round(total_P_frac * 100, 2)) + "%.\n")
    return total_P_frac


def check_resource_availability(cpu_threshold=20, memory_threshold=20):
    """
    Check hardware resource availability
    Parameters
    ----------
    cpu_threshold : float
        Percentage of free CPU
    memory_threshold : float
        Percentage of free memory
    Returns
    -------
    bool
        Whether sufficient hardware resource is available or not
    """
    # Check CPU availability
    current_cpu_usage = psutil.cpu_percent(interval=1)
    cpu_available = current_cpu_usage < (100 - cpu_threshold)
    # Check Memory availability
    memory = psutil.virtual_memory()
    memory_available = memory.available / memory.total * 100
    memory_sufficient = memory_available > memory_threshold
    # Check Swap Memory availability
    # Check Swap availability
    swap = psutil.swap_memory()
    swap_available = (
        swap.free / swap.total * 100 if swap.total > 0 else 0
    )  # 0% if no swap is configured
    swap_sufficient = swap_available > memory_threshold
    return cpu_available and memory_sufficient and swap_sufficient


def wait_for_resources(basename, cpu_threshold=20, memory_threshold=20):
    """
    Wait for free hardware resources
    Parameters
    ----------
    basename : str
        Basename to search
    cpu_threshold : float
        Percentage of free CPU
    memory_threshold : float
        Percentage of free memory
    Returns
    -------
    int
        Number of free jobs
    """
    time.sleep(5)
    count = 0
    running_file_list = glob.glob(basename + "*")
    while True:
        resource_available = check_resource_availability(
            cpu_threshold=cpu_threshold, memory_threshold=memory_threshold
        )
        if resource_available:
            new_running_file_list = glob.glob(basename + "*")
            if len(running_file_list) == 0:
                gc.collect()
                return -1
            if len(new_running_file_list) < len(running_file_list):
                free_jobs = len(running_file_list) - len(new_running_file_list)
                gc.collect()
                return free_jobs
            else:
                if count == 0:
                    print("Waiting for free hardware resources ....\n")
                gc.collect()
                time.sleep(1)
        else:
            if count == 0:
                print("Waiting for free hardware resources ....\n")
            gc.collect()
            time.sleep(1)
        count += 1


def radec_to_altaz(ra, dec, obstime, LAT, LON, ALT):
    """
    Function to convert radec to altaz for a given Earth location
    Parameters
    ----------
    ra : str
            RA either in degree or 'hh:mm:ss' or '%fh%fm%fs' format
    dec : str
            DEC either in degree or 'dd:mm:ss' or '%fd%fm%fs'format
    obstime : str
            Time of the observation in 'yyyy-mm-dd hh:mm:ss' format
    LAT : float
            Latitude of the Earth location in degree
    LON : float
            Longitude of Earth location in degree
    ALT : float
            Altitude of the Earth location in meter
    Returns
    -------
    float
            Altitude in degree
    float
            Azimuth in degree
    """
    LOCATION = EarthLocation.from_geodetic(
        lat=LAT * u.deg, lon=LON * u.deg, height=ALT * u.m
    )
    observing_time = Time(obstime)
    aa = AltAz(location=LOCATION, obstime=observing_time)
    try:
        ra = float(ra)
        dec = float(dec)
        coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
    except:
        try:
            coord = SkyCoord(ra, dec)
        except:
            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    altaz_object = coord.transform_to(aa)
    alt = altaz_object.alt.degree
    az = altaz_object.az.degree
    return alt, az


def get_solar_coords(lat=None, lon=None, elev=None, obstime=""):
    """
    Get solar coordinates and elevation elevation at geographic location at a time
    Parameters
    ----------
    lat : float
        Latitude in degree (default: YAMAGAWA latitude)
    lon : float
        Longitude in degree (default: YAMAGAWA longitude)
    elev : float
        Elevation from mean sea lievel in meter (default: YAMAGAWA elevation)
    obstime : str
        Date and time in ISOT format
    Returns
    -------
    str
        Solar RA in h:m:s format
    str
        Solar DEC in d:m:s format
    float
        Solar altitude in degree
    float
        Solar azimuth in degree
    """
    if lat == None:
        lat = MWALAT
    if lon == None:
        lon = MWALON
    if elev == None:
        elev = MWAALT
    latitude = lat * u.deg  # In degree
    longitude = lon * u.deg  # In degree
    elevation = elev * u.m  # In meter
    if obstime == "":
        time = Time.now()
    else:
        time = Time(obstime)
    location = EarthLocation(lat=latitude, lon=longitude, height=elevation)
    sun_coords = get_sun(time)
    altaz_frame = AltAz(obstime=time, location=location)
    sun_altaz = sun_coords.transform_to(altaz_frame)
    solar_alt = sun_altaz.alt.deg
    solar_az = sun_altaz.az.deg
    sun_ra = sun_coords.ra.to_string(unit="hourangle", sep=":")
    sun_dec = sun_coords.dec.to_string(unit="deg", sep=":")
    return sun_ra, sun_dec, solar_alt, solar_az


def get_source_brightness(flux, alpha, freq, reffreq=150):
    """
    Return the flux density of the source at the specified frequency
    Parameters
    ----------
    flux_150 : float
        Flux density at 150 MHz
    alpha : float
        Spectral index
    freq : float
        Frequency in MHz to evaluate the model
    reffreq : float
        Reference frequency of the given flux density
    Returns
    -------
    float
        Flux density
    """
    appflux = flux * ((freq / reffreq) ** (alpha))
    return appflux


def get_quiet_sun_flux(freq):
    """
    Get quiet Sun flux density in Jy
    Parameters
    ----------
    freq : float
        Frequency in MHz
    Returns
    -------
    float
        Flux density in JY
    """
    p = np.poly1d([-1.93715165e-06, 7.84627718e-04, -3.15744433e-02, 2.32834400e-01])
    flux = p(freq) * 10**4  # Polynomial return in SFU
    return flux


def angular_distance(RA_ref, DEC_ref, RA_list, DEC_list):
    """
    Calculate angular distances from a reference
    Parameters
    -----------
    RA_ref : float
        Right Ascension of the reference point in degrees
    DEC_ref : float
        Declination of the reference point in degrees
    RA_list : list or np.ndarray
        List of Right Ascension values in degrees.
    DEC_list : list or np.ndarray
        List of Declination values in degrees.
    Returns
    -------
    list
        Angular distances in degrees.
    """
    ref_coord = SkyCoord(ra=RA_ref * u.deg, dec=DEC_ref * u.deg, frame="icrs")
    # Define the list of coordinates
    coord_list = SkyCoord(ra=RA_list * u.deg, dec=DEC_list * u.deg, frame="icrs")
    # Compute angular distances
    distances = ref_coord.separation(coord_list)
    return distances.deg  # Return distances in degrees


def get_ateam_sources(
    msname,
    metafits,
    MWA_PB_file,
    sweet_spot_file,
    min_beam_threshold=0.001,
    threshold_flux=0.0,
):
    """
    This function will give the list of A-team sources which are above the horizon
    Parameters
    ----------
    msname : str
        Name of the measurement name
    metafits : str
        Metafits name
    MWA_PB_file : str
        MWA primary beam file
    sweet_spot_file : str
        MWA sweetspot file
    min_beam_threshold : float
        Minimum beam gain
    threshold_flux : float
        Threshold flux in Jy
    Returns
    -------
    str
        List of A-team sources to peel
    """
    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0, unit="MHz")
    msmd.close()
    header = fits.getheader(metafits)
    pointing_RA = header["RA"]
    pointing_DEC = header["DEC"]
    obs_time = header["DATE-OBS"]
    beamsize = calc_field_of_view(msname, FWHM=True) / 7200.0 # HWHM
    pos = {}
    source_info = {}
    pos["CasA"] = ["23h23m24.0s", "+58d48m54.00s", 13000.0, -0.5, 150]
    pos["HerA"] = ["16h51m08.2s", "+04d59m33s", 520.0, -1.1, 150]
    pos["HydA"] = ["09h18m05.65s", "-12d05m43.99s", 350.0, -0.9, 150]
    pos["PicA"] = ["05h19m49.73s", "-45d46m43.70s", 570.0, -1.0, 150]
    pos["CenA"] = ["13h25m27.6s", "-43d01m09.00s", 1040.0, -0.65, 150]
    pos["CygA"] = ["19h59m28.35s", "+40d44m02.09s", 9000.0, -1.0, 150]
    pos["VirA"] = ["12h30m49.42s", "+12d23m28.04s", 1200.0, -1.0, 150]
    pos["Crab"] = ["05h34m31.94s", "+22d00m52.2s", 1500.0, -0.5, 150]
    pos["ForA"] = ["03h23m00.0s", "-37d12m00.0s", 605.0, -0.88, 150]
    pos["3C444"] = ["22h14m25.75s", "-17d01m36.29s", 85.0, -0.9, 150]
    pos["3C353"] = ["17h20m28.0s", "-00d58m46.0s", 680.0, -0.6, 150]
    pos["PKS2356-61"] = ["23h59m04.36s", "-60d54m59.41s", 123.8, -0.75, 145]
    pos["PKS2153-69"] = ["21h57m05.98s", "-69d41m23.68", 131.4, -0.56, 145]
    pos["PKS0408-65"] = ["04h08m20.0s", "-65d45m08.0s", 5.9, -0.85, 150]
    pos["PKS0410-75"] = ["04h10m48.0s", "-75d07m19.0s", 5.0, -0.85, 150]
    pos["3C33"] = ["01h08m52.85s", "+13d20m13.75s", 54.2, -0.33, 150]
    pos["3C161"] = ["06h27m10.09s", "-05d53m04.76s", 74.9, -0.37, 160]
    pos["PKS0442-28"] = ["04h44m37.70s", "-28d09m54.41s", 37.0, -0.85, 150]
    pos["3C409"] = ["20h14m27.6s", "+23d34m53s", 99.0, -0.57, 160]
    pos["PKS0351-27"] = ["03h51m35.7s", "-27d44m35s", 52.0, -1.09, 80]
    callist = list(pos.keys())
    for cal in callist:
        ra = pos[cal][0]
        dec = pos[cal][1]
        flux = pos[cal][2]
        alpha = pos[cal][3]
        reffreq = pos[cal][4]
        alt, az = radec_to_altaz(ra, dec, obs_time, MWALAT, MWALON, MWAALT)
        if alt > 0:
            radeg, decdeg = ra_dec_to_deg(ra, dec)
            distance_from_pointing_center = angular_distance(
                pointing_RA, pointing_DEC, radeg, decdeg
            )
            if distance_from_pointing_center > beamsize:
                result = get_pb_radec(
                    radeg,
                    decdeg,
                    freq,
                    metafits,
                    MWA_PB_file,
                    sweet_spot_file,
                    iau_order=False,
                )
                beam_I = result[2]
                appflux = (
                    get_source_brightness(flux, alpha, freq, reffreq=reffreq) * beam_I
                )
                if beam_I > min_beam_threshold and appflux > threshold_flux:
                    source_info[cal] = [ra, dec, alt, az, appflux]
    sun_ra, sun_dec, sun_alt, sun_az = get_solar_coords(
        lat=MWALAT, lon=MWALON, elev=MWAALT, obstime=obs_time
    )
    if sun_alt > 0:
        radeg, decdeg = ra_dec_to_deg(sun_ra, sun_dec)
        result = get_pb_radec(
            radeg, decdeg, freq, metafits, MWA_PB_file, sweet_spot_file, iau_order=False
        )
        beam_I = result[2]
        appflux = get_quiet_sun_flux(freq) * beam_I
        source_info["Sun"] = [sun_ra, sun_dec, sun_alt, sun_az, appflux]
    print("Total sources", len(source_info))
    return source_info
