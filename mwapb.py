import mwa_hyperbeam, os, astropy, numpy as np, datetime, time, skyfield.api as si, warnings, erfa, glob, traceback, gc
from astropy.io import fits
import astropy.wcs as pywcs
from astropy.time import Time
from numpy.linalg import inv
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from casatasks import exportfits, importfits
from optparse import OptionParser
from scipy.interpolate import RectBivariateSpline
from joblib import Parallel, delayed
from correct_ionosphere_warp import correct_warp

warnings.filterwarnings("ignore")
os.system("rm -rf casa*log")

"""
Code is written by Devojyoti Kansabanik, 25 Sep, 2024
"""
MWALON = 116.67
MWALAT = -26.7
MWAALT = 377.8
MWAPOS = EarthLocation.from_geodetic(
    lon="116:40:14.93", lat="-26:42:11.95", height=377.8
)
######################################################################


def get_azza_from_fits(filename, metafits="", verbose=False):
    """
    Get azimuith and zenith angle arrays from fits file
    Parameters
    ----------
    filename : str
            Name of the fits file
    metafits : str
            Metafits file name
    verbose : bool
            Verbose output or not
    Returns
    -------
    dict
            {'za_rad': theta,'astro_az_rad': phi}
    """
    try:
        f = fits.open(filename)
    except IOError as err:
        print("Unable to open %s for reading\n%s", filename, err)
        return None

    h = f[0].header
    f.close()
    wcs = pywcs.WCS(h)
    naxes = h["NAXIS"]

    if "HPX" in h["CTYPE1"]:
        print("Cannot deal with HPX coordinates")
        return None

    if "RA" not in h["CTYPE1"]:
        print("Coordinate 1 should be RA")
        return None
    if "DEC" not in h["CTYPE2"]:
        print("Coordinate 1 should be DEC")
        return None

    # construct the basic arrays
    x = np.arange(1, h["NAXIS1"] + 1)
    y = np.arange(1, h["NAXIS2"] + 1)
    # assume we want the first frequency
    # if we have a cube this will have to change
    ff = 1
    Y, X = np.meshgrid(y, x)
    Xflat = X.flatten()
    Yflat = Y.flatten()
    FF = ff * np.ones(Xflat.shape)
    if naxes >= 4:
        Tostack = [Xflat, Yflat, FF]
        for i in range(3, naxes):
            Tostack.append(np.ones(Xflat.shape))
    else:
        Tostack = [Xflat, Yflat]
    pixcrd = np.vstack(Tostack).transpose()
    try:
        sky = wcs.wcs_pix2world(pixcrd, 1)
    except Exception as e:
        print("Problem converting to WCS: %s" % e)
        return None
    # extract the important pieces
    ra = sky[:, 0]
    dec = sky[:, 1]
    # and make them back into arrays
    RA = ra.reshape(X.shape)
    Dec = dec.reshape(Y.shape)
    # get the date so we can convert to Az,El
    if os.path.exists(metafits) == False:
        try:
            d = h["DATE-OBS"]
        except Exception:
            print("Unable to read observation date DATE-OBS from %s" % filename)
            return None
    else:
        metadata = fits.getheader(metafits)
        d = metadata["DATE-OBS"]
    if "." in d:
        d = d.split(".")[0]
    dt = datetime.datetime.strptime(d, "%Y-%m-%dT%H:%M:%S")
    mwatime = Time(dt)
    if verbose:
        print(
            "Image time = "
            + str(mwatime.iso)
            + ", GPS time = "
            + str(mwatime.gps)
            + "\n"
        )

    source = SkyCoord(
        ra=RA, dec=Dec, frame="icrs", unit=(astropy.units.deg, astropy.units.deg)
    )
    source.location = MWAPOS
    source.obstime = mwatime
    s = time.time()
    source_altaz = source.transform_to("altaz")
    Alt, Az = (
        source_altaz.alt.deg,
        source_altaz.az.deg,
    )  # Transform to Topocentric Alt/Az at the current epoch
    # go from altitude to zenith angle
    theta = (90 - Alt) * np.pi / 180
    phi = Az * np.pi / 180
    return {"za_rad": theta.transpose(), "astro_az_rad": phi.transpose()}


def get_IQUV(filename, stokesaxis=4):
    """Get IQUV from a CASA image file exported to fits."""
    data = fits.getdata(filename)
    stokes = {}
    if stokesaxis == 3:
        stokes["I"] = data[0, 0, :, :]
        stokes["Q"] = data[0, 1, :, :]
        stokes["U"] = data[0, 2, :, :]
        stokes["V"] = data[0, 3, :, :]
    else:
        stokes["I"] = data[0, 0, :, :]
        stokes["Q"] = data[1, 0, :, :]
        stokes["U"] = data[2, 0, :, :]
        stokes["V"] = data[3, 0, :, :]
    return stokes


def get_inst_pols(stokes):
    """Return instrumental polaristations matrix (Vij)"""
    XX = stokes["I"] + stokes["Q"]
    XY = stokes["U"] + stokes["V"] * 1j
    YX = stokes["U"] - stokes["V"] * 1j
    YY = stokes["I"] - stokes["Q"]
    Vij = np.array([[XX, XY], [YX, YY]])
    Vij = np.swapaxes(np.swapaxes(Vij, 0, 2), 1, 3)
    return Vij


def B2IQUV(B, iau_order=False):
    """
    Convert sky brightness matrix to I, Q, U, V
    Parameters
    ----------
    B : numpy.array
            Brightness matrix array
    iau_order : bool
            Whether brightness matrix is in IAU or MWA convention
    Returns
    -------
    dict
            Stokes dictionary
    """
    B11 = B[:, :, 0, 0]
    B12 = B[:, :, 0, 1]
    B21 = B[:, :, 1, 0]
    B22 = B[:, :, 1, 1]
    if iau_order:
        stokes = {}
        stokes["I"] = (B11 + B22) / 2.0
        stokes["Q"] = (B11 - B22) / 2.0
        stokes["U"] = (B12 + B21) / 2.0
        stokes["V"] = 1j * (B21 - B12) / 2.0
    else:
        stokes = {}
        stokes["I"] = (B11 + B22) / 2.0
        stokes["Q"] = (B22 - B11) / 2.0
        stokes["U"] = (B21 + B12) / 2.0
        stokes["V"] = 1j * (B12 - B21) / 2.0
    return stokes


def all_sky_beam_interpolator(
    MWA_PB_file, sweet_spot_file, sweet_spot_num, freq, resolution, iau_order
):
    """
    Calculate all sky beam interpolation for given sweet spot pointing
    Parameters
    ----------
    MWA_PB_file : str
        MWA primary beam file
    sweet_spot_file : str
        MWA sweet spot file name
    sweet_spot_num : int
        Sweet spot number
    freq : float
        Frequency in MHz
    resolution : float
        Spatial resolution in degree
    iau_order : bool
        PB Jones in IAU order or not
    Returns
    -------
    numpy.array
        All sky primary beam Jones array
    """
    os.environ["MWA_BEAM_FILE"] = MWA_PB_file
    beam = mwa_hyperbeam.FEEBeam()
    az_scale = np.arange(0, 360, resolution)
    alt_scale = np.arange(0, 90, resolution)
    az, alt = np.meshgrid(az_scale, alt_scale)
    za_rad = np.deg2rad(90 - alt.ravel())  # Zenith angle in radian
    az_rad = np.deg2rad(az.ravel())  # Azimuth in radian
    sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
    delay = sweet_spots[int(sweet_spot_num)][-1]
    ##############################################
    # Calculating Jones array in 1 deg alt-az grid
    ##############################################
    jones = beam.calc_jones_array(
        az_rad,
        za_rad,
        freq,
        delay,
        [1] * 16,
        True,
        np.deg2rad(MWALAT),
        iau_order,
    )
    jones = jones.swapaxes(0, 1).reshape(4, alt_scale.shape[0], az_scale.shape[0])
    j00_r = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.real(jones[0, ...]))
    )
    j00_i = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.imag(jones[0, ...]))
    )
    j01_r = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.real(jones[1, ...]))
    )
    j01_i = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.imag(jones[1, ...]))
    )
    j10_r = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.real(jones[2, ...]))
    )
    j10_i = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.imag(jones[2, ...]))
    )
    j11_r = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.real(jones[3, ...]))
    )
    j11_i = RectBivariateSpline(
        x=alt_scale, y=az_scale, z=np.nan_to_num(np.imag(jones[3, ...]))
    )
    return j00_r, j00_i, j01_r, j01_i, j10_r, j10_i, j11_r, j11_i


def get_jones_array(
    alt_arr,
    az_arr,
    freq,
    MWA_PB_file,
    sweet_spot_file,
    gridpoint,
    iau_order,
    interpolated,
    verbose,
):
    """
    Get primary beam jones matrix
    Parameters
    ----------
    alt_arr : numpy.array
        Flattened altitude array in degrees
    az_arr : numpy.array
        Flattened azimuth array in degrees
    MWA_PB_file : str
        Primary beam file name
    sweet_spot_file : str
        MWA sweet spot file name
    gridpoint : int
        Gridpoint number
    iau_order : bool
        IAU order of the beam
    interpolated : bool
        Use spatially interpolated beams or not
    Returns
    -------
    numpy.array
        Jones array (shape : coordinate_arr_shape, 2 ,2)
    """
    if interpolated:
        if verbose:
            print("Using interpolated beam ....\n")
        s = time.time()
        j00_r, j00_i, j01_r, j01_i, j10_r, j10_i, j11_r, j11_i = (
            all_sky_beam_interpolator(
                MWA_PB_file, sweet_spot_file, int(gridpoint), freq, 1.0, iau_order
            )
        )
        # Change resolution based on frequency
        s = time.time()
        with Parallel(n_jobs=8) as parallel:
            results = parallel(
                [
                    delayed(j00_r)(alt_arr, az_arr, grid=False),
                    delayed(j00_i)(alt_arr, az_arr, grid=False),
                    delayed(j01_r)(alt_arr, az_arr, grid=False),
                    delayed(j01_i)(alt_arr, az_arr, grid=False),
                    delayed(j10_r)(alt_arr, az_arr, grid=False),
                    delayed(j10_i)(alt_arr, az_arr, grid=False),
                    delayed(j11_r)(alt_arr, az_arr, grid=False),
                    delayed(j11_i)(alt_arr, az_arr, grid=False),
                ]
            )
        del parallel       
        (
            j00_r_arr,
            j00_i_arr,
            j01_r_arr,
            j01_i_arr,
            j10_r_arr,
            j10_i_arr,
            j11_r_arr,
            j11_i_arr,
        ) = results
        j00 = j00_r_arr + 1j * j00_i_arr
        j01 = j01_r_arr + 1j * j01_i_arr
        j10 = j10_r_arr + 1j * j10_i_arr
        j11 = j11_r_arr + 1j * j11_i_arr
        j00 = j00.reshape(az_arr.shape)
        j01 = j01.reshape(az_arr.shape)
        j10 = j10.reshape(az_arr.shape)
        j11 = j11.reshape(az_arr.shape)
        jones_array = np.array([j00, j01, j10, j11]).T
    else:
        beam = mwa_hyperbeam.FEEBeam(MWA_PB_file)
        sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
        delay = sweet_spots[int(gridpoint)][-1]
        za_arr = 90 - alt_arr
        jones_array = beam.calc_jones_array(
            np.deg2rad(az_arr),
            np.deg2rad(za_arr),
            freq,
            delay,
            [1] * 16,
            True,
            np.deg2rad(MWALAT),
            iau_order,
        )
    jones_array = jones_array.reshape(jones_array.shape[0], 2, 2)
    gc.collect()
    return jones_array


def mwapb_cor(
    imagename,
    outfile,
    MWA_PB_file,
    sweet_spot_file,
    metafits="",
    iau_order=True,
    pb_jones_file="",
    save_pb_file="",
    differential_pb=False,
    interpolated=True,
    verbose=False,
    gridpoint=-1,
    nthreads=1,
    conv="iau",
    restore=False,
    MWA_PB_path="",
    output_stokes="",
):
    """
    Correct FITS or CASA image for MWA primary beam
    Parameters
    ----------
    imagename : str
            Name of the input file
    outfile : str
            Basename of the outputfile
    metafits : str
            Metafits file path
    iau_order : bool
            PB Jones in IAU order or not
    pb_jones_file : str
            Numpy file with primary beam jones matrices
    save_pb_file : str
            Save primary beam jones matrices for future use in this file
    differential_pb : bool
            Correct using differential primary beam
    interpolated : bool
        Calculate spatially interpolated beams or not
    verbose : bool
            Verbose output or not
    gridpoint : int
            MWA gridpoint number (default : -1, provide if you do not have metafits file)
    nthreads : int
            Number of cpu threads use for parallel computing
    conv : str
            Image coordinate convention (default : \'iau\') (options : \'iau\' and \'mwa\')
    restore : bool
            Whether correct for MWA primary beam or restore the correction
    MWA_PB_path : str
            MWA primary beam data file path (default : In built datafile from P-AIRCARS)
    output_stokes : str
            Output Stokes planes ('I' or 'IQUV', default : input image stokes)
    Returns
    -------
    str
            Output image name
    """
    os.environ["MWA_BEAM_FILE"] = MWA_PB_file
    beam = mwa_hyperbeam.FEEBeam()
    outfile = os.path.basename(outfile)
    if imagename[-1] == "/":
        imagename = imagename[:-1]
    try:
        image_header = fits.getheader(imagename)
        naxes = int(image_header["NAXIS"])
        stokesaxis = 4
        if naxes >= 4:
            if "STOKES" not in image_header["CTYPE4"]:
                stokesaxis = 3
                if "STOKES" not in image_header["CTYPE3"]:
                    print("Coordinate 3 or 4 should be STOKES")
                    return 1
        else:
            print("Number of axes is not 4.\n")
            return 1
        if stokesaxis == 4:
            if int(image_header["NAXIS4"]) != 4:
                if int(image_header["CRVAL4"]) != 1:
                    print("1.Stokes axes are either not 'I' or 'IQUV'.\n")
                    return 1
                else:
                    data = fits.getdata(imagename)
                    header = fits.getheader(imagename)
                    header["NAXIS4"] = 4.0
                    header["CRVAL4"] = 1.0
                    header["CDELT4"] = 1.0
                    data = np.repeat(data, 4, 0)
                    data[1, ...] *= 0
                    data[2, ...] *= 0
                    data[3, ...] *= 0
                    fits.writeto(
                        imagename + ".allStokes",
                        data=data,
                        header=header,
                        overwrite=True,
                    )
                    fitsfile = imagename + ".allStokes"
                    org_stokes = "I"
            else:
                org_stokes = "IQUV"
                fitsfile = imagename
            imagetype = "fits"
        else:
            if int(image_header["NAXIS3"]) != 4:
                if int(image_header["CRVAL3"]) != 1:
                    print("Stokes axes are either not 'I' or 'IQUV'.\n")
                    return 1
                else:
                    data = fits.getdata(imagename)
                    header = fits.getheader(imagename)
                    header["NAXIS3"] = 4.0
                    header["CRVAL3"] = 1.0
                    header["CDELT3"] = 1.0
                    data = np.repeat(data, 4, 1)
                    data[:, 1, ...] *= 0
                    data[:, 2, ...] *= 0
                    data[:, 3, ...] *= 0
                    fits.writeto(
                        imagename + ".allStokes",
                        data=data,
                        header=header,
                        overwrite=True,
                    )
                    fitsfile = imagename + ".allStokes"
                    org_stokes = "I"
            else:
                org_stokes = "IQUV"
            imagetype = "fits"
    except Exception as e:
        try:
            if imagename[-1] == "/":
                imagename = imagename[:-1]
            if os.path.exists(imagename + ".fits"):
                os.system("rm -rf " + imagename + ".fits")
            exportfits(
                imagename=imagename, fitsimage=imagename + ".fits", stokeslast=False
            )
            fitsfile = imagename + ".fits"
            image_header = fits.getheader(fitsfile)
            naxes = int(image_header["NAXIS"])
            stokesaxis = 4
            if naxes >= 4:
                if "STOKES" not in image_header["CTYPE4"]:
                    stokesaxis = 3
                    if "STOKES" not in image_header["CTYPE3"]:
                        print("Coordinate 3 or 4 should be STOKES")
                        return 1
            else:
                print("Number of axes is not 4.\n")
                return 1
            if stokesaxis == 4:
                if int(image_header["NAXIS4"]) != 4:
                    if int(image_header["CRVAL4"]) != 1:
                        print("1.Stokes axes are either not 'I' or 'IQUV'.\n")
                        return 1
                    else:
                        data = fits.getdata(fitsfile)
                        header = fits.getheader(fitsfile)
                        header["NAXIS4"] = 4.0
                        header["CRVAL4"] = 1.0
                        header["CDELT4"] = 1.0
                        data = np.repeat(data, 4, 0)
                        data[1, ...] *= 0
                        data[2, ...] *= 0
                        data[3, ...] *= 0
                        fits.writeto(
                            fitsfile + ".allStokes",
                            data=data,
                            header=header,
                            overwrite=True,
                        )
                        fitsfile = fitsfile + ".allStokes"
                        org_stokes = "I"
                else:
                    org_stokes = "IQUV"
                    fitsfile = fitsfile
                imagetype = "CASA"
            else:
                if int(image_header["NAXIS3"]) != 4:
                    if int(image_header["CRVAL3"]) != 1:
                        print("Stokes axes are either not 'I' or 'IQUV'.\n")
                        return 1
                    else:
                        data = fits.getdata(fitsfile)
                        header = fits.getheader(fitsfile)
                        header["NAXIS3"] = 4.0
                        header["CRVAL3"] = 1.0
                        header["CDELT3"] = 1.0
                        data = np.repeat(data, 4, 1)
                        data[:, 1, ...] *= 0
                        data[:, 2, ...] *= 0
                        data[:, 3, ...] *= 0
                        fits.writeto(
                            fitsfile + ".allStokes",
                            data=data,
                            header=header,
                            overwrite=True,
                        )
                        fitsfile = fitsfile + ".allStokes"
                        org_stokes = "I"
                else:
                    org_stokes = "IQUV"
            imagetype = "CASA"
        except Exception as e:
            print("Image is not in fits or CASA image format.\n")
            traceback.print_exc()
            gc.collect()
            return 1
    if verbose:
        if restore == False:
            print(
                "Correcting image : "
                + os.path.basename(imagename)
                + " for MWA primary beam response......\n"
            )
        else:
            print(
                "Undo the correction image : "
                + os.path.basename(imagename)
                + " for MWA primary beam response......\n"
            )
    naxes = image_header["NAXIS"]
    freqaxis = 3
    stokesaxis = 4
    if naxes >= 4:
        if "FREQ" not in image_header["CTYPE3"]:
            freqaxis = 4
            stokesaxis = 3
            if "FREQ" not in image_header["CTYPE4"]:
                print("Coordinate 3 or 4 should be FREQ")
                return 1
        if freqaxis == 3:
            freq = image_header["CRVAL3"]  # read number of frequency channels
        else:
            freq = image_header["CRVAL4"]
    else:
        print("Number of axes is not 4.\n")
        return 1
    if metafits == None or os.path.isfile(metafits) == False:
        if gridpoint == -1:
            print("Either provide correct metafits file or grid point number.\n")
            gc.collect()
            return 1
        else:
            sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
            delay = sweet_spots[int(gridpoint)][-1]
    else:
        metadata = fits.getheader(metafits)
        gridpoint = metadata["GRIDNUM"]
        sweet_spots = np.load(sweet_spot_file, allow_pickle=True).all()
        delay = sweet_spots[int(gridpoint)][-1]
    if nthreads <= 0:
        nthreads = 1
    os.environ["RAYON_NUM_THREADS"] = str(nthreads)
    if verbose:
        print("Calculating azimuith and zenith angles...\n")
    alt_az_array = get_azza_from_fits(fitsfile, metafits=metafits, verbose=verbose)
    if pb_jones_file == "" or os.path.exists(pb_jones_file) == False:
        if differential_pb:
            if verbose:
                print("Calculating MWA differential primary beam...\n")
            jones_array = get_jones_array(
                90 - np.rad2deg(alt_az_array["za_rad"].flatten()),
                np.rad2deg(alt_az_array["astro_az_rad"].flatten()),
                freq,
                MWA_PB_file,
                sweet_spot_file,
                gridpoint,
                iau_order,
                interpolated,
                verbose,
            )
            image_header = fits.getheader(fitsfile)
            radeg = image_header["CRVAL1"]
            decdeg = image_header["CRVAL2"]
            msg, phasecenter_jones_array, I_beam, XX_power_beam, YY_power_beam_array = (
                get_pb_radec(radeg, decdeg, freq / 10**6, metafits=metafits)
            )
            phasecenter_jones_array = phasecenter_jones_array[np.newaxis, ...]
            phasecenter_jones_array = np.repeat(
                phasecenter_jones_array, jones_array.shape[0], axis=0
            )
            phasecenter_jones_array = phasecenter_jones_array.reshape(
                phasecenter_jones_array.shape[0], 2, 2
            )
            diff_jones_array = np.matmul(inv(phasecenter_jones_array), jones_array)
            del jones_array
            jones_array = diff_jones_array
        else:
            if verbose:
                print("Calculating MWA primary beam...\n")
            jones_array = get_jones_array(
                90 - np.rad2deg(alt_az_array["za_rad"].flatten()),
                np.rad2deg(alt_az_array["astro_az_rad"].flatten()),
                freq,
                MWA_PB_file,
                sweet_spot_file,
                gridpoint,
                iau_order,
                interpolated,
                verbose,
            )
        if save_pb_file != "":
            if verbose:
                print(
                    "Saving primary beam Jones matrices in : "
                    + str(save_pb_file)
                    + ".npy\n"
                )
            np.save(save_pb_file, np.array([iau_order, jones_array], dtype="object"))
    elif pb_jones_file != "" and os.path.exists(pb_jones_file):
        if verbose:
            print("Loading primary beam Jones matrices from : " + pb_jones_file + "\n")
        pb = np.load(pb_jones_file, allow_pickle=True)
        pb_jones_order = pb[0]
        jones_array = pb[1]
        expected_shape = (alt_az_array["astro_az_rad"].flatten().shape[0], 2, 2)
        if jones_array.shape != expected_shape or pb_jones_order != iau_order:
            if verbose:
                if pb_jones_order != iau_order:
                    print(
                        "Given primary beam convention does not match with intented convention. Re-estimating primary beam Jones\n"
                    )
                else:
                    print(
                        "Loaded primary beam Jones are of different shape. Re-estimating primary beam Jones.\n"
                    )
            if differential_pb:
                if verbose:
                    print("Calculating MWA differential primary beam...\n")
                jones_array = get_jones_array(
                    90 - np.rad2deg(alt_az_array["za_rad"].flatten()),
                    np.rad2deg(alt_az_array["astro_az_rad"].flatten()),
                    freq,
                    MWA_PB_file,
                    sweet_spot_file,
                    gridpoint,
                    iau_order,
                    interpolated,
                    verbose,
                )
                image_header = fits.getheader(fitsfile)
                radeg = image_header["CRVAL1"]
                decdeg = image_header["CRVAL2"]
                (
                    msg,
                    phasecenter_jones_array,
                    I_beam,
                    XX_power_beam,
                    YY_power_beam_array,
                ) = get_pb_radec(radeg, decdeg, freq / 10**6, metafits=metafits)
                phasecenter_jones_array = phasecenter_jones_array[np.newaxis, ...]
                phasecenter_jones_array = np.repeat(
                    phasecenter_jones_array, jones_array.shape[0], axis=0
                )
                phasecenter_jones_array = phasecenter_jones_array.reshape(
                    phasecenter_jones_array.shape[0], 2, 2
                )
                diff_jones_array = np.matmul(inv(phasecenter_jones_array), jones_array)
                del jones_array
                jones_array = diff_jones_array
            else:
                if verbose:
                    print("Calculating MWA primary beam...\n")
                jones_array = get_jones_array(
                    90 - np.rad2deg(alt_az_array["za_rad"].flatten()),
                    np.rad2deg(alt_az_array["astro_az_rad"].flatten()),
                    freq,
                    MWA_PB_file,
                    sweet_spot_file,
                    gridpoint,
                    iau_order,
                    interpolated,
                    verbose,
                )
    if verbose:
        print("Correcting image using primary beam....\n")
    imagedata = fits.getdata(fitsfile)
    nanpos = np.where(np.isnan(imagedata) == True)
    zeropos = np.where(imagedata == 0)
    if stokesaxis == 4:
        imagedata_swaped = np.swapaxes(np.swapaxes(imagedata, 0, 2), 1, 3)
    else:
        imagedata_swaped = np.swapaxes(
            np.swapaxes(np.swapaxes(imagedata, 0, 2), 1, 3), 2, 3
        )
    if stokesaxis == 3:
        imagedata_reshaped = imagedata_swaped[:, :, :, 0].reshape(
            imagedata.shape[-1] * imagedata.shape[-1], 2, 2
        )
    else:
        imagedata_reshaped = imagedata_swaped[:, :, :, 0].reshape(
            imagedata.shape[-1] * imagedata.shape[-2], 2, 2
        )

    stokes = get_IQUV(fitsfile, stokesaxis=stokesaxis)
    Vij = get_inst_pols(stokes)
    Vij_reshaped = Vij.reshape(Vij.shape[0] * Vij.shape[1], 2, 2)
    jones_array_H = np.transpose(jones_array.conj(), axes=((0, 2, 1)))
    if eval(str(restore)) == False:
        Vij_corrected = np.matmul(
            inv(jones_array), np.matmul(Vij_reshaped, inv(jones_array_H))
        )
    else:
        Vij_corrected = np.matmul(jones_array, np.matmul(Vij_reshaped, jones_array_H))
    Vij_reshaped = Vij_corrected.reshape(Vij.shape)
    B = B2IQUV(Vij_reshaped, iau_order=iau_order)
    if stokesaxis == 3:
        imagedata[0, 0, :, :] = np.real(B["I"])
        imagedata[0, 1, :, :] = np.real(B["Q"])
        imagedata[0, 2, :, :] = np.real(B["U"])
        imagedata[0, 3, :, :] = np.real(B["V"])
    else:
        imagedata[0, 0, :, :] = np.real(B["I"])
        imagedata[1, 0, :, :] = np.real(B["Q"])
        imagedata[2, 0, :, :] = np.real(B["U"])
        imagedata[3, 0, :, :] = np.real(B["V"])

    output_fitsfile = (
        os.path.dirname(os.path.abspath(imagename)) + "/" + outfile + ".fits"
    )
    if os.path.exists(output_fitsfile):
        os.system("rm -rf " + output_fitsfile)
    header_keys = image_header.keys()
    if (
        "BMAJ" not in header_keys
        or "BMIN" not in header_keys
        or "BPA" not in header_keys
    ):
        try:
            f = fits.open(fitsfile)[1].data[0]
            maj = f[0]
            minor = f[1]
            pa = f[2]
            image_header["BMAJ"] = maj
            image_header["BMIN"] = minor
            image_header["BPA"] = pa
            has_beam = True
        except:
            has_beam = False
    else:
        maj = image_header["BMAJ"] * 3600
        minor = image_header["BMIN"] * 3600
        pa = image_header["BPA"]
        has_beam = True

    fits.writeto(output_fitsfile, data=imagedata, header=image_header, overwrite=True)
    if imagetype == "CASA":
        image_extension = imagename.split(".")[-1]
        if os.path.exists(
            os.path.dirname(os.path.abspath(imagename))
            + "/"
            + outfile
            + "."
            + image_extension
        ):
            os.system(
                "rm -rf "
                + os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension
            )
        if has_beam:
            print(
                "importfits(fitsimage='"
                + output_fitsfile
                + "',imagename='"
                + os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension
                + "',beam=["
                + str(maj)
                + "'arcsec',"
                + str(minor)
                + "'arcsec',"
                + str(pa)
                + "'deg'])\n"
            )
            importfits(
                fitsimage=output_fitsfile,
                imagename=os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension,
                beam=[str(maj) + "arcsec", str(minor) + "arcsec", str(pa) + "deg"],
            )
        else:
            importfits(
                fitsimage=output_fitsfile,
                imagename=os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension,
            )
        os.system("rm -rf " + output_fitsfile)
        if (output_stokes == "" and org_stokes == "I") or output_stokes == "I":
            if os.path.exists(
                os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension
                + ".StokesI"
            ):
                os.system(
                    "rm -rf "
                    + os.path.dirname(os.path.abspath(imagename))
                    + "/"
                    + outfile
                    + "."
                    + image_extension
                    + ".StokesI"
                )
            imsubimage(
                imagename=os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension,
                outfile=os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension
                + ".StokesI",
                stokes="I",
            )
            if os.path.exists(
                os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension
            ):
                os.system(
                    "rm -rf "
                    + os.path.dirname(os.path.abspath(imagename))
                    + "/"
                    + outfile
                    + "."
                    + image_extension
                )
            os.system(
                "mv "
                + os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension
                + ".StokesI "
                + os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension
            )
        os.system("rm -rf " + output_fitsfile + " " + fitsfile + " casa*log")
        if verbose:
            print(
                "Output image written to : "
                + os.path.dirname(os.path.abspath(imagename))
                + "/"
                + outfile
                + "."
                + image_extension
                + "\n"
            )
        gc.collect()
        return (
            os.path.dirname(os.path.abspath(imagename))
            + "/"
            + outfile
            + "."
            + image_extension
        )
    else:
        if (output_stokes == "" and org_stokes == "I") or output_stokes == "I":
            data = fits.getdata(output_fitsfile)
            fits.writeto(
                output_fitsfile, data=data[0, ...], header=image_header, overwrite=True
            )
        if verbose:
            print("Output image written to : " + output_fitsfile + "\n")
        os.system("rm -rf casa*log")
        gc.collect()
        return output_fitsfile


def main():
    usage = "Correct images for MWA primary beam response"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--imagename",
        dest="imagename",
        default=None,
        help="Name of the image file",
        metavar="String",
    )
    parser.add_option(
        "--metafits",
        dest="metafits",
        default=None,
        help="Name of the metafits file",
        metavar="String",
    )
    parser.add_option(
        "--MWA_PB_file",
        dest="MWA_PB_file",
        default=None,
        help="MWA primary beam file",
        metavar="File path",
    )
    parser.add_option(
        "--sweetspot_file",
        dest="sweetspot_file",
        default=None,
        help="MWA primary beam sweetspot file path",
        metavar="File path",
    )
    parser.add_option(
        "--IAU_order",
        dest="iau_order",
        default=True,
        help="PB Jones in IAU order or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--gridpoint",
        dest="gridpoint",
        default=-1,
        help="MWA sweet spot pointing number",
        metavar="Integer",
    )
    parser.add_option(
        "--num_threads",
        dest="nthreads",
        default=1,
        help="Numbers of CPU threads to use",
        metavar="Integer",
    )
    parser.add_option(
        "--image_conv",
        dest="conv",
        default="iau",
        help="Image convension",
        metavar="String",
    )
    parser.add_option(
        "--outfile",
        dest="outfile",
        default="MWAPB_cor",
        help="Output file basename",
        metavar="String",
    )
    parser.add_option(
        "--output_stokes",
        dest="output_stokes",
        default="",
        help="Output Stokes",
        metavar="String",
    )
    parser.add_option(
        "--restore_correction",
        dest="restore",
        default=False,
        help="Correct for primary beam or restore the correction",
        metavar="Boolean",
    )
    parser.add_option(
        "--pb_jones_file",
        dest="pb_jones_file",
        default="",
        help="Numpy file of primary beam Jones matrices",
        metavar="Boolean",
    )
    parser.add_option(
        "--save_pb",
        dest="save_pb",
        default="",
        help="Save primary beam Jones matrices in this file",
        metavar="Boolean",
    )
    parser.add_option(
        "--differential_pb",
        dest="differential_pb",
        default=False,
        help="Differential primary beam",
        metavar="Boolean",
    )
    parser.add_option(
        "--interpolated",
        dest="interpolated",
        default=True,
        help="Interpolated beam",
        metavar="Boolean",
    )
    parser.add_option(
        "--warp_cat",
        dest="warp_cat",
        default=None,
        help="Ionosphere warp catalog",
        metavar="String",
    )
    parser.add_option(
        "--verbose",
        dest="verbose",
        default=False,
        help="Verbose output or not",
        metavar="Boolean",
    )
    (options, args) = parser.parse_args()
    nthreads = int(options.nthreads)
    start_time = time.time()
    try:
        pbcor_image = mwapb_cor(
            str(options.imagename),
            str(options.outfile),
            str(options.MWA_PB_file),
            str(options.sweetspot_file),
            metafits=options.metafits,
            iau_order=eval(str(options.iau_order)),
            pb_jones_file=options.pb_jones_file,
            verbose=eval(str(options.verbose)),
            gridpoint=int(options.gridpoint),
            nthreads=nthreads,
            conv=str(options.conv),
            restore=eval(str(options.restore)),
            save_pb_file=options.save_pb,
            differential_pb=eval(str(options.differential_pb)),
            interpolated=eval(str(options.interpolated)),
            output_stokes=options.output_stokes,
        )
        if options.warp_cat != None and os.path.exists(options.warp_cat):
            print(
                "Ionospheric warp correction using: "
                + os.path.basename(options.warp_cat)
                + "\n"
            )
            pbcor_image = correct_warp(
                pbcor_image, options.warp_cat, ncpu=int(nthreads), keep_original=False
            )
            header = fits.getheader(pbcor_image)
            data = fits.getdata(pbcor_image)
            header["UNWARP"] = "Y"
            fits.writeto(pbcor_image, data, header, overwrite=True)
        if eval(str(options.verbose)):
            print("Total time: " + str(round(time.time() - start_time, 2)) + "s.\n")
        gc.collect()
        return 0
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        if eval(str(options.verbose)):
            print("Total time: " + str(round(time.time() - start_time, 2)) + "s.\n")
        return 1


if __name__ == "__main__":
    result = main()
    os._exit(result)
