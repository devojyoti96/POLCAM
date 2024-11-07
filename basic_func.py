from casatools import table, msmetadata
import numpy as np


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


def MWA_field_of_view(msname, FWHM=True):
    """
    Calculate optimum field of view in arcsec
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
    msmd.open(msname)
    freq = msmd.meanfreq(0)
    msmd.close()
    if FWHM == True:
        FOV = (
            np.sqrt(610) * 150 * 10**6 / freq
        )  # 600 deg^2 is the image FoV at 150MHz for MWA. So extrapolating this to central frequency
    else:
        FOV = (
            60 * 110 * 10**6 / freq
        )  # 3600 deg^2 is the image FoV at 110MHz for MWA upto first null. So extrapolating this to central frequency
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
    psf = np.rad2deg(1.2 / maxuv_l) * 3600.0  # In arcsec
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
    pixel = int(psf / num_pixel_in_psf)
    return pixel


def calc_imsize(msname, num_pixel_in_psf):
    """
    Calculate image pixel size
    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : int
            Number of pixels in one PSF
    Returns
    -------
    int
            Number of pixels
    """
    cellsize = calc_cellsize(msname, num_pixel_in_psf)
    fov = MWA_field_of_view(msname, FWHM=True)
    imsize = int(fov / cellsize)
    pow2 = round(np.log2(imsize / 10.0), 0)
    imsize = int((2**pow2) * 10)
    if imsize > 8192:
        imsize = 8192
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
    freq = msmd.meanfreq(0)
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
