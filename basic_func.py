from casatools import table, msmetadata
from casatasks import importfits, exportfits
from astropy.io import fits
from astropy.table import Table
import numpy as np, os, psutil, time, glob, gc, scipy, copy

os.system("rm -rf casa*log")


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
    pixel = int(psf / num_pixel_in_psf)
    return pixel


def calc_imsize(msname, num_pixel_in_psf, FWHM=True):
    """
    Calculate image pixel size
    Parameters
    ----------
    msname : str
        Name of the measurement set
    num_pixel_in_psf : int
            Number of pixels in one PSF
    FWHM : bool
        Image upto FWHM or first null
    Returns
    -------
    int
            Number of pixels
    """
    cellsize = calc_cellsize(msname, num_pixel_in_psf)
    fov = MWA_field_of_view(msname, FWHM=FWHM)
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
    minuv_m = 112
    maxuv_m = 2500
    minuv_l = round(minuv_m / wavelength, 1)
    maxuv_l = round(maxuv_m / wavelength, 1)
    uvrange = str(minuv_l) + "~" + str(maxuv_l) + "lambda"
    return uvrange


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
    finished_touch_file = basedir + "/.Finished_" + basename
    os.system("rm -rf " + finished_touch_file + "*")
    finished_touch_file_error = finished_touch_file + "_error"
    finished_touch_file_success = finished_touch_file + "_0"
    cmd_file_content = f"""{cmd}\nsleep 2 \nexit_code=$?\nif [ $? -ne 0 ]\nthen touch {finished_touch_file_error}\nelse touch {finished_touch_file_success}\nfi"""
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

def make_leakage_surface(dataq,datai):
    """
    Make leakage surface
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
    # Create the design matrix for the least squares fit
    q_stack = np.column_stack((x, y, z))
    AQ = np.c_[
        np.ones(q_stack.shape[0]),
        q_stack[:, :2],
        np.prod(q_stack[:, :2], axis=1),
        q_stack[:, :2] ** 2
    ]
    # Solve for coefficients using least squares
    CQ, _, _, _ = scipy.linalg.lstsq(AQ, q_stack[:, 2])
    # Generate the surface values
    k_indices, l_indices = np.meshgrid(range(dataq.shape[0]), range(dataq.shape[1]), indexing="ij")
    data_backup = (
        CQ[4] * k_indices**2 +
        CQ[5] * l_indices**2 +
        CQ[3] * k_indices * l_indices +
        CQ[1] * k_indices +
        CQ[2] * l_indices +
        CQ[0]
    )
    return data_backup
	
def cal_field_averaged_polfrac(pointing_ra_deg,pointing_dec_deg,fov=30,ggsm_table='GGSM.fits',pogs_table='POGS.fits'):
    def angular_distance(ra1, dec1, ra_list, dec_list):
        ra1 = np.radians(ra1)
        dec1 = np.radians(dec1)
        ra_list = np.radians(ra_list)
        dec_list = np.radians(dec_list)
        cos_theta = (np.sin(dec1) * np.sin(dec_list) +
                     np.cos(dec1) * np.cos(dec_list) * np.cos(ra1 - ra_list))
        cos_theta = np.clip(cos_theta, -1, 1)
        angular_dist = np.arccos(cos_theta) 
        return np.degrees(angular_dist)
    print ("Searching a "+str(fov)+" degree field of view centered at RA : "+str(pointing_ra_deg)+" and DEC: "+str(pointing_dec_deg)+"degree.\n")
    ###########################
    # Stokes I calculation
    ###########################
    stokes_I=Table.read(ggsm_table)
    ra=np.array(stokes_I['RAJ2000'].tolist())
    dec=np.array(stokes_I['DEJ2000'].tolist())
    I_flux=np.array(stokes_I['S_200'].tolist())
    source_angular_distances=angular_distance(pointing_ra_deg, pointing_dec_deg, ra, dec)
    pos=np.where(source_angular_distances<=fov)
    I_total_flux=round(np.nansum(I_flux[pos]),2)
    
    #############################
    # Linpol calculation
    #############################
    stokes_P=Table.read(pogs_table)
    ra=np.array(stokes_P['ra'].tolist())
    dec=np.array(stokes_P['dec'].tolist())
    polint=np.array(stokes_P['polint'].tolist())
    source_angular_distances=angular_distance(pointing_ra_deg, pointing_dec_deg, ra, dec)
    pos=np.where(source_angular_distances<=fov)
    P_total_flux=round(np.nansum(polint[pos]),2)
    
    print ("Total Stokes I flux: "+str(I_total_flux)+" and total polarized flux: "+str(P_total_flux)+" Jy.\n")
    total_P_frac=round(P_total_flux/I_total_flux,2)
    print ("Total polarization fracton: "+str(round(total_P_frac*100,2))+"%.\n")
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


def wait_for_resources(finished_file_prefix, cpu_threshold=20, memory_threshold=20):
    """
    Wait for free hardware resources
    Parameters
    ----------
    finished_file_prefix : str
        Finished file prefix name
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
    finished_file_list = glob.glob(finished_file_prefix + "*")
    while True:
        resource_available = check_resource_availability(
            cpu_threshold=cpu_threshold, memory_threshold=memory_threshold
        )
        if resource_available:
            new_finished_file_list = glob.glob(finished_file_prefix + "*")
            if len(new_finished_file_list) - len(finished_file_list) > 0:
                free_jobs = len(new_finished_file_list) - len(finished_file_list)
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
