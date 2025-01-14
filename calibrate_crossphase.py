import numpy as np, os, copy, time, psutil, sys, warnings
from casacore.tables import table
from datetime import datetime
import numexpr as ne
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")


class SuppressOutput:
    """
    Context manager to suppress stdout and stderr.
    """

    def __enter__(self):
        self._stdout = sys.stdout  # Save the current stdout
        self._stderr = sys.stderr  # Save the current stderr
        sys.stdout = open(os.devnull, "w")  # Redirect stdout to /dev/null
        sys.stderr = open(os.devnull, "w")  # Redirect stderr to /dev/null
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()  # Close the devnull stream
        sys.stderr.close()  # Close the devnull stream
        sys.stdout = self._stdout  # Restore original stdout
        sys.stderr = self._stderr  # Restore original stderr


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
    with SuppressOutput():
        tb = table(msname)
        flag = tb.getcol("FLAG")
        tb.close()
    chan_flags = np.all(np.all(flag, axis=-1), axis=0)
    return chan_flags


def create_corssphase_table(msname, caltable, freqs, crossphase, flags):
    with SuppressOutput():
        freqres = freqs[1] - freqs[0]
        tb = table(caltable + "/SPECTRAL_WINDOW", readonly=False)
        freqs = np.array(freqs)[np.newaxis, :]
        tb.putcol("CHAN_FREQ", freqs)
        tb.putcol("NUM_CHAN", len(freqs))
        tb.putcol("REF_FREQUENCY", np.nanmean(freqs))
        tb.putcol("CHAN_WIDTH", np.array([freqres] * len(freqs))[np.newaxis, :])
        tb.putcol("EFFECTIVE_BW", np.array([freqres] * len(freqs))[np.newaxis, :])
        tb.putcol("RESOLUTION", np.array([freqres] * len(freqs))[np.newaxis, :])
        tb.close()
        tb = table(caltable, readonly=False)
        ant = tb.getcol("ANTENNA1")
        gain = tb.getcol("CPARAM")
        cross_phase_gain = np.repeat(
            np.exp(1j * np.deg2rad(crossphase))[np.newaxis, :], len(ant), axis=0
        )
        gain[..., 0] = cross_phase_gain
        gain[..., 1] = cross_phase_gain * 0 + 1
        tb.putcol("CPARAM", gain)
        flags = flags[np.newaxis, :, np.newaxis]
        flags = np.repeat(np.repeat(flags, len(ant), axis=0), 2, axis=2)
        tb.putcol("FLAG", flags)
        tb.close()
    return caltable


def average_with_padding(array, chanwidth, axis=0, pad_value=np.nan):
    """
    Averages an array along a specified axis with a given chunk width (chanwidth),
    padding the array if its size along that axis is not divisible by chanwidth.
    Parameters
    ----------
    array : ndarray
        Input array to average.
    chanwidth : int
        Width of chunks to average.
    axis : int
        Axis along which to perform the averaging.
    pad_value : float
        Value to pad with if padding is needed (default: np.nan).
    Returns
    --------
    ndarray
        Array averaged along the specified axis.
    """
    # Compute the shape along the specified axis
    original_size = array.shape[axis]
    pad_size = -original_size % chanwidth
    # If padding is needed, apply it directly along the target axis
    if pad_size > 0:
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, pad_size)
        array = np.pad(array, pad_width, constant_values=pad_value)
    # Compute the new shape and reshape the array for chunking
    new_shape = list(array.shape)
    new_shape[axis] = array.shape[axis] // chanwidth
    new_shape.insert(axis + 1, chanwidth)
    reshaped_array = array.reshape(new_shape)
    # Use nanmean along the chunk axis for averaging
    averaged_array = np.nanmean(reshaped_array, axis=axis + 1)
    return averaged_array


def crossphasecal(
    msname,
    caltable="",
    uvrange="",
    gaintable="",
    chanwidth=1,
    bandtype="B",
    polyorder=3,
):
    """
    Function to calculate MWA cross hand phase
    Parameters
    ----------
    msname : str
            Name of the measurement set
    caltable : str
        Name of the caltable
    uvrange : str
        UV-range for calibration
    gaintable : str
            Previous gaintable
    chanwidth : int
        Channels to average
    bandtype : str
        Band type (B or BPOLY)
    polyorder : int
        For BPOLY mode, polynomial order
    Returns
    -------
    str
            Name of the caltable
    """
    starttime = time.time()
    ncpu = int(psutil.cpu_count() * 0.8)
    if ncpu < 1:
        ncpu = 1
    ne.set_num_threads(ncpu)
    starttime = time.time()
    if caltable == "":
        caltable = msname.split(".ms")[0] + ".kcross"
    #######################
    with SuppressOutput():
        tb = table(gaintable)
        if type(gaintable) == list:
            gaintable = gaintable[0]
        gain = tb.getcol("CPARAM")
        tb.close()
        del tb
    with SuppressOutput():
        tb = table(msname + "/SPECTRAL_WINDOW")
        freqs = tb.getcol("CHAN_FREQ").flatten()
        cent_freq = tb.getcol("REF_FREQUENCY")[0]
        wavelength = (3 * 10**8) / cent_freq
        tb.close()
        del tb
    with SuppressOutput():
        tb = table(msname)
        ant1 = tb.getcol("ANTENNA1")
        ant2 = tb.getcol("ANTENNA2")
        data = tb.getcol("DATA")
        model_data = tb.getcol("MODEL_DATA")
        flag = tb.getcol("FLAG")
        uvw = tb.getcol("UVW")
        weight = tb.getcol("WEIGHT")
        # Col shape, baselines, chans, corrs
        weight = np.repeat(weight[:, np.newaxis, 0], model_data.shape[1], axis=1)
        tb.close()
    if uvrange != "":
        uvdist = np.sqrt(uvw[:, 0] ** 2 + uvw[:, 1] ** 2)
        if "~" in uvrange:
            minuv_m = float(uvrange.split("lambda")[0].split("~")[0]) * wavelength
            maxuv_m = float(uvrange.split("lambda")[0].split("~")[-1]) * wavelength
        elif ">" in uvrange:
            minuv_m = float(uvrange.split("lambda")[0].split(">")[-1]) * wavelength
            maxuv_m = np.nanmax(uvdist)
        else:
            minuv_m = 0.1
            maxuv_m = float(uvrange.split("lambda")[0].split("<")[-1]) * wavelength
        uv_filter = (uvdist >= minuv_m) & (uvdist <= maxuv_m)
        # Filter data based on uv_filter
        data = data[uv_filter, :, :]
        model_data = model_data[uv_filter, :, :]
        flag = flag[uv_filter, :, :]
        weight = weight[uv_filter, :]
        ant1 = ant1[uv_filter]
        ant2 = ant2[uv_filter]
    #######################
    data[flag] = np.nan
    model_data[flag] = np.nan
    xy_data = data[..., 1]
    yx_data = data[..., 2]
    xy_model = model_data[..., 1]
    yx_model = model_data[..., 2]
    gainX1 = gain[ant1, :, 0]
    gainY1 = gain[ant1, :, -1]
    gainX2 = gain[ant2, :, 0]
    gainY2 = gain[ant2, :, -1]
    del data, model_data, uvw, flag, gain
    if chanwidth > 1:
        xy_data = average_with_padding(xy_data, chanwidth, axis=1, pad_value=np.nan)
        yx_data = average_with_padding(yx_data, chanwidth, axis=1, pad_value=np.nan)
        xy_model = average_with_padding(xy_model, chanwidth, axis=1, pad_value=np.nan)
        yx_model = average_with_padding(yx_model, chanwidth, axis=1, pad_value=np.nan)
        gainX1 = average_with_padding(gainX1, chanwidth, axis=1, pad_value=np.nan)
        gainX2 = average_with_padding(gainX2, chanwidth, axis=1, pad_value=np.nan)
        gainY1 = average_with_padding(gainY1, chanwidth, axis=1, pad_value=np.nan)
        gainY2 = average_with_padding(gainY2, chanwidth, axis=1, pad_value=np.nan)
        weight = average_with_padding(weight, chanwidth, axis=1, pad_value=np.nan)
    argument = ne.evaluate(
        "weight * xy_data * conj(xy_model * gainX1) * gainY2 + weight * yx_model * gainY1 * conj(gainX2 * yx_data)"
    )
    crossphase = np.angle(np.nansum(argument, axis=0), deg=True)
    freqs = average_with_padding(freqs, chanwidth, axis=0, pad_value=np.nan)
    if chanwidth > 1:
        chan_flags = np.array([False] * len(crossphase))
    else:
        chan_flags = get_chans_flags(msname)
    chan_unflags = np.where((crossphase != 0) & (np.isnan(crossphase) == False))[0]
    p = np.polyfit(freqs[chan_unflags], crossphase[chan_unflags], polyorder)
    poly = np.poly1d(p)
    crossphase_fit = poly(freqs)
    if bandtype == "BPOLY":
        crossphase = crossphase_fit
        chan_flags *= False
    else:
        res = np.sin(np.radians(crossphase)) - np.sin(np.radians(crossphase_fit))
        c = 0
        while c < 3:
            std = np.nanstd(res)
            pos = np.where(np.abs(res) > 3 * std)[0]
            if len(pos) == 0:
                break
            chan_flags[pos] = True
            res[pos] = np.nan
            c += 1
    os.system(
        "python3 create_caltable.py --msname "
        + msname
        + " --caltable "
        + caltable
        + " --nchan "
        + str(len(crossphase))
        + "> /dev/null 2>&1"
    )
    create_corssphase_table(msname, caltable, freqs, crossphase, chan_flags)
    return caltable
