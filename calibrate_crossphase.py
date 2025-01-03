import numpy as np, os, copy, time, psutil, sys
from casacore.tables import table
from datetime import datetime
import numexpr as ne
from scipy.interpolate import interp1d

class SuppressOutput:
    """
    Context manager to suppress stdout and stderr.
    """
    def __enter__(self):
        self._stdout = sys.stdout  # Save the current stdout
        self._stderr = sys.stderr  # Save the current stderr
        sys.stdout = open(os.devnull, 'w')  # Redirect stdout to /dev/null
        sys.stderr = open(os.devnull, 'w')  # Redirect stderr to /dev/null
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
    
def crossphasecal(msname, caltable="", uvrange="", gaintable=''):
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
    Returns
    -------
    str
            Name of the caltable
    """
    ncpu=int(psutil.cpu_count()*0.8)
    if ncpu<1:
        ncpu=1 
    ne.set_num_threads(ncpu)
    starttime=time.time()
    if caltable == "":
        caltable = msname.split(".ms")[0] + ".kcross"
    #######################
    with SuppressOutput():
        tb = table(gaintable)
        if type(gaintable)==list:
            gaintable=gaintable[0]
        gain=tb.getcol("CPARAM")
        tb.close()
        del tb
    with SuppressOutput():    
        tb=table(msname + "/SPECTRAL_WINDOW")
        freqs = tb.getcol("CHAN_FREQ").flatten()
        cent_freq = tb.getcol("REF_FREQUENCY")[0]
        wavelength = (3 * 10**8) / cent_freq
        tb.close()
        del tb
    with SuppressOutput():    
        tb=table(msname)
        ant1=tb.getcol("ANTENNA1")
        ant2=tb.getcol("ANTENNA2")
        data = tb.getcol("DATA")
        model_data = tb.getcol("MODEL_DATA")
        flag = tb.getcol("FLAG")
        uvw = tb.getcol("UVW")
        weight=tb.getcol('WEIGHT')
        # Col shape, baselines, chans, corrs
        weight = np.repeat(weight[:,np.newaxis, 0], model_data.shape[1], axis=1)
        tb.close()
    if uvrange != "":
        uvdist = np.sqrt(uvw[:,0]**2 + uvw[:,1]**2)
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
        weight=weight[uv_filter, :]
        ant1=ant1[uv_filter]
        ant2=ant2[uv_filter]
    #######################
    data[flag] = np.nan
    model_data[flag] = np.nan
    xy_data = data[...,1]
    yx_data = data[...,2]
    xy_model = model_data[...,1]
    yx_model = model_data[...,2]
    gainX1=gain[ant1,:,0]
    gainY1=gain[ant1,:,-1]
    gainX2=gain[ant2,:,0]
    gainY2=gain[ant2,:,-1]
    del data,model_data,uvw,flag,gain
    argument = ne.evaluate("weight * xy_data * conj(xy_model * gainX1) * gainY2 + weight * yx_model * gainY1 * conj(gainX2 * yx_data)")
    crossphase = np.angle(np.nansum(argument, axis=0), deg=True)
    crossphase=np.mod(crossphase,360)
    chan_flags = get_chans_flags(msname)
    p=np.polyfit(freqs[~chan_flags],crossphase[~chan_flags],3)
    poly=np.poly1d(p)
    crossphase_fit=poly(freqs)
    np.save(caltable, np.array([freqs, crossphase, crossphase_fit, chan_flags], dtype="object"))
    os.system("mv " + caltable + ".npy " + caltable)
    print (time.time()-starttime)
    return caltable


def apply_crossphasecal(
    msname, gaintable="", datacolumn="DATA"):
    """
    Apply crosshand phase on the data
    Parameters
    ----------
    msname : str
        Name of the measurement set
    gaintable : str
        Crosshand phase gaintable
    datacolumn : str
        Data column to read and modify the same data column
    """
    ncpu=int(psutil.cpu_count()*0.8)
    if ncpu<1:
        ncpu=1 
    ne.set_num_threads(ncpu)
    if gaintable == "":
        print("Please provide gain table name.\n")
        return
    freqs, crossphase, crossphase_fit, chan_flags = np.load(gaintable, allow_pickle=True)
    freqs = freqs.astype("float32")
    crossphase = crossphase.astype("float32")
    crossphase = np.deg2rad(crossphase)
    pos = np.where(chan_flags == False)
    f = interp1d(freqs[pos], crossphase[pos], kind="linear", fill_value="extrapolate")
    with SuppressOutput():
        tb=table(msname + "/SPECTRAL_WINDOW")
        ms_freq = tb.getcol("CHAN_FREQ").flatten()
        tb.close()
    crossphase = f(ms_freq)
    with SuppressOutput():
        tb = table(msname, readonly=False)
        data = tb.getcol(datacolumn)
        crossphase = np.repeat(crossphase[..., np.newaxis], data.shape[-1], axis=-1)
        xy_data = data[1, ...]
        yx_data = data[2, ...]
        xy_data_cor = ne.evaluate("exp(1j * crossphase) * xy_data")
        yx_data_cor = ne.evaluate("exp(-1j * crossphase) * yx_data")
        data[1, ...] = xy_data_cor
        data[2, ...] = yx_data_cor
        tb.putcol(datacolumn, data)
        tb.flush()
        tb.close()
    return
