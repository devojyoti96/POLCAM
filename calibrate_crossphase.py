import numpy as np, os
from casatools import table
from casatasks import applycal

os.system("rm -rf casa*log")


def crossphasecal(msname, caltable="", gaintable=[]):
    """
    Function to calculate MWA cross hand phase
    Parameters
    ----------
    msname : str
            Name of the measurement set
    caltable : str
        Name of the caltable
    gaintable : str
            Previous gaintable
    Returns
    -------
    str
            Name of the caltable
    """
    if len(gaintable) > 0:
        applycal(vis=msname, gaintable=gaintable, applymode="calflag", flagbackup=True)
        datacolumn = "CORRECTED_DATA"
    else:
        datacolumn = "DATA"
    if caltable == "":
        caltable = msname.split(".ms")[0] + ".kcross"
    tb = table(msname)
    cor_data = tb.getcol(datacolumn)
    model_data = tb.getcol("MODEL_DATA")
    flag = tb.getcol("FLAG")
    tb.close()
    tb.open(msname + "/SPECTRAL_WINDOW")
    freqs = tb.getcol("CHAN_FREQ").flatten()
    tb.close()
    cor_data[flag] = np.nan
    model_data[flag] = np.nan
    xy_data = cor_data[1, ...]
    yx_data = cor_data[2, ...]
    xy_model = model_data[1, ...]
    yx_model = model_data[2, ...]
    argument = xy_data * xy_model.conjugate() + yx_data.conjugate() * yx_model
    crossphase = np.angle(np.nansum(argument, axis=1), deg=True)
    np.save(caltable, [freqs, crossphase])
    return caltable


def apply_crossphasecal(msname, gaintable="", datacolumn="DATA"):
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
    if gaintable == "":
        print("Please provide gain table name.\n")
        return
    freqs, crossphase = np.load(gaintable, allow_pickle=True)
    tb = table(msname, nomodify=False)
    data = tb.getcol(datacolumn)
    crossphase_rad = np.repeat(
        np.deg2rad(crossphase)[..., np.newaxis], data.shape[-1], axis=-1
    )
    xy_data = data[1, ...]
    yx_data = data[2, ...]
    xy_data_cor = np.exp(1j * crossphase_rad) * xy_data
    yx_data_cor = np.exp(-1j * crossphase_rad) * yx_data
    data[1, ...] = xy_data_cor
    data[2, ...] = yx_data_cor
    tb.putcol(datacolumn, data)
    tb.flush()
    tb.close()
    return
