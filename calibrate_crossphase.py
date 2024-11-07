import numpy as np, os
from casatools import table,agentflagger,msmetadata
from casatasks import applycal, split
from basic_func import  get_chans_flags
from datetime import datetime
from scipy.interpolate import interp1d
os.system("rm -rf casa*log")


def crossphasecal(msname, caltable="", uvrange="", gaintable=[]):
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
    if len(gaintable) > 0:
        print ('applycal(vis=\''+msname+'\',gaintable='+str(gaintable)+', applymode=\"calflag\", flagbackup=True)\n')
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
    tb.open(msname+"/SPECTRAL_WINDOW")
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
    chan_flags=get_chans_flags(msname)
    np.save(caltable, np.array([freqs, crossphase, chan_flags],dtype='object'))
    os.system('mv '+caltable+'.npy '+caltable)
    return caltable


def apply_crossphasecal(msname, gaintable="", datacolumn="DATA", applymode="calflag", flagbackup=True):
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
    applymode : str
        Apply calibration and flags    
    flagbackup : bool
        Keep backup of the flags before applying solutions    
    """
    if gaintable == "":
        print("Please provide gain table name.\n")
        return
    freqs, crossphase, chan_flags = np.load(gaintable, allow_pickle=True)
    freqs=freqs.astype('float32')
    crossphase=crossphase.astype('float32')
    crossphase=np.deg2rad(crossphase)
    pos=np.where(chan_flags==False)
    f = interp1d(freqs[pos], crossphase[pos], kind='linear', fill_value='extrapolate')
    msmd=msmetadata()
    msmd.open(msname)
    ms_freq=msmd.chanfreqs(0)
    msmd.close()  
    crossphase=f(ms_freq)  
    if flagbackup:
        af=agentflagger()
        af.open(msname)                                                                            
        versionlist=af.getflagversionlist()
        if len(versionlist)!=0:
            for version_name in versionlist:
                if 'apply_crossphasecal' in version_name:
                    try:
                        version_num=int(version_name.split(':')[0].split(' ')[0].split('_')[-1])+1
                    except:
                        version_num=1
                else:
                    version_num=1
        else:
            version_num=1
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        af.saveflagversion('apply_crossphasecal_'+str(version_num),'Flags autosave on '+dt_string)
        af.done()
    tb = table(msname, nomodify=False)
    data = tb.getcol(datacolumn)
    crossphase = np.repeat(crossphase[..., np.newaxis], data.shape[-1], axis=-1)
    xy_data = data[1, ...]
    yx_data = data[2, ...]
    xy_data_cor = np.exp(1j * crossphase) * xy_data
    yx_data_cor = np.exp(-1j * crossphase) * yx_data
    data[1, ...] = xy_data_cor
    data[2, ...] = yx_data_cor
    tb.putcol(datacolumn, data)
    tb.flush()
    tb.close()
    return
