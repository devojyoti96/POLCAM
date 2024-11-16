import os, numpy as np, time, multiprocessing as mp, psutil, glob, traceback, gc
from casacore.tables import table as casacore_table, makecoldesc
from optparse import OptionParser

os.system("rm -rf casa*log")


def import_model(msname, metafits, beamfile, sourcelist, ncpu=-1):
    """
    Simulate visibilities and import in the measurement set
    Parameters
    ----------
    msname : str
        Name of the measurement set
    metafits : str
        Name of the metafits file    
    beamfile : str
        Beam file name
    sourcelist : str
        Source file name
    ncpu : int
        Number of cpu threads to use    
    """
    if ncpu>0:
        os.environ['RAYON_NUM_THREADS']=str(ncpu)
    try:
        starttime = time.time()
        print(
            "#######################\nImporting model for ms:"
            + msname
            + "\n###############################\n"
        )
        data_table = casacore_table(msname + "/SPECTRAL_WINDOW", readonly=True)
        nchan = data_table.getcol("NUM_CHAN")[0]
        freqres = data_table.getcol("RESOLUTION")[0][0] / 10**3
        data_table.close()
        data_table = casacore_table(msname + "/POLARIZATION", readonly=True)
        npol = data_table.getcol("NUM_CORR")[0]
        data_table.close()
        data_table = casacore_table(msname + "/ANTENNA", readonly=True)
        nant = data_table.nrows()
        data_table.close()
        data_table = casacore_table(msname, readonly=False)
        times = np.unique(data_table.getcol("TIME"))
        ntime = times.size
        timeres = data_table.getcol("EXPOSURE")[0]
        data_table.close()
        os.system("rm -rf " + msname.split(".ms")[0] + "_model.ms")
        hyperdrive_cmd = (
            "hyperdrive vis-simulate -m "
            + metafits
            + " --beam-file "
            + beamfile
            + " --freq-res "
            + str(freqres)
            + " --time-res "
            + str(timeres)
            + " -s "
            + sourcelist
            + " -n 1000 --output-model-files "
            + msname.split(".ms")[0]
            + "_model.ms --output-model-freq-average "
            + str(freqres)
            + "kHz --num-fine-channels "
            + str(nchan)
            + " --num-timesteps "
            + str(ntime)
            + " --output-model-time-average "
            + str(timeres)
            + "s"
        )
        print(hyperdrive_cmd + "\n")
        os.system(hyperdrive_cmd + " > tmp_" + os.path.basename(msname).split(".ms")[0])
        os.system("rm -rf tmp_" + os.path.basename(msname).split(".ms")[0])
        model_msname = msname.split(".ms")[0] + "_model.ms"
        ########################
        # Importing model
        ########################
        print("Importing model...\n")
        data_table = casacore_table(msname, readonly=False)
        model_table = casacore_table(model_msname, readonly=False)
        baselines = [*zip(data_table.getcol("ANTENNA1"), data_table.getcol("ANTENNA2"))]
        m_array = model_table.getcol("DATA")
        pos = np.array([i[0] != i[1] for i in baselines])
        model_array = np.empty((len(baselines), nchan, npol), dtype="complex")
        model_array[pos, ...] = m_array
        model_array[~pos, ...] = 0.0
        column_names = data_table.colnames()
        if "MODEL_DATA" in column_names:
            data_table.putcol("MODEL_DATA", model_array)
        else:
            coldesc = makecoldesc("MODEL_DATA", model_table.getcoldesc("DATA"))
            data_table.addcols(coldesc)
            data_table.putcol("MODEL_DATA", model_array)
        data_table.close()
        model_table.close()
        del m_array, model_array
        print("Model import done in : " + str(time.time() - starttime) + "s")
        os.system("rm -rf casa*log")
        os.system("rm -rf " + model_msname)
        gc.collect()
        return 0
    except Exception as e:
        print("Model simulation and import failed for : ", msname)
        traceback.print_exc()
        gc.collect()
        os.system("rm -rf casa*log")
        return 1


################################
def main():
    usage = "Simulate and import visibilities"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--msname",
        dest="msname",
        default=None,
        help="Name of the measurement set",
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
        "--beamfile",
        dest="beamfile",
        default=None,
        help="Name of the MWA PB file",
        metavar="String",
    )
    parser.add_option(
        "--sourcelist",
        dest="sourcelist",
        default=None,
        help="Source model file",
        metavar="String",
    )
    parser.add_option(
        "--ncpu",
        dest="ncpu",
        default=-1,
        help="Numbers of CPU threads to be used",
        metavar="Integer",
    )
    (options, args) = parser.parse_args()
    if options.msname == None:
        print("Please provide the measurement set name.\n")
        return 1
    if options.metafits == None:
        print("Please provide the metafits file name.\n")
        return 1    
    if options.beamfile == None:
        print("Please provide the MWA PB file.\n")
        return 1
    if options.sourcelist == None:
        print("Please provide the sourcelist file.\n")
        return 1
    msg = import_model(options.msname, options.metafits, options.beamfile, options.sourcelist, ncpu = int(options.ncpu))
    return msg


if __name__ == "__main__":
    result = main()
    os._exit(result)
