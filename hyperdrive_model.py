import os, numpy as np, time, multiprocessing as mp, psutil
import glob
from casacore.tables import table as casacore_table, makecoldesc
from optparse import OptionParser

os.system("rm -rf casa*log")


def import_model(msname, beamfile, sourcelist):
    """
    Simulate visibilities and import in the measurement set
    Parameters
    ----------
    msname : str
        Name of the measurement set
    beamfile : str
        Beam file name
    sourcelist : str
        Source file name
    """
    try:
        starttime = time.time()
        metafits = (
            os.path.dirname(os.path.abspath(msname))
            + "/"
            + os.path.basename(msname).split(".ms")[0].split("_")[0]
            + ".metafits"
        )
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
        return 0
    except Exception as e:
        print("Model simulation and import failed for : ", msname)
        print ('Exception:',e)
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
    (options, args) = parser.parse_args()
    if options.msname == None:
        print("Please provide the measurement set name.\n")
        return 1
    if options.beamfile == None:
        print("Please provide the MWA PB file.\n")
        return 1
    if options.sourcelist == None:
        print("Please provide the sourcelist file.\n")
        return 1
    msg = import_model(options.msname, options.beamfile, options.sourcelist)
    return msg


if __name__ == "__main__":
    result = main()
    os._exit(result)
