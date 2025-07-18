from casatasks import rerefant
from calibrate_crossphase import crossphasecal
import os

os.system("rm -rf casa*log")


def change_refant(msname, tablein="", caltable_prefix="", refant=1):
    """
    Parameters
    ----------
    msname : str
        Name of the measurement set
    tablein : str
        Input bandpass table name
    caltable_prefix : str
        Caltable name prefix for new bandpass aand crossphase table
    refant : int
        New reference antenna index
    Returns
    -------
    str
        Bandpass table name
    str
        Crosshand phase table name
    """
    caltable_prefix += "_refant_" + str(refant)
    rerefant(
        vis=msname,
        tablein=tablein,
        caltable=caltable_prefix + ".bcal",
        refantmode="strict",
        refant=str(refant),
    )
    if os.path.exists(caltable_prefix + ".bcal") == False:
        print("Error in re-referencing for antenna: ", refant)
        return None, None
    else:
        bcal_caltable = caltable_prefix + ".bcal"
        crossphase_caltable = crossphasecal(
            msname,
            caltable=caltable_prefix + ".kcross",
            gaintable=[bcal_caltable],
        )
        return bcal_caltable, crossphase_caltable
