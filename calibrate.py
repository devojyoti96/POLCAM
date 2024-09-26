from casatasks import bandpass, flagdata
from calibrate_crossphase import crossphasecal
import os

os.system("rm -rf casa*log")

########
# Inputs
########
msname = input("Name of the calibrator measurement set:")
refant = input("Reference antenna:")

print("Flagging.." + msname)
flagdata(vis=msname, mode="tfcrop")
print("################\nCalibrating ms :" + msname + "\n#######################\n")
bandpass(
    vis=msname,
    caltable=msname.split(".ms")[0] + "_ref_" + str(refant) + ".bcal",
    refant=str(refant),
    solint="inf",
    uvrange=">50lambda",
)
print("Estimating crosshand phase...\n")
crossphase_caltable = crossphasecal(
    msname,
    caltable=msname.split(".ms")[0] + "_ref_" + str(refant) + ".kcross",
    gaintable=[msname.split(".ms")[0] + "_ref_" + str(refant) + ".bcal"],
)
