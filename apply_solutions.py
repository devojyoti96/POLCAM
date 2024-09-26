from casatasks import applycal
from calibrate_crossphase import apply_crossphasecal
import os

os.system("rm -rf casa*log")

########
# Inputs
########
msname = input("Name of the target measurement set:")
bandpass_table = input("Path of the bandpass table:")
kcross_table = input("Path of the crosshand phase table:")

print("Applying bandpass solutons...\n")
applycal(vis=msname, gaintable=[bandpass_table], applymode="calflag", flagbackup=True)
print("Applying crosshand phase solutions...\n")
apply_crossphasecal(msname, gaintable=kcross_table, datacolumn="CORRECTED_DATA")
