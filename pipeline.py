import os,glob, psutil

beamfile=f"{os.getcwd()}/mwa_full_embedded_element_pattern.h5"
sourcelist=f"{os.getcwd()}/GGSM.fits"
ncpu=int(psutil.cpu_count()*0.8)
mem=psutil.virtual_memory().total*0.8/1024**3 # In GB

########
#Inputs
########
cal_dir="/media/devojyoti/Data1/MWA_work/MWA_FR_work/datadir/20241004/calibrators"
target_dir="/media/devojyoti/Data1/MWA_work/MWA_FR_work/datadir/20241004/targets"
workdir="/media/devojyoti/Data1/MWA_work/MWA_FR_work/analysis"
refant=1


import_model=False
do_basic_Cal=True



####################
#Pipeline starting
####################
calms_list=glob.glob(f"{cal_dir}/*.ms")
targetms_list=glob.glob(f"{target_dir}/*.ms")
caldir=f"{workdir}/caldir_B"

###########################
# Importing models
###########################
if import_model:
    print ("Start importing sky models....")
    for calms in calms_list:
        metafits=f"{cal_dir}/{os.path.basename(calms).split('.ms')[0].split('_')[0]}.metafits"
        cmd=f"python3 hyperdrive_model.py --msname {calms} --metafits {metafits} --beamfile {beamfile} --sourcelist {sourcelist} --ncpu {ncpu}"
        print ("############################################")
        print (cmd)
        print ("############################################")
        os.system(cmd)
        
        
################################
# Basic calibration
################################
if do_basic_Cal:
    print ("Starting basic calibration....")
    for calms in calms_list:
        cmd=f"python3 calibrate.py --msname {calms} --refant {refant} --do_kcross True --do_flag True --caldir {caldir} --bandtype B"
        print ("############################################")
        print (cmd)
        print ("############################################")
        os.system(cmd)




















        
