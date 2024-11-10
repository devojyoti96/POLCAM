import glob,os,psutil,traceback, gc
from basic_func import *
from optparse import OptionParser
os.system("rm -rf casa*log")

beamfile=os.getcwd()+'/mwa_full_embedded_element_pattern.h5'
source_model_file=os.getcwd()+'/GGSM.txt'

def perform_model_import(msdir,basedir,cpu_percent=10,mem_percent=20):
    """
    Perform model import in all ms
    Parameters
    ----------
    msdir : str
        MS directory
    basedir : str
        Base directory
    cpu_percent : float
        Free CPU percentage
    mem_percent : float
        Free memory percentage
    Returns
    -------
    int
        Success message (0 or 1)       
    """ 
    print ("Model import jobs are being started ....\n")
    try:
        os.system("rm -rf "+basedir+'/.Finished_model*')             
        mslist=glob.glob(msdir+'/*.ms')
        trial_ms=mslist[0]
        mssize=get_column_size(trial_ms,'DATA') # In GB
        total_memory=psutil.virtual_memory().available/(1024**3) # In GB
        max_jobs=int(total_memory/mssize)
        total_cpus=psutil.cpu_count()
        ncpu=int(total_cpus/max_jobs)
        if ncpu<1:
            ncpu=1
        print ('Maximum numbers of jobs to spawn at once:',max_jobs)
        count=0
        free_jobs=-1
        for ms in mslist:
            cmd='python3 hyperdrive_model.py --msname '+ms+' --beamfile '+beamfile+' --sourcelist '+source_model_file+' --ncpu '+str(ncpu)
            basename='model_'+os.path.basename(ms).split('.ms')[0]+'_hyperdrive'
            batch_file=create_batch_script_nonhpc(cmd, basedir, basename)
            os.system("bash " + batch_file)
            print ('Spawned command: '+cmd+'\n')
            count+=1
            if free_jobs>0:
                free_jobs-=1
            if count>=max_jobs or free_jobs==0:
                free_jobs=wait_for_resources(basedir+'/.Finished_model',cpu_threshold=cpu_percent, memory_threshold=mem_percent)
                print ('Freeded jobs: ',free_jobs)
        while True:
            finished_files=glob.glob(basedir+'/.Finished_model*')    
            if len(finished_files)>=count:
                break
        print ("#####################\nModel import jobs are finished successfully.\n#####################\n")
        gc.collect()
        return 0
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        print ("#####################\nModel import jobs are finished unsuccessfully.\n#####################\n")
        return 1            

def perform_all_calibration(msdir,basedir,refant=1,cpu_percent=10,mem_percent=20):
    """
    Perform bandpass and crosshand phase calibration for all ms
    Parameters
    ----------
    msdir : str
        MS directory
    basedir : str
        Base directory
    refant : int
        Reference antenna index
    cpu_percent : float
        Free CPU percentage
    mem_percent : float
        Free memory percentage
    Returns
    -------
    int
        Success message (0 or 1)       
    """  
    print ("Calibration jobs are being started ....\n")
    try:
        os.system("rm -rf "+basedir+'/.Finished_calibrate*')          
        mslist=glob.glob(msdir+'/*.ms')
        caldir=basedir+'/caldir'
        trial_ms=mslist[0]
        mssize=get_column_size(trial_ms,'DATA') # In GB
        total_memory=psutil.virtual_memory().available/(1024**3) # In GB
        max_jobs=int(total_memory/mssize)
        print ('Maximum numbers of jobs to spawn at once:',max_jobs)
        count=0
        free_jobs=-1
        for ms in mslist:
            cmd='python3 calibrate.py --msname '+ms+' --refant '+str(refant)+' --caldir '+caldir
            basename='calibrate_'+os.path.basename(ms).split('.ms')[0]+'_bcal_kcross'
            batch_file=create_batch_script_nonhpc(cmd, basedir, basename)
            os.system("bash " + batch_file)
            print ('Spawned command: '+cmd+'\n')
            count+=1
            '''if free_jobs>0:
                free_jobs-=1'''
            if count>=max_jobs:# or free_jobs==0:
                free_jobs = wait_for_resources(basedir+'/.Finished_calibrate', cpu_threshold=cpu_percent, memory_threshold=mem_percent)
                total_memory=psutil.virtual_memory().available/(1024**3) # In GB
                max_jobs=int(total_memory/mssize)
                count=0
                print ('Maximum freed jobs: ',max_jobs)
        while True:
            finished_files=glob.glob(basedir+'/.Finished_calibrate*')    
            if len(finished_files)>=count:
                break    
        print ("#####################\nCalibration jobs are finished successfully.\n#####################\n")
        gc.collect()
        return 0        
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        print ("#####################\nCalibration jobs are finished unsuccessfully.\n#####################\n")
        return 1            
            
def perform_all_applycal(msdir,caldir,basedir,do_flag=True,cpu_percent=10,mem_percent=20):
    """
    Apply calibration solutions of all target ms
    Parameters
    ----------
    msdir : str
        Name of the target ms directory
    caldir : str
        Name of calibration table directory
    basedir : str
        Base directory
    do_flag : bool
        Do flagging or not
    cpu_percent : float
        Free CPU percentage
    mem_percent : float
        Free memory percentage
    Returns
    -------
    int
        Success message (0 or 1)            
    """   
    print ("Apply calibration solution jobs are being started ....\n")
    try:         
        os.system("rm -rf "+basedir+'/.Finished_applycal*')          
        mslist=glob.glob(msdir+'/*.ms')               
        bcal_tables=glob.glob(caldir+'/*.bcal')
        kcross_tables=glob.glob(caldir+'/*.kcross')
        trial_ms=mslist[0]
        mssize=get_column_size(trial_ms,'DATA') # In GB
        total_memory=psutil.virtual_memory().available/(1024**3) # In GB
        max_jobs=int(total_memory/(2*mssize))
        print ('Maximum numbers of jobs to spawn at once:',max_jobs)
        count=0
        free_jobs=-1
        for ms in mslist:
            mssize=get_column_size(ms,'DATA') # In GB
            tb = table()
            tb.open(ms + "/SPECTRAL_WINDOW")
            freq = tb.getcol("CHAN_FREQ")
            tb.close()
            start_coarse_chan = freq_to_MWA_coarse(freq[0] / 10**6)
            end_coarse_chan = freq_to_MWA_coarse(freq[-1] / 10**6)
            coarse_chan_str=str(start_coarse_chan)+'_'+str(end_coarse_chan)
            bcal=''
            kcross=''
            for caltable in bcal_tables:
                if coarse_chan_str in caltable:
                    bcal=caltable
                    break
            for caltable in kcross_tables:
                if coarse_chan_str in caltable:
                    kcross=caltable
                    break 
            if bcal=='' or kcross=='':
                print ('Caltable(s) for the same coarse channels do(es) not exist.\n')
            else:
                cmd='python3 apply_solutions.py --msname '+ms+' --bandpass_table '+str(bcal)+' --kcross_table '+str(kcross)+' --do_flag '+str(do_flag)
                basename='applycal_'+os.path.basename(ms).split('.ms')[0]+'_bcal_kcross'
                batch_file=create_batch_script_nonhpc(cmd, basedir, basename)
                os.system("bash " + batch_file)
                print ('Spawned command: '+cmd+'\n')
                count+=1
                if free_jobs>0:
                    free_jobs-=1
                if count>=max_jobs or free_jobs==0:
                    free_jobs = wait_for_resources(basedir+'/.Finished_applycal', cpu_threshold=cpu_percent, memory_threshold=mem_percent)  
                    print ('Freed jobs: ',free_jobs)
        while True:
            finished_files=glob.glob(basedir+'/.Finished_applycal*')    
            if len(finished_files)>=count:
                break
        print ("#####################\nApply calibration solution jobs are finished unsuccessfully.\n#####################\n")
        gc.collect()
        return 0
    except Exception as e:
        traceback.print_exc()
        gc.collect()
        print ("#####################\nApply calibration solution jobs are finished unsuccessfully.\n#####################\n") 
        return 1        
                       
           
################################
def main():
    usage = "Master controller for MWA Polcal pipeline"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--calms_dir",
        dest="calms_dir",
        default=None,
        help="Name of the calibrator measurement set directory",
        metavar="String",
    )
    parser.add_option(
        "--caltable_dir",
        dest="caltable_dir",
        default=None,
        help="Name of the calibrator solutions directory",
        metavar="String",
    )
    parser.add_option(
        "--targetms_dir",
        dest="targetms_dir",
        default=None,
        help="Name of the target measurement set directory",
        metavar="String",
    )
    parser.add_option(
        "--basedir",
        dest="basedir",
        default=None,
        help="Name of the base working directory",
        metavar="String",
    )
    parser.add_option(
        "--refant",
        dest="refant",
        default=1,
        help="Reference antenna",
        metavar="Integer",
    )
    parser.add_option(
        "--do_target_flag",
        dest="do_target_flag",
        default=True,
        help="Perform target flagging or not",
        metavar="Boolean",
    )
    parser.add_option(
        "--import_model",
        dest="import_model",
        default=True,
        help="Import model or not",
        metavar=True,
    )
    parser.add_option(
        "--free_cpu_percent",
        dest="cpu_percent",
        default=10,
        help="Amount of free CPU percentage",
        metavar="Float",
    )
    parser.add_option(
        "--free_mem_percent",
        dest="mem_percent",
        default=20,
        help="Amount of free memory percentage",
        metavar="Float",
    )
    (options, args) = parser.parse_args()  
    if options.calms_dir==None and options.caltable_dir==None:
        print ("No calibrator observations or solutions are provided.\n")
        return 1
    if options.basedir==None:
        print ("Please provide a base directory name.\n")
        return 1
    elif os.path.exists(options.basedir)==False:        
        os.makedirs(options.basedir)
    
    if options.calms_dir!=None:
        os.system("rm -rf "+options.calms_dir+'/*model.ms*')
        os.system("rm -rf "+options.calms_dir+'/*.bcal')
        if eval(str(options.import_model))==True:
            msg=perform_model_import(options.calms_dir,options.basedir,cpu_percent=float(options.cpu_percent),mem_percent=float(options.mem_percent))
            gc.collect() 
        else:
            msg=0    
        if msg==1:
            return 1
        else:
            msg=perform_all_calibration(options.calms_dir,options.basedir,refant=int(options.refant),cpu_percent=float(options.cpu_percent),\
                                        mem_percent=float(options.mem_percent))
            gc.collect()                            
            if msg==1:
                return 1
            elif options.targetms_dir!=None:
                caldir=options.basedir+'/caldir'
                msg = perform_all_applycal(options.targetms_dir,caldir,options.basedir,do_flag=eval(str(options.do_target_flag)),\
                        cpu_percent=float(options.cpu_percent),mem_percent=float(options.mem_percent))                              
                gc.collect() 
                if msg ==1:
                    return 1
                else:
                    return 0 
            else:
                return 0                               
    elif options.targetms_dir!=None:
        msg = perform_all_applycal(options.targetms_dir,options.caltable_dir,options.basedir,do_flag=eval(str(options.do_target_flag)),\
                        cpu_percent=float(options.cpu_percent),mem_percent=float(options.mem_percent))                              
        gc.collect() 
        if msg ==1:
            return 1 
        else:
            return 0               
                    

if __name__ == "__main__":
    result = main()
    os._exit(result)   
    gc.collect()
        
           
           
           
           
           
           
           
           
           
           
           
           
           
           
           
            
            


