from natsort import natsorted
import os
import glob
import re
from subprocess import call

if __name__ == '__main__':

    netsdir = './data/nets'

    pattern = re.compile("net_\d+$")

    for subdir, dirs, files in os.walk(netsdir):
        net_name = subdir.split('/')[-1]
        netparamfiles = natsorted([f for f in files if pattern.match(f)])
        if len(netparamfiles) >0:
            # get latest network parameter
            net_number = netparamfiles[-1]
            ouput_path = subdir+'/preds_'+net_number
            if not os.path.exists(ouput_path):
                import prediction_scripts as ps
                print "validating ",net_name,net_number
                os.makedirs(ouput_path)
                print "calling ",'python prediction_scripts.py --net_number '+net_number+' --net_name '+net_name
                call(['python prediction_scripts.py --net_number='+net_number+\
                    ' --net_name='+net_name+" --save_validation=\""+subdir+'/preds_'+net_number+"/score.txt\""], shell=True)
                # call('python somescript.py', shell=True)
                # call(['python', 'prediction_scripts.py','--net_number='+net_number+\
                    # ' --net_name='+net_name])#, env=os.environ.copy())
                # val = ps.pred_script_v2_wrapper(
                #           net_number=net_number,
                #             net_name=net_name)