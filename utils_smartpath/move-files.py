import shutil
import glob
import os

for k in glob.glob(r'**\*.tif'):
    new_name =k.split(os.sep)[0]+'.tif'
    shutil.copy(k,new_name)
