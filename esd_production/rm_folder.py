import os
import shutil

output = '/disks/shear10/brouwer_veersemeer/shearcode_output/'

folders = os.listdir(output)

for folder in folders:
	path = '/disks/shear10/brouwer_veersemeer/shearcode_output/%s/splits_shearbootstrap'%folder

	print
	print path
	if os.path.isdir(path):
		filelist = os.listdir(path)

		for filename in filelist:
			print filename
			os.remove('%s/%s'%(path, filename))

