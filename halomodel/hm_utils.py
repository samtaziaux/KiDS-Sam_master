import os
from glob import glob
#from ConfigParser import SafeConfigParser

def read_config(config_file, version='0.5.7'):
    config = open(config_file)
    for line in config:
        if line.replace(' ', '').replace('\t', '')[0] == '#':
            continue
        line = line.split()
        elif line[0] == 'data':
            datafiles = sorted(glob(line[1]))
            datacols = [int(i) for i in line[2].split(',')]
        elif line[0] == 'covariance':
            covfile = glob(line[1])
            if len(covfile) > 1:
                msg = 'More than one possible covariance file'
                raise ValueError(msg)
            covfile = covfile[0]
            covcols = [int(i) for i in line[2].split(',')]
        elif line[0] == 'halomodel':
            model = line[1]
        # also read param names - follow the satellites Early Science function
        elif line[0][:11] == 'model_param':
            model_params.append(line[1])

        elif line[0] == 'hm_param':
            if line[2] not in valid_types:
                msg = 'ERROR: Please provide only valid prior types in the'
                msg += ' parameter file (%s). Value %s is invalid.' \
                       %(paramfile, line[1])
                msg = ' Valid types are %s' %valid_types
                print msg
                exit()
            params.append(line[1])
            prior_types.append(line[2])
            if line[2] == 'read':
                filename = os.path.join(path, line[3])
                val1.append(loadtxt(filename, usecols=(int(line[4]),)))
                val2.append(-1)
            else:
                val1.append(float(line[3]))
                if len(line) > 4:
                    val2.append(float(line[4]))
                else:
                    val2.append(-1)
            if line[2] in ('normal', 'lognormal'):
                if len(line) > 5:
                    val3.append(float(line[5]))
                    val4.append(float(line[6]))
                else:
                    val3.append(-inf)
                    val4.append(inf)
                starting.append(float(line[3]))
            else:
                val3.append(-inf)
                val4.append(inf)
            if line[2] == 'uniform':
                starting.append(float(line[-1]))
        elif line[0] == 'values':
            if line[2] != 'fixed':
                msg = 'ERROR: Arrays can only contain fixed values.'
                print msg
                exit()
            param_types.append(line[0])
            params.append(line[1])
            prior_types.append(line[2])
            val1.append(array(line[3].split(','), dtype=float))
            val2.append(-1)
            val3.append(-inf)
            val4.append(inf)
        elif line[0] == 'metadata':
            meta_names.append(line[1].split(','))
            fits_format.append(line[2].split(','))
    return
