# import warnings
# warnings.filterwarnings("ignore")

import os


trk1v2tck_str = '/ifs/loni/faculty/shi/spectrum/yxia/code/src_barr/run_TrackVis2MRTrix3Tracks.sh'
matlab_path = '/usr/local/MATLAB/R2015b/'


def trk1v2tck(trk_file, reference, output_tck_file, stepSize=1):
    os.system(f'{trk1v2tck_str} {matlab_path} {trk_file} {stepSize} {reference} {output_tck_file}')

