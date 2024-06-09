import sys
# sys.path.insert(0, "/home/yihaoxia/Coding/mypackages")
sys.path.insert(0, "/ifs/loni/faculty/shi/spectrum/yxia/code/ydiffpak")
sys.path.insert(0, "/ifs/loni/faculty/shi/spectrum/yxia/code/ytrkpak")


from utils import *
from qsub_utils import *
from diff_utils import load
from fiber_utils import Tract, smart_merge_tcks
# from roi_utils import gen_roi_dict, gen_label_dict, extract_roi, lobe_mapping_dict
from roi_utils import extract_roi, lobe_mapping_dict


def extract_training_sub_dirs(train_num=20):
    import os
    import pickle
    
    adni_trk_datalist_file = '/ifs/loni/faculty/shi/spectrum/yxia/0project/2023_CommonTrk/data/adni_trk_datalist.pkl'
    with open(adni_trk_datalist_file, 'rb') as file:
        adni_trk_datalist = pickle.load(file)

    base_dir = '/ifs/loni/faculty/shi/spectrum/yxia/dataset/ADNI2023_DTI2'
    sub_dirs = []
    for subname, subdate in zip(*adni_trk_datalist):
#         print(subname, subdate)
        sub_dir = f'{base_dir}/{subname}/{subdate}'
        sub_dirs.append(sub_dir)

    sub_dirs = sub_dirs[::-1][:train_num]
    return sub_dirs



def trekker_bundle(fod_file, output_vtk, include1, include2, seed_file, mask_file,
    stepSize=0.05, minRadiusOfCurvature=1.5, probeCount=1, probeLength=0.5, minFODamp=0.05, 
    number=10000, max_length=300, min_length=20,numberOfThreads=0, excludes=[]):
    
    trekker_str='/ifs/loni/faculty/shi/spectrum/yxia/software/trekker/binaries/trekker_linux_x64_v0.7'
    trekker_cmd=f'{trekker_str} -fod {fod_file} -seed_image {seed_file} -seed_count {number} -enableOutputOverwrite'
    # trekker_cmd=f'{trekker_cmd} -pathway_A=require_entry {include1} -pathway_A=stop_at_exit {include1} -pathway_B=require_entry {include2} -pathway_B=stop_at_exit {include2}'
    # trekker_cmd=f'{trekker_cmd} -pathway_A=discard_if_enters {include1} -pathway_A=discard_if_enters {include1} -pathway_B=discard_if_enters {include2} -pathway_B=discard_if_enters {include2}'
    trekker_cmd=f'{trekker_cmd} -pathway_A=require_entry {include1} -pathway_A=stop_at_exit {include1} -pathway_B=require_entry {include2} -pathway_B=stop_at_exit {include2}'

    if excludes:
        for exclude in excludes:
            print(f'Using exclude_file: {exclude}')
            trekker_cmd=f'{trekker_cmd} -pathway_A=discard_if_enters {exclude} -pathway_B=discard_if_enters {exclude}'

    trekker_cmd=f'{trekker_cmd} -stepSize {stepSize} -minRadiusOfCurvature {minRadiusOfCurvature} -probeCount {probeCount} -probeLength {probeLength}' 
    trekker_cmd=f'{trekker_cmd} -minFODamp {minFODamp} -minLength {min_length} -maxLength {max_length} -numberOfThreads {numberOfThreads} -output {output_vtk}'
    
    return trekker_cmd

def generate_simple_freesurfer_roi(bundle_name, roi1_names, pfix_name, free_label_dicts, ana_file, roi_dir):

    ################################ Generate ROIs
    # include 1
    tar1_roi_file = pjoin(roi_dir, f'{bundle_name}_{pfix_name}.nii.gz')
    print(f'ROI name: {tar1_roi_file}')
    tar1_roi_labels = [free_label_dicts[roi1_name] for roi1_name in roi1_names]
    extract_roi(ana_file, tar1_roi_labels, save_roi_file=tar1_roi_file)

    return tar1_roi_file

def generate_simple_label(bundle_name, roi1_labels, pfix_name, ana_file, roi_dir):

    ################################ Generate ROIs
    # include 1
    tar1_roi_file = pjoin(roi_dir, f'{bundle_name}_{pfix_name}.nii.gz')
    print(roi1_labels)
    extract_roi(ana_file, roi1_labels, save_roi_file=tar1_roi_file)

    return tar1_roi_file

def generate_ctx_exclude_freesurfer_roi(bundle_name, roi1_names, pfix_name, free_label_dicts, ana_file, roi_dir):

    ################################ Generate exclusion ROIs of ctx other than roi1_names
    # exclude 1
    tar1_roi_file = pjoin(roi_dir, f'{bundle_name}_{pfix_name}.nii.gz')
    print(f'ROI name: {tar1_roi_file}')
    ex_roi_names = [roi_name for roi_name in list(free_label_dicts.keys()) if ('ctx-' in roi_name) and (roi_name not in roi1_names)]

    tar1_roi_labels = [free_label_dicts[roi1_name] for roi1_name in ex_roi_names]
    extract_roi(ana_file, tar1_roi_labels, save_roi_file=tar1_roi_file)

    return tar1_roi_file


def tckgen_bundle(fod_file, tck_file, include_files, exclude_files, seed_files, mask_file, 
                  step=0.2, angle=9, number=10000, max_length=300, min_length=20, cutoff=0.025, trials=3000, downsample=4,
                  nthreads=0, max_attempts_per_seed=1000):
    
    tckgen_str = '/ifs/loni/faculty/shi/spectrum/yxia/software/mrtrix3/bin/tckgen'
    
    if not exists(tck_file):
        # initial trkgen
        trkgen_cmd = f'{tckgen_str} -algorithm iFOD1 -force -nthreads {nthreads}'
        trkgen_cmd = f'{trkgen_cmd} -mask {mask_file}'

        # include and exclude
        if type(seed_files) == str:
            trkgen_cmd = f'{trkgen_cmd} -seed_image {seed_files}'
        elif type(seed_files) == list and len(seed_files) > 0:
            seed_cmd = ' '.join([f'-seed_image {file}' for file in seed_files])
            trkgen_cmd = f'{trkgen_cmd} {seed_cmd}'

        # include and exclude
        if len(include_files) > 0:
            include_cmd = ' '.join([f'-include {file}' for file in include_files])
            trkgen_cmd = f'{trkgen_cmd} {include_cmd}'

        if len(exclude_files) > 0:
            exclude_cmd = ' '.join([f'-exclude {file}' for file in exclude_files])
            trkgen_cmd = f'{trkgen_cmd} {exclude_cmd}'

        # parameters
        trkgen_cmd = f'{trkgen_cmd} -step {step} -angle {angle} -select {number} -trials {trials} -max_attempts_per_seed {max_attempts_per_seed}'
        trkgen_cmd = f'{trkgen_cmd} -maxlength {max_length} -minlength {min_length} -cutoff {cutoff} -downsample {downsample}'

        # final
        trkgen_cmd = f'{trkgen_cmd} {fod_file} {tck_file}'

        return trkgen_cmd


def para_tckgen_bundle(fod_file, tck_file, include_files, exclude_files, seed_files, mask_file, 
                       session_num, tar_number, session_num_limit, 
                       step=0.2, angle=9, number=10000, max_length=300, min_length=20, cutoff=0.025, trials=3000, downsample=4, nthreads=0, max_attempts_per_seed=1000):
    
    trkgen_cmds = []
    
    sub_tck_files_pattern = tck_file.replace('.tck', f'_t_*.tck')
    sub_tck_files = glob(sub_tck_files_pattern)
    
    if not exists(tck_file):
        if len(sub_tck_files) >= session_num:
            print('*')
            expected_session_num = smart_merge_tcks(sub_tck_files, mask_file, tck_file, tar_number)
        else:
            expected_session_num = session_num

        # reorgnize session_num
        if (expected_session_num <= session_num_limit) and (expected_session_num > session_num):
            print(f'Session num is increase to {expected_session_num}')
            session_num = expected_session_num
        else:
            print(f'Expected Session num {expected_session_num} is too large!s')

        # run stage
        for ii in range(session_num):
            sub_tck_file = tck_file.replace('.tck', f'_t_{ii}.tck')
            if not exists(sub_tck_file):
                trkgen_cmd = tckgen_bundle(fod_file, sub_tck_file, include_files, exclude_files, seed_files, mask_file, 
                              step=step, angle=angle, number=number, max_length=max_length, min_length=min_length, cutoff=cutoff, trials=trials, downsample=downsample, nthreads=nthreads, max_attempts_per_seed=max_attempts_per_seed)
                trkgen_cmds.append(trkgen_cmd)
    else:
        if sub_tck_files:
            print(f'rm {sub_tck_files_pattern}')
            os.system(f'rm {sub_tck_files_pattern}')

    return trkgen_cmds