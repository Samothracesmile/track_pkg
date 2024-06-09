import shutil
import os
from os import makedirs, remove, listdir
from os.path import join as pjoin, split as psplit, abspath, basename, dirname, isfile, exists
from shutil import copyfile, rmtree, copy2, copytree

import warnings
import skimage

import numpy as np
import pandas as pd
from glob import glob
from copy import deepcopy

from subprocess import check_call, Popen
import matplotlib.pyplot as plt

# find function
import importlib
import json

def take_notes(sline, file_path, mode='a'):

    with open(file_path, mode) as file:
        file.write(f"{sline}, ")

def find_func(target_func_name, module_name):
    modellib = importlib.import_module(module_name)

    target_func = None
    for name, func in modellib.__dict__.items():
        if name.lower() == target_func_name.lower():
            target_func = func
            
    return target_func 

def isNaN(num):
    return num != num

# create dir
def create_dir(_dir):
    if not exists(_dir):
        makedirs(_dir)      

##########################################################################################
def remove_file(_file):
    if os.path.exists(_file):
        os.remove(_file) 
        print(_file, ' is removed!')

def to_numpy(x):
    return x.data.cpu().numpy()

def save_args(args, args_file):
    '''
    save args to txt
    '''
    with open(args_file, 'a+') as f:
        json.dump(args.__dict__, f, indent=2)

class Dict2Class(object):
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])

def load_args(args_file):
    '''
    load args from txt
    '''

    with open(args_file, 'r') as f:
        args = json.load(f)

    args = Dict2Class(args)
    # MyNamedTuple = namedtuple('MyNamedTuple', args)
    # args = MyNamedTuple(**args)

    return args
##########################################################################################


# create subdir
from difflib import SequenceMatcher
def extr_common_postfix(string1, string2):

    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    return string1[match.a: match.a + match.size]

def common_postfix(string_list):
    string_list = [basename(file) for file in string_list]
    
    common_string = string_list[0]
    for string in string_list[1:]:
        common_string = extr_common_postfix(common_string, string)
        
    return common_string

def add_fname_postfix(fname, postfix):
    '''Add postfix to the fname with a new subfolder'''
    nfname = fname.replace('.nii.gz',f'_{postfix}.nii.gz')
    nfname = pjoin(dirname(nfname), postfix, basename(nfname))
    return nfname

def add_fname_dirfix(fname, dirfix):
    '''Add postfix to the fname with a new subfolder'''
    nfname = pjoin(dirname(fname), dirfix, basename(fname))
    return nfname


# save npz
def save_npz(npz_file, np_array, data_type='float16'):
    create_dir(dirname(npz_file))
    np.savez(npz_file, np_array)

def load_npz(npz_file, data_type='float16'):
    npzfile = np.load(npz_file)
    np_array = npzfile[npzfile.files[0]] 
    return np_array.astype(data_type) 


# statistic outliers
def statistic_outlier_threshold(score_array, scal=2):

    mean_score = np.mean(score_array)
    std_score = np.std(score_array)

    thr = mean_score + scal*std_score

    new_score_array = score_array[score_array <= thr]

    return new_score_array, thr


def remove_nan(score_array):
    score_array = np.array(score_array).flatten()
    nan_mask = np.isnan(score_array)
    score_array = score_array[~nan_mask]
    return score_array

def iterative_statistic_outlier_threshold(score_array, scal=2, iter_num=10, diff_thr=0.001):

    score_array = np.array(score_array).flatten()
    nan_mask = np.isnan(score_array)
    score_array = score_array[~nan_mask]

    thr_box = []
    new_score_array, thr = statistic_outlier_threshold(score_array, scal=scal)
    thr_box.append(thr)
    if iter_num > 0:
        for _ in range(iter_num):
            new_score_array, thr = statistic_outlier_threshold(new_score_array, scal=scal)
            thr_box.append(thr)
    else:
        diff = 999
        while diff > diff_thr:
            new_score_array, thr = statistic_outlier_threshold(new_score_array, scal=scal)
            thr_box.append(thr)
            diff = np.abs(thr_box[-1] - thr_box[-2])
    return thr_box

# import imageio
import imageio.v2 as imageio

def create_gif(images_list, gif_filename, duration):
    images = [imageio.imread(image) for image in images_list]
    imageio.mimsave(gif_filename, images, duration=duration)