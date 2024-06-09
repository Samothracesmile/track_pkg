from utils import *
from fiber_utils import *

import torch
from torch.utils import data


class TrkDataset(data.Dataset):
    def __init__(self, trk_file, reference, num_pts=256, subsample_size=-1, padding_len=0, max_corr=1, min_corr=0, std_corr=1, mean_corr=0, ifinvmerge=False, check_flg=False, correct_non=True, smoothing_extre=True):
        '''
        trk_file(Str): .tck or .trk 1. file or 2. file_pattern 
        refence(Str): nii 1. file or 2. file_pattern 
        subsample_size: number of streamlines for each bundle
        
        ''' 

        if (type(trk_file) is str) and ('*' in trk_file):
            print('loading from pattern')
            trk_files = sorted(glob(trk_file))

        # elif type(trk_file) is list:
        #     print('loading from list')
        #     trk_files = sorted(trk_file)

            references = sorted(glob(reference))
            
            if len(references) != len(trk_files): # for reference to a common image e.g. MNI atlas
                assert len(references) == 1
                references = len(trk_files)*references

            large_bundle = []
            for atrk_file, areference in zip(trk_files, references):
                stf = Tract(atrk_file, areference, check_flg=check_flg, correct_non=correct_non, smoothing_extre=smoothing_extre).stf
                bundle = subsampleTract(stf.streamlines, subsample_size, seed_idx=0) # subsampling 
                bundle = np.array(set_number_of_points(bundle, nb_points=num_pts)) # set equal number
                # bundle = np.array(streamlines2image_n(bundle, stf.affine)) # bundle to image space
                large_bundle.extend(bundle)
            bundle = large_bundle
            
        else:
            print('loading from one file')
            stf = Tract(trk_file, reference, check_flg=check_flg, correct_non=correct_non, smoothing_extre=smoothing_extre).stf
            bundle = subsampleTract(stf.streamlines, subsample_size, seed_idx=0)
            bundle = np.array(set_number_of_points(bundle, nb_points=num_pts)) # set equal number
            # bundle = np.array(streamlines2image_n(bundle, stf.affine)) # bundle to image space
        
        # padding 
        bundle = flip_pad_bundle(bundle, padding_len=padding_len)
        # print(bundle[0].shape)

        # normalize
        bundle = normalize_bundle(bundle, max_corr, min_corr, std_corr, mean_corr)

        # inverse
        if ifinvmerge: 
            bundle = invcopy_trk(bundle, inv_mode='merge')
        inv_bundle = invcopy_trk(bundle, inv_mode='inv')

        bundle = np.transpose(bundle, (0,2,1))
        inv_bundle = np.transpose(inv_bundle, (0,2,1))
        # print(inv_bundle.shape)
        
        self.n_feature = bundle.shape[-1]
        self.tol_trk_num = bundle.shape[0]
        self.trk_len = bundle.shape[1]
        self.v1 = torch.from_numpy(bundle).float()
        self.v2 = torch.from_numpy(inv_bundle).float()
        
    def __len__(self):
        return self.tol_trk_num

    def __getitem__(self, idx):
        return self.v1[idx], self.v2[idx], torch.from_numpy(np.array(idx))
    