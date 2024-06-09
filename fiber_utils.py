import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from dipy.io.streamline import load_trk, save_trk, load_tck, save_tck
from dipy.io.stateful_tractogram import Space, StatefulTractogram, Origin
from dipy.tracking.streamline import set_number_of_points

import nibabel.streamlines.array_sequence as nsa

import random
from random import randrange


vtk2tck_str='/usr/local/mrtrix3/bin/tckconvert'
tck2trk_str='/ifshome/daydogan/code/Workspace_Matlab/pipelineExecutables/run_MRTrixTrack2TrackVis.sh'


# from nibabel import load, Nifti1Image

class Tract(object):
    def __init__(self, fname, reference, space='rasmm', origin='nifti', check_flg=True, correct_non=True, smoothing_extre=False):

        # space
        if space == 'voxmm':
            to_space = Space.VOXMM
        elif space == 'rasmm':
            to_space = Space.RASMM
        elif space == 'vox':
            to_space = Space.VOX
        
        # origin
        if origin == 'nifti':
            to_origin = Origin.NIFTI
        elif origin == 'trkvis':
            to_origin = Origin.TRACKVIS


        if '.tck' in fname:
            try:
                stf = load_tck(fname, reference, to_space=to_space, to_origin=to_origin, bbox_valid_check=check_flg, trk_header_check=check_flg)
            except:
                print(f'Warning! Valid check is False for {fname}')
                stf = load_tck(fname, reference, to_space=to_space, to_origin=to_origin, bbox_valid_check=False, trk_header_check=False)
        elif '.trk' in fname:
            try:
                stf = load_trk(fname, reference, to_space=to_space, to_origin=to_origin, bbox_valid_check=check_flg, trk_header_check=check_flg)
            except:
                print(f'Warning! Valid check is False for {fname}')
                stf = load_trk(fname, reference, to_space=to_space, to_origin=to_origin, bbox_valid_check=False, trk_header_check=False)

        self.fname = fname
        self.reference = reference
        self.stf = stf
        self.affine = stf.affine

        self.nan_streamlines = 0
        self.extrem_streamlines = 0
        self.extrem_pts = 0


        def curve_mask_interpo(data, mask, window_size=5):
            import scipy
        
            data_box = []
            for data1, mask1 in zip(data.T, mask.T):
                if np.any(mask1):
                    data1[mask1] = np.interp(np.flatnonzero(mask1), np.flatnonzero(~mask1), data1[~mask1])
                    data1 = scipy.ndimage.median_filter(data1, size=window_size)
                data_box.append(data1)
            return np.array(data_box).T

        #################################################################
        # remove nan in streamlines
        if correct_non:
            streamlines = self.stf.streamlines
            new_streamlines = []
            for data in self.stf.streamlines:
                mask = np.isnan(data)
                if np.any(mask):
                    self.nan_streamlines += 1
                    data = curve_mask_interpo(data, mask, window_size=5)
                new_streamlines.append(data)
            self.stf.streamlines = new_streamlines
        ##################################################################

        #################################################################
        # remove extrem
        if smoothing_extre:
            streamlines = self.stf.streamlines
            new_streamlines = []
            for data in self.stf.streamlines:
                mask = (data > 200) | (data < -20) | (data == 0) | np.isnan(data)
                if np.any(mask):
                    self.extrem_streamlines += 1
                    self.extrem_pts += np.sum(mask)
                    data = curve_mask_interpo(data, mask, window_size=5)
                    
                    mask = (data > (np.mean(data, axis=0) + 4*np.std(data, axis=0))) | (data < (np.mean(data, axis=0) - 4*np.std(data, axis=0)))
                    self.extrem_pts += np.sum(mask)
                    data = curve_mask_interpo(data, mask, window_size=5)

                new_streamlines.append(data)
            self.stf.streamlines = new_streamlines
        ##################################################################

    def __len__(self):
        return len(self.stf.streamlines)

    def Tract2Image_v(self, save_trk_fname):
        """
        target_tract: Tract object to be converted to the world space
        target_nifty: Nibabel nifty object provide affine information
        res_tract_fname: filename of trk file for counterpart in the world space 
        """
        self.stf.streamlines = streamlines2image_v(self.stf.streamlines, trk.affine)
        save_trk(self.stf, save_trk_fname, bbox_valid_check=False)

    def Tract2Image_n(self, save_trk_fname):
        """
        target_tract: Tract object to be converted to the world space
        target_nifty: Nibabel nifty object provide affine information
        res_tract_fname: filename of trk file for counterpart in the world space 
        """
        self.stf.streamlines = streamlines2image_n(self.stf.streamlines, trk.affine)
        save_trk(self.stf, save_trk_fname, bbox_valid_check=False)


def merge_tcks(tck_files, refrence_file, merge_tck_file, select_num=None, bbox_valid_check=True, check_flg=True, correct_non=False, smoothing_extre=False, force_flg=False):
    '''merge multiple tcks to one'''
    
    if (not os.path.exists(merge_tck_file)) or force_flg:
        # using the first file for refrence
        refrence_tckfile = tck_files[0]
        reftck = Tract(refrence_tckfile, refrence_file, correct_non=correct_non, check_flg=check_flg, smoothing_extre=smoothing_extre)

        # merge streamlines
        tck_streamlines = []
        for tck_file in tck_files:
            print(tck_file)

            subtck = Tract(tck_file, refrence_file, correct_non=correct_non, check_flg=check_flg, smoothing_extre=smoothing_extre)
            
            if len(subtck) > 10:
                tck_streamlines.append(subtck.stf.streamlines)
            else:
                os.system(f'rm {tck_file}')

        tck_streamlines = nsa.concatenate(tck_streamlines, axis=0)
        
        # select number of streamlines
        if select_num is not None:
            reftck.stf.streamlines = tck_streamlines[:select_num]
        else:
            reftck.stf.streamlines = tck_streamlines
        
        if (not os.path.exists(merge_tck_file)) or (force_flg == True):
            if '.trk' in refrence_tckfile:
                save_trk(reftck.stf, merge_tck_file, bbox_valid_check=bbox_valid_check)
            else:
                save_tck(reftck.stf, merge_tck_file, bbox_valid_check=bbox_valid_check)


def smart_merge_tcks(tck_files, refrence_file, merge_tck_file, tar_number):
    '''Merge tcks and estimate needed session'''

    # calculate streamline number
    total_sm_num = np.sum(np.array([len(Tract(tck_file, refrence_file, correct_non=False).stf.streamlines) for tck_file in tck_files]))
    # check streamline number
    if total_sm_num >= tar_number:
        # merge_tcks
        merge_tcks(tck_files, refrence_file, merge_tck_file, select_num=tar_number, bbox_valid_check=True, force_flg=True)
        expected_session_num = len(tck_files)
    else:
        # estimate session number
        expected_session_num = (len(tck_files)*tar_number)/(total_sm_num+1) + 2
    return int(expected_session_num)
    


def trk_boundary(new_streamlines):
    '''
    Extract numerical boundary of bundle
    '''

    min_corr = np.min(np.array([np.min(sl, axis=0) for sl in new_streamlines]), axis=0)
    max_corr = np.max(np.array([np.max(sl, axis=0) for sl in new_streamlines]), axis=0)

    return min_corr, max_corr


def multi_trk_boundary(trk_files=None, atlas_file=None, np_file=None, force_flg=False):
    '''
    Extract numerical boundary of multiple bundle (expected in the common space)
    '''
    if (np_file is None) or (not os.path.exists(np_file)) or force_flg:
        assert (trk_files is not None)
        assert (atlas_file is not None)
        
        min_corrs, max_corrs = [], []
        for trk_file in trk_files:
            trk_mni = Tract(trk_file, atlas_file, check_flg=False)
            min_corr, max_corr = trk_boundary(trk_mni.stf.streamlines)

            min_corrs.append(min_corr)
            max_corrs.append(max_corr)

        multi_max_corr = np.max(np.array(max_corrs), axis=0)
        multi_min_corr = np.min(np.array(min_corrs), axis=0)
        
        if np_file is not None:
            np.save(np_file, [multi_max_corr, multi_min_corr])
    else:
        multi_max_corr, multi_min_corr = np.load(np_file)
    
    return  multi_max_corr, multi_min_corr


def examine_boundary(trk_data):
    
    '''
    Examing the trk dataset to see if it is normalized
    trk_data: pytorch dataset
    '''
    
    x1s = []
    for batch_idx, (x1, x2, xidx) in enumerate(trk_data):
        x1s.append(x1.data.cpu().numpy())
    aaa = np.transpose(np.array(x1s), (0,2,1))
    
    max_corr = np.max(np.max(aaa, axis=0), axis=0)
    min_corr = np.min(np.min(aaa, axis=0), axis=0)
    
    return max_corr, min_corr


def tck2trk(tck_file, trk_file, reference):
    '''
    Convert tck2trk
    
    '''
    stf = load_tck(tck_file, reference, to_space=Space.RASMM, 
              to_origin=Origin.NIFTI, bbox_valid_check=True, trk_header_check=True)
    
    save_trk(stf, trk_file, bbox_valid_check=True)
    

def invcopy_trk(tar_bundle, inv_mode='merge'):

    if inv_mode == 'merge':
        reverted_tar_bundle = np.array([sl[::-1] for sl in tar_bundle])
        return np.concatenate([tar_bundle, reverted_tar_bundle])
    elif inv_mode == 'inv':
        reverted_tar_bundle = np.array([sl[::-1] for sl in tar_bundle])
        return reverted_tar_bundle
    else:
        return tar_bundle


def bundle2file(bundle, ref_trk, save_trk_fname, bbox_valid_check=True):
    '''
    Save bundle to file
    '''

    # if type(ref_trk) == str:
    #     Tract(name, reference, space='rasmm', origin='nifti', check_flg=True, correct_non=True):

    # create_dir(dirname(save_trk_fname))

    ref_trk.stf.streamlines = bundle
    if '.tck' in save_trk_fname:
        save_tck(ref_trk.stf, save_trk_fname, bbox_valid_check=bbox_valid_check)
    elif '.trk' in save_trk_fname:
        save_trk(ref_trk.stf, save_trk_fname, bbox_valid_check=bbox_valid_check)


def streamlines2image_n(streamlines, affine, voxel_shift=0):
    '''
    For numerical comparison with voxel
    voxel_shift can be 0.5 for corresponding the mid of voxel
    '''
    
    shift =  affine[:-1, 3]
    res =  np.abs(affine[0, 0])
    new_streamlines = [voxel_shift+(sl - shift)/res for sl in streamlines]
    
    return new_streamlines

def streamlines2image_n2(streamlines, affine, voxel_shift=0.5):
    '''
    For numerical comparison with voxel
    voxel_shift can be 0.5 for corresponding the mid of voxel
    '''
    
    inverse_affine = np.linalg.inv(affine)
    new_streamlines = [np.hstack([sl, np.ones((sl.shape[0], 1))]).dot(inverse_affine.T)[:,:3] for sl in streamlines]
    
    return new_streamlines

def streamlines2image_v(streamlines, affine):
    '''
    For visualization
    '''
    
    shift =  affine[:-1, 3]
    res =  np.abs(affine[0, 0])
    new_streamlines = [(sl + shift)/res for sl in streamlines]
    
    return new_streamlines

# normalize trk and unnormalize trk
def normalize_bundle(bundle, max_corr=1, min_corr=0, std_corr=1, mean_corr=0):
    '''
    Normalize bundle
    '''
    # max-min normalization
    norm_rate = max_corr - min_corr
    bundle = [(sl-min_corr)/norm_rate for sl in bundle]

    # std normalization
    bundle = [(sl-mean_corr)/std_corr for sl in bundle]
    
    return bundle

def unnormalize_bundle(bundle, max_corr=1, min_corr=0, std_corr=1, mean_corr=0, only_scale=False):
    '''
    Unnormalize bundle
    '''
    norm_rate = max_corr - min_corr
    print(norm_rate)
    
    if only_scale:
        unnorm_bundle = [(nsl*norm_rate) for nsl in bundle]
    else:
        unnorm_bundle = [(nsl*norm_rate + min_corr) for nsl in bundle]

    if only_scale:
        unnorm_bundle = [(nsl*std_corr) for nsl in unnorm_bundle]
    else:
        unnorm_bundle = [(nsl*std_corr + mean_corr) for nsl in unnorm_bundle]

    return unnorm_bundle
    
# flip padding 
def flip_pad_bundle(bundle, padding_len=20):
    
    def fiber_flip_padding(fiber, padding_len):
        oneend = fiber[:(padding_len+1)][::-1][:-1]
        anotherend = fiber[-(padding_len+1):][::-1][1:]
        padded_fiber = np.concatenate([oneend, fiber, anotherend])
        return padded_fiber

    if padding_len > 0:
        bundle = [fiber_flip_padding(fiber, padding_len) for fiber in bundle]
        
    return bundle


def unflip_pad_bundle(bundle, padding_len=20):
    unpadded_bundle = [fiber[padding_len:-padding_len] for fiber in bundle]
    return unpadded_bundle


def subsampleTract(bundle, subsample_size, seed_idx=0):
    """  extracted_bundle
    subsample_size: number of streamlines in subsampled tract
    """
    streamline_length = len(bundle)

    # avoid the large subsample_size
    if (streamline_length <= subsample_size) or (subsample_size < 0): # convert to list
        sub_bundle = [sl for sl in bundle] 
        return sub_bundle
    
    else:
        random.seed(seed_idx)
        random_intlist = random.sample(range(streamline_length), subsample_size)
        sub_bundle = [bundle[i] for i in random_intlist]    

    return sub_bundle    

    # def subdownsampleTract(self, subsample_size, downsampling_rate = 1, seed_idx=0):
    #     """  
    #     subsample_size: number of streamlines in subsampled tract
    #     downsampling_rate: rate of downsampling of points in each streamline
    #     """
    #     streamline_length = len(self.streamlines)

    #     # avoid the large subsample_size
    #     subsample_size = min(streamline_length,subsample_size)

    #     random.seed(seed_idx)
    #     random_intlist = random.sample(range(streamline_length), subsample_size)
    #     subsample_tract_streamlines = [self.streamlines[i][::downsampling_rate,:] for i in random_intlist]    

    #     return subsample_tract_streamlines

    # def subsampleTract(self, subsample_size, seed_idx=0):
    #     """  extracted_bundle
    #     subsample_size: number of streamlines in subsampled tract
    #     """
    #     streamline_length = len(self.streamlines)

    #     # avoid the large subsample_size
    #     subsample_size = min(streamline_length,subsample_size)

    #     random.seed(seed_idx)
    #     random_intlist = random.sample(range(streamline_length), subsample_size)
    #     subsample_tract_streamlines = [self.streamlines[i] for i in random_intlist]    

    #     return subsample_tract_streamlines    
    
    # def downsampleTract(self, downsampling_rate = 1):
    #     """  
    #     downsampling_rate: rate of downsampling of points in each streamline
    #     """
    #     downsample_tract_streextracted_bundleamlines = [streamline[::downsampling_rate,:] for streamline in self.streamlines]    

    #     return downsample_tract_streextracted_bundleamlines    

# ######################################################### commonness measurement #########################################################
# def distMatrix(s1,s2):
#     s1_norm = np.expand_dims(np.sum(s1**2,axis=1), axis=1)
#     s2_norm = np.expand_dims(np.sum(s2**2,axis=1), axis=0)

#     return np.sqrt(s1_norm + s2_norm - 2.0 * np.matmul(s1, np.transpose(s2)))


# import skimage
from scipy.ndimage import binary_erosion, binary_dilation
import skimage.measure

def find_longest_segment(mask_img):

    # remove noise 
    mask_img = binary_dilation(mask_img)
    mask_img = binary_dilation(mask_img)
    mask_img = binary_erosion(mask_img)
    mask_img = binary_erosion(mask_img)

    mask_scale = skimage.measure.label(mask_img, connectivity=1)
    
    # find outlier label
    outlier_label_set = set(mask_scale[~mask_img])
    assert len(outlier_label_set) == 1
    outlier_label = list(outlier_label_set)[0]
    
    segment_labels = set(mask_scale)
    max_len_label = -1
    max_len = -1

    for label in segment_labels:
        if label != outlier_label:
            segment_len = np.sum(mask_scale == label)
            if segment_len >= max_len:
                max_len = segment_len
                max_len_label = label

    large_segment_mask = mask_scale == max_len_label

    return large_segment_mask


def find_affine(points1, points2):
    '''Find affine transform between two points'''

    # Define two sets of corresponding 3D points
    # points1 = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
    # points2 = np.array([[u1, v1, w1], [u2, v2, w2], [u3, v3, w3]])

    # Calculate the centroids of each set of points
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    # Subtract the centroids from the points
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    # Calculate the scaling and rotation matrix (3x3 matrix)
    H = centered_points2.T @ centered_points1

    # Perform Singular Value Decomposition (SVD) to factorize the matrix
    U, S, Vt = np.linalg.svd(H)

    # Calculate the rotation matrix
    R = Vt.T @ U.T

    # Calculate the translation vector
    t = -R @ centroid1.T + centroid2.T

    # Create the affine transform matrix (4x4 matrix)
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = R
    affine_matrix[:3, 3] = t

    print("Affine Transform Matrix:")
    print(affine_matrix)
    
    return affine_matrix



from nibabel import trackvis as tv
class Tract2(object):
    def __init__(self, fname):
        self.fname = fname
        self.streams, self.header = tv.read(fname)
        self.streamlines = [np.nan_to_num(i[0]) for i in self.streams] #streamlines in list
        self.streamline_scales = [i[1] for i in self.streams] #streamlines in list
        self.streamline_props = [i[2] for i in self.streams] #streamlines in list

    def __len__(self):
        return len(self.streamlines)


def save_tract_old(tract_bundle, filename, scales=None, props=None, tract_hdr=None, ref_trk_file=None):
    '''
    tract_bundle: Numpy array list
    tract_hdr: .trk head file 
    scales: Numpy array list for scales
    '''    
    
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


    if scales is None:
        scales = [None]*len(tract_bundle)

    if props is None:
        props = [None]*len(tract_bundle)


    # use the header form ref trk
    if ref_trk_file is not None:
        tract_hdr = Tract2(ref_trk_file).header

        
    for_save = [(streamline,scale,prop) for streamline, scale, prop in zip(tract_bundle, scales, props)]

    if tract_hdr is not None:
        if not len(for_save)==tract_hdr['n_count']:
            new_header = tract_hdr.copy()
            new_header['n_count'] = len(for_save)

            tv.write(filename, tuple(for_save), new_header)
        else:
            tv.write(filename, tuple(for_save), tract_hdr)

    else:
        tv.write(filename, tuple(for_save))