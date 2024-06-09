# import sys
# sys.path.insert(0, "/ifs/loni/faculty/shi/spectrum/yxia/code/ytrkpak")

from utils import *
from fiber_utils import *
from fiber_dataloader import TrkDataset

import torch

def find_model_from_module(target_model_name, module_name):
    modellib = importlib.import_module(module_name)

    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls
            
    assert model is not None, f'{target_model_name} cannot be found in {module_name}'
    return model


def concatenate_predicts(x_preds):
    
    x = np.vstack([x_pred[0] for x_pred in x_preds])
    x_mu_bar = np.vstack([x_pred[1] for x_pred in x_preds])
    x_aleatoric_uncer = np.vstack([x_pred[2] for x_pred in x_preds])
    x_epistemic_uncer = np.vstack([x_pred[3] for x_pred in x_preds])
    
    return np.swapaxes(x, 1, 2), np.swapaxes(x_mu_bar, 1, 2), np.swapaxes(x_aleatoric_uncer, 1, 2), np.swapaxes(x_epistemic_uncer, 1, 2)


def uncer_estimator(args_save_file_pattern, test_trk_file, model_num=10, gpu_id=0, save_flg=False, uncert_epoch='_ep_1000'):
    
    def ensemble_predict(args_list, x1, device):
        x1_mu_box = []
        x1_var_box = []
        for args in args_list:
            model_path = args.model_path
            model_path = model_path.replace('_ep_1000', uncert_epoch)
            print(model_path)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            x1_mu, x1_var = model(x1.to(device))

    #         print(to_numpy(x1_mu).shape)
    #         print(to_numpy(x1_var).shape)

            x1_mu_box.append(to_numpy(x1_mu))
            x1_var_box.append(to_numpy(x1_var))

        x1_mu_box = np.array(x1_mu_box)
        x1_var_box = np.array(x1_var_box)        

    #     print(x1_mu_box.shape)
    #     print(x1_var_box.shape)

        x1_mu_bar = np.mean(x1_mu_box, axis = 0)
        x1_aleatoric_uncer = np.sqrt(np.mean(x1_var_box, axis=0))
        x1_epistemic_uncer = np.sqrt(np.mean(np.square(x1_mu_box), axis=0) - np.square(x1_mu_bar))
    #     x1_total_uncer = np.sqrt(x1_aleatoric_uncer**2 + x1_epistemic_uncer**2)

        return x1_mu_bar, x1_aleatoric_uncer, x1_epistemic_uncer

#     args_save_files = sorted(glob(args_save_file_pattern))
    args_save_files = [args_save_file_pattern.replace('*', str(model_i)) for model_i in range(model_num)]
    args_list = [load_args(args_save_file) for args_save_file in args_save_files]
    args = load_args(args_save_files[0])
    print(30*'*')
#     print(args_save_files)
    print(args_save_files[0])
    print(f'Total number of models: {len(args_list)}')


    print(30*'*')
    print('tar_trk_pattern:', args.tar_trk_pattern)
    print('test_trk_file:', test_trk_file)
    print(30*'*')

    # 2.2 args.gpuid 
    # args.gpuid = -1
    args.gpuid = gpu_id

    # 0. set device
    if args.gpuid < 0:
        device = torch.device('cpu')
        print('Using CPU!')
    else:
        device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available()  else 'cpu')
        print(f'Using GPU! {args.gpuid}')

    # multi_max_corr, multi_min_corr = multi_trk_boundary(trk_files=args.space_ref_trk_pattern, atlas_file=args.space_ref_file, 
    #                                                     np_file=args.space_boundary_file, force_flg=False)
    
    # load tar_trk for normalization
    tar_trk = Tract(args.tar_trk_pattern, args.space_ref_file, space='rasmm', origin='nifti', check_flg=False, correct_non=True)
    min_corr, max_corr = trk_boundary(tar_trk.stf.streamlines)
    print(30*'*')
    print(args.tar_trk_pattern)
    print(max_corr, min_corr)
    print(30*'*')

    trk_data = TrkDataset(test_trk_file, args.space_ref_file, num_pts=args.num_pts, 
                          subsample_size=args.subsample_size, padding_len=args.padding_len, 
                          max_corr=max_corr, min_corr=min_corr, 
                          ifinvmerge=False, check_flg=False)

    print(args.tar_trk_pattern)
    print('Number of streamlines in training data: ', len(trk_data))

    # args.batch_size = 2000
    trk_data_loader = torch.utils.data.DataLoader(trk_data, batch_size=args.batch_size, shuffle=False, num_workers=1)


    CNN1D_AE = find_model_from_module(args.model_name, args.modellib_name)
    model = CNN1D_AE(z_dim=args.z_dim, num_pts=args.num_pts+2*args.padding_len, dropoutp=args.dropoutp, reduce_mode=args.reduce_mode).to(device)


    # prediction
    x1_preds = []
    x2_preds = []

    for batch_idx, (x1, x2, _) in enumerate(trk_data_loader):
        x1_mu_bar, x1_aleatoric_uncer, x1_epistemic_uncer = ensemble_predict(args_list, x1, device)
        x2_mu_bar, x2_aleatoric_uncer, x2_epistemic_uncer = ensemble_predict(args_list, x2, device)
    #     print(x1_mu_bar.shape)
    #     print(x2_mu_bar.shape)

        x1_preds.append((to_numpy(x1), x1_mu_bar, x1_aleatoric_uncer, x1_epistemic_uncer))
        x2_preds.append((to_numpy(x2), x2_mu_bar, x2_aleatoric_uncer, x2_epistemic_uncer))

    x1, x1_mu_bar, x1_aleatoric_uncer, x1_epistemic_uncer = concatenate_predicts(x1_preds)
    x2, x2_mu_bar, x2_aleatoric_uncer, x2_epistemic_uncer = concatenate_predicts(x2_preds)

    # print(x1_mu_bar.shape)
    # print(x2_mu_bar.shape)
    # print(x1.shape)
    # print(x2.shape)

    # merging inverse
    x1 = np.mean(np.array([x1, x2[:,::-1,:]]), axis=0)
    x1_mu_bar = np.mean(np.array([x1_mu_bar, x2_mu_bar[:,::-1,:]]), axis=0)
    x1_aleatoric_uncer = np.mean(np.array([x1_aleatoric_uncer, x2_aleatoric_uncer[:,::-1,:]]), axis=0)
    x1_epistemic_uncer = np.mean(np.array([x1_epistemic_uncer, x2_epistemic_uncer[:,::-1,:]]), axis=0)

    # unpadding
    x1 = unflip_pad_bundle(x1, padding_len=args.padding_len)
    x1_mu_bar = unflip_pad_bundle(x1_mu_bar, padding_len=args.padding_len)
    x1_aleatoric_uncer = unflip_pad_bundle(x1_aleatoric_uncer, padding_len=args.padding_len)
    x1_epistemic_uncer = unflip_pad_bundle(x1_epistemic_uncer, padding_len=args.padding_len)

    # unnormalizing
    x1 = unnormalize_bundle(x1, max_corr, min_corr, only_scale=False)
    x1_mu_bar = unnormalize_bundle(x1_mu_bar, max_corr, min_corr, only_scale=False)
    x1_aleatoric_uncer = unnormalize_bundle(x1_aleatoric_uncer, max_corr, min_corr, only_scale=True)
    x1_epistemic_uncer = unnormalize_bundle(x1_epistemic_uncer, max_corr, min_corr, only_scale=True)
    

    x1, x1_mu_bar, x1_aleatoric_uncer, x1_epistemic_uncer = np.array(x1), np.array(x1_mu_bar), np.array(x1_aleatoric_uncer), np.array(x1_epistemic_uncer)
    
    #######################################################
    # convert 3 channel uncertainty to 1
    x1_epistemic_uncer = np.expand_dims(np.sum(x1_epistemic_uncer, axis=-1), axis=-1)
    x1_aleatoric_uncer = np.expand_dims(np.sum(x1_aleatoric_uncer, axis=-1), axis=-1)
    # x1_epistemic_uncer = np.expand_dims(np.sum(np.square(x1_epistemic_uncer), axis=-1), axis=-1)
    # x1_aleatoric_uncer = np.expand_dims(np.sum(np.square(x1_aleatoric_uncer), axis=-1), axis=-1)
    #######################################################
    
    if save_flg:
        ###########################################################################################################################################################

        # save results
        ref_trk = Tract(test_trk_file, args.space_ref_file, space='rasmm', origin='nifti', check_flg=False, correct_non=True)

        save_recon = pjoin(args.trk_recon_res_dir, test_subname, basename(test_trk_file).replace('.trk', '_recon.trk'))
        print(save_recon)
        save_tract_old(x1_mu_bar, save_recon, scales=x1_epistemic_uncer, props=None, tract_hdr=None, ref_trk_file=test_trk_file)

        # aleatoric
        save_aleatoric = pjoin(args.trk_recon_res_dir, test_subname, basename(test_trk_file).replace('.trk', '_aleatoric.trk'))
        print(save_aleatoric, np.min(x1_aleatoric_uncer), np.max(x1_aleatoric_uncer))
        save_tract_old(x1, save_aleatoric, scales=x1_aleatoric_uncer, props=None, tract_hdr=None, ref_trk_file=test_trk_file)

        # epistemic
        save_epistemic = pjoin(args.trk_recon_res_dir, test_subname, basename(test_trk_file).replace('.trk', '_epistemic.trk'))
        print(save_epistemic, np.min(x1_epistemic_uncer), np.max(x1_epistemic_uncer))
        save_tract_old(x1, save_epistemic, scales=x1_epistemic_uncer, props=None, tract_hdr=None, ref_trk_file=test_trk_file)
        ###########################################################################################################################################################
    
    return x1, x1_mu_bar, x1_aleatoric_uncer, x1_epistemic_uncer

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
