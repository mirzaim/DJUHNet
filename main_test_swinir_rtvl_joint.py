
import argparse

import torch
from torch.utils.data import DataLoader

from data.dataset_sr_rtvl import DatasetSR_RTVL

from models.network_swinir_img_rtval import SwinIR as net
from models.network_image_retrieval import FeatureExNetwork

from bicubic_pytorch.core import imresize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--rtvl_model_path', type=str,
                        default='model_zoo/model1.pt')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--folder_cl', type=str, default=None, help='input image classes')
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    dataset_opt = {
        'phase': 'test',
        'scale': args.scale,
        'H_size': 512,
        'dataroot_H': args.folder_gt,
        'dataroot_L': args.folder_lq,
        'img_classes': args.folder_cl,
    }
    dataset = DatasetSR_RTVL(dataset_opt)
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    print(f'loading model from {args.model_path}')
    model = define_model(args)
    model.eval()
    model = model.to(device)


    rtvl_model = FeatureExNetwork(num_classes=args.num_classes)
    rtvl_model.load_state_dict(torch.load(args.rtvl_model_path))
    model.eval()
    rtvl_model.to(device)

    fvs_bic, fvs_sr, labels = [], [], []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            y = data['img_class'].to(model.device)

            model.feed_data(data)
            UL_feature, fv_bic, _ = rtvl_model(imresize(model.L.detach(), args.scale))
            model.netG.rep_vec_list.append(UL_feature.detach())
            model.test()

            E_img = model.E.detach()
            _, fv_sr, _ = rtvl_model(E_img)
            fvs_bic.append(fv_bic), fvs_sr.append(fv_sr), labels.append(y)
    fvs_bic, fvs_sr, labels = torch.cat(fvs_bic), torch.cat(fvs_sr), torch.cat(labels)

    border = int(len(labels) * 0.9)
    bic_result = get_mAP(fvs_bic[border:], fvs_bic[:border], labels[border:], labels[:border], k=5)
    sr_result = get_mAP(fvs_sr[border:], fvs_sr[:border], labels[border:], labels[:border], k=5)

    print(f'bic result {bic_result}')
    print(f'sr result {sr_result}')


def define_model(args):
    model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv', rep_vec_dim=1536)
    param_key_g = 'params'
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        
    return model

def get_mAP(query_fvs, index_fvs, query_label, index_label, k=5):
    l2_dist = torch.cdist(query_fvs, index_fvs)
    top_results = index_label[torch.topk(l2_dist, k=k, largest=False).indices]
    resutls = (top_results == query_label).long()
    resutls = resutls.cumsum(dim=1) / (torch.arange(resutls.shape[-1], device=resutls.get_device())+1) * resutls
    return resutls.mean().item()

if __name__ == '__main__':
    main()
