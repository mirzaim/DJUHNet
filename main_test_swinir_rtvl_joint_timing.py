
import argparse
import json
import time

import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from data.dataset_sr_rtvl import DatasetSR_RTVL

from models.network_swinir_img_rtval import SwinIR as net
# from models.network_swinir import SwinIR as net
from models.network_image_retrieval import ResNet50

from bicubic_pytorch.core import imresize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=8,
                        help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--training_patch_size', type=int, default=48, help='patch size used in training SwinIR. '
                        'Just used to differentiate two different settings in Table 2 of the paper. '
                        'Images are NOT tested patch by patch.')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--rtvl_model_path', type=str,
                        default='model_zoo/model1.pt')
    parser.add_argument('--rep_dim', type=int, default=544)
    parser.add_argument('--dataset_folder', type=str, default=None,
                        help='input low-quality test image folder')
    parser.add_argument('--folder_cl', type=str,
                        default=None, help='input image classes')
    parser.add_argument('--hash_bit', type=int,
                        default=32, help='number of hashingbits')
    args = parser.parse_args()

    dataset_opt = {
        'phase': 'test',
        'scale': args.scale,
        'n_channels': 3,
        'H_size': args.scale * args.training_patch_size,
        'img_classes': args.folder_cl,
    }
    dataset_query = DatasetSR_RTVL({'dataset_fold': 'query',
                                    'dataroot_H': args.dataset_folder+'/queryH_pad256',
                                    'dataroot_L': args.dataset_folder+'/queryL_pad256',
                                    **dataset_opt})
    dataset_index = DatasetSR_RTVL({'dataset_fold': 'index',
                                    'dataroot_H': args.dataset_folder+'/indexH',
                                    'dataroot_L': None,
                                    # 'dataroot_L': args.dataset_folder+'/indexL',
                                    **dataset_opt})

    query_dloader = DataLoader(dataset_query,
                               batch_size=1,
                               shuffle=True,
                               num_workers=2,
                               pin_memory=True)
    index_dloader = DataLoader(dataset_index,
                               batch_size=1,
                               shuffle=True,
                               num_workers=2,
                               pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    print(f'loading model from {args.model_path}')
    model = define_model(args)
    model.eval()
    model = model.to(device)

    rtvl_model = ResNet50(args.hash_bit)
    rtvl_model.load_state_dict(torch.load(args.rtvl_model_path))
    rtvl_model.eval()
    rtvl_model.to(device)

    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(
        1, -1, 1, 1).to(device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)

    print('start testing.')
    fvs_index, label_index = [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(index_dloader)):
            y = data['img_class'].to(device)
            img_H = data['H'].to(device)

            fv_H = rtvl_model((img_H - norm_mean) / norm_std)
            fv_H = fv_H.tanh().sign()

            fvs_index.append(fv_H), label_index.append(y)
        fvs_index, label_index = torch.cat(fvs_index).cpu(
        ).numpy(), torch.cat(label_index).cpu().numpy()

    running_time = []    
    for _ in range(10):
        fvs_query_L, fvs_query_SR, fvs_query_H, label_query = [], [], [], []
        start = time.time()
        with torch.no_grad():
            for i, data in enumerate(tqdm(query_dloader)):
                y = data['img_class'].to(device)
                img_L = data['L'].to(device)
                img_H = data['H'].to(device)

                fv_L = rtvl_model(
                    imresize((img_L - norm_mean) / norm_std, args.scale))
                model.rep_vec_list.append(fv_L.detach())
                img_E = model(img_L)

                fv_E = rtvl_model((img_E - norm_mean) / norm_std)
                fv_H = rtvl_model((img_H - norm_mean) / norm_std)

                fv_L, fv_E, fv_H = fv_L.tanh().sign(), fv_E.tanh().sign(), fv_H.tanh().sign()

                fvs_query_L.append(fv_L), fvs_query_SR.append(
                    fv_E), fvs_query_H.append(fv_H), label_query.append(y)
            fvs_query_L, fvs_query_SR = torch.cat(fvs_query_L).cpu(
            ).numpy(), torch.cat(fvs_query_SR).cpu().numpy()
            fvs_query_H, label_query = torch.cat(fvs_query_H).cpu(
            ).numpy(), torch.cat(label_query).cpu().numpy()


        mAP = CalcTopMap(fvs_index, fvs_query_SR, label_index, label_query, -1)
        end = time.time()

        running_time.append(end - start)
    avg_time = sum(running_time) / (len(running_time) * len(label_query))
    print(f'Running time is {avg_time * 1000}ms.')



def define_model(args):
    model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv', rep_vec_dim=args.rep_dim)
    param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys(
    ) else pretrained_model, strict=True)

    return model


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(
            np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def CalcTopMapWithPR(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall

def get_curve_data(mAP, cum_prec, cum_recall, num_dataset, steps=100):
    index_range = num_dataset // steps
    index = [i * steps - 1 for i in range(1, index_range + 1)]
    max_index = max(index)
    overflow = num_dataset - index_range * steps
    index = index + [max_index + i for i in range(1, overflow + 1)]
    c_prec = cum_prec[index]
    c_recall = cum_recall[index]

    pr_data = {
        "index": index,
        "P": c_prec.tolist(),
        "R": c_recall.tolist()
    }
    return pr_data

if __name__ == '__main__':
    main()
