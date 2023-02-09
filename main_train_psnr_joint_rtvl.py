import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

from models.network_image_retrieval import FeatureExNetwork

from pytorch_metric_learning import losses
from bicubic_pytorch.core import imresize


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            train_num_classes = dataset_opt['num_classes']
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (initialize model)
    # ----------------------------------------
    '''
    alpha1, alpha2 = 0.1, 0.1
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss()
    loss_cl = losses.ContrastiveLoss(neg_margin=3)
    rtvl_model = FeatureExNetwork(num_classes=train_num_classes)
    rtvl_model.load_state_dict(torch.load(opt['path']['pretrained_rtvl']))
    rtvl_model.to(model.device)
    rtvl_optimizer = torch.optim.Adam(rtvl_model.parameters(), lr=0.001)

    '''
    # ----------------------------------------
    # Step--5 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            if i % 2:
                model.feed_data(train_data)
                model.test()

                L_img, H_img, E_img, = model.L.detach(), model.H.detach(), model.E.detach()
                y = train_data['img_class'].to(model.device)

                rtvl_optimizer.zero_grad()

                E_feature, E_q_faeture, E_pred = rtvl_model(E_img)
                UL_feature, UL_q_faeture, UL_pred = rtvl_model(imresize(L_img, opt['scale']))
                H_feature, H_q_faeture, H_pred = rtvl_model(H_img)
                    
                loss = loss_ce(E_pred, y) + \
                       alpha1 * loss_mse(UL_feature, H_feature) + \
                       alpha2 * loss_cl(torch.cat((E_feature, H_feature)), torch.cat((y, y)))
                loss.backward()
                rtvl_optimizer.step()

            else:

                # -------------------------------
                # 1) update learning rate
                # -------------------------------
                model.update_learning_rate(current_step)

                # -------------------------------
                # 2) feed patch pairs
                # -------------------------------
                model.feed_data(train_data)

                # -------------------------------
                # 3) optimize parameters
                # -------------------------------
                model.optimize_parameters(current_step)         

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                message += f'rtvl_loss: {loss} '
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)
                torch.save(rtvl_model.state_dict(), f'{model.save_dir}/{current_step}_rtvl.pth')

            # # -------------------------------
            # # 6) testing
            # # -------------------------------
            # if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

            #     avg_psnr = 0.0
            #     idx = 0

            #     for test_data in test_loader:
            #         idx += 1
            #         image_name_ext = os.path.basename(test_data['L_path'][0])
            #         img_name, ext = os.path.splitext(image_name_ext)

            #         img_dir = os.path.join(opt['path']['images'], img_name)
            #         util.mkdir(img_dir)

            #         model.feed_data(test_data)
            #         model.test()

            #         visuals = model.current_visuals()
            #         E_img = util.tensor2uint(visuals['E'])
            #         H_img = util.tensor2uint(visuals['H'])

            #         # -----------------------
            #         # save estimated image E
            #         # -----------------------
            #         save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
            #         util.imsave(E_img, save_img_path)

            #         # -----------------------
            #         # calculate PSNR
            #         # -----------------------
            #         current_psnr = util.calculate_psnr(E_img, H_img, border=border)

            #         logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

            #         avg_psnr += current_psnr

            #     avg_psnr = avg_psnr / idx

            #     # testing log
            #     logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()
