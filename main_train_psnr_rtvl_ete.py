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

from models.network_image_retrieval import ResNet50
from models.loss_rtvl import DPSHLoss, quantization_swdc_loss

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
    parser.add_argument('--opt', type=str, default=json_path,
                        help='Path to option JSON file.')
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
        util.mkdirs(
            (path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt['path']['models'], net_type='E')
    if init_iter_G > 0:
        opt['path']['pretrained_netG'] = init_path_G
    if init_iter_E > 0:
        opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt['path']['models'], net_type='optimizerG')
    if init_iter_optimizerG > 0:
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
        utils_logger.logger_info(logger_name, os.path.join(
            opt['path']['log'], logger_name+'.log'))
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
            train_size = int(math.ceil(len(train_set) /
                             dataset_opt['dataloader_batch_size']))
            train_num_classes = dataset_opt['num_classes']
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(
                    train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
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
    # Step--4 (initialize retrieval model)
    # ----------------------------------------
    '''
    gama, quantization_alpha = 1.0, 0.1
    n_bit = opt['train']['n_bit']
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(
        1, -1, 1, 1).to(model.device)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(
        1, -1, 1, 1).to(model.device)

    rtvl_model = ResNet50(n_bit)
    rtvl_model.load_state_dict(torch.load(opt['path']['pretrained_rtvl']))
    rtvl_model.to(model.device)

    loss_hash = DPSHLoss(train_num_classes, n_bit,
                         len(train_set), model.device)
    loss_quant = quantization_swdc_loss

    params = [
        {'params': rtvl_model.parameters(), 'lr': 1e-5},
        {'params': [v for _, v in model.netG.named_parameters()
                    if v.requires_grad], 'lr': 2e-4}
    ]
    rtvl_optimizer = torch.optim.Adam(params)

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

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2-3) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            model.G_optimizer.zero_grad()
            rtvl_optimizer.zero_grad()

            model.netG_forward()

            loss = 0.0

            E_img = model.E
            y = train_data['img_class'].to(model.device)
            idx = train_data['index'].to(model.device)

            E_img = (E_img - norm_mean) / norm_std

            fv = rtvl_model(E_img)

            loss += model.G_lossfn_weight * model.G_lossfn(model.E, model.H)
            model.log_dict['G_loss'] = loss.item()

            loss += gama * loss_hash(fv, y, idx)
            loss += quantization_alpha * loss_quant(fv, model.device)

            loss.backward()

            model.G_optimizer.step()
            rtvl_optimizer.step()

            if model.opt_train['E_decay'] > 0:
                model.update_E(model.opt_train['E_decay'])

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                message += f'rtvl_loss: {round(loss.item(), 6)} '
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)
                torch.save(rtvl_model.state_dict(),
                           f'{model.save_dir}/{current_step}_rtvl.pth')

            # # -------------------------------
            # # 6) testing
            # # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                pass


if __name__ == '__main__':
    main()
