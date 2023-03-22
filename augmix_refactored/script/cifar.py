# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch AugMix training on CIFAR-10/100.

Supports WideResNet, AllConv, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""

import argparse
import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn

#direc = "/work/ws-tmp/sa058646-retreat/augmix-refactored"
#import sys
#sys.path.append(direc)

from augmix_refactored.config import Config
from augmix_refactored.tools.tests import test, test_c
from augmix_refactored.tools.trainer import train
from augmix_refactored.utils.dataloader import get_data
from augmix_refactored.utils.utils import (get_lr_scheduler, get_model,
                                           get_optimizer, setup_logger)                                       

def main():

    parser = argparse.ArgumentParser(
        description='Trains a CIFAR Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = Config.get_parser(parser)
    args = parser.parse_args()

    config: Config = None
    if args.config_path:
        config = Config.load_from_yaml(args.config_path)
        config.apply_parsed_args(args)
    else:
        config = Config.from_parsed_args(args)

    torch.manual_seed(1)
    np.random.seed(1)

    config.save_folder = os.path.join(
        config.save_folder, config.dataset, config.model, datetime.now().strftime("%y-%m-%d_%H_%M_%S_%f"))

    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)
    if not os.path.isdir(config.save_folder):
        raise Exception('%s is not a dir' % config.save_folder)

    # Configuring logging.
    log_path = os.path.join(
        config.save_folder if not config.log_path else config.log_path,
        'training_log.csv')

    logging = setup_logger(log_path)
    config.log_path = log_path

    # TODO Explain during presentation
    #import ipdb;ipdb.set_trace()

    logging.info("Current Configuration: \n" + config.to_yaml().replace("\n", "\n\n"))

    # Load datasets
    train_loader, test_loader, test_data, base_c_path, num_classes = get_data(
        config=config)

    # Create model
    net = get_model(config, num_classes)
    optimizer = get_optimizer(config, net=net)

    # Distribute model across all visible GPUs
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    start_epoch = 0

    if config.resume_path:
        if os.path.isfile(config.resume_path):
            checkpoint = torch.load(config.resume_path)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info('Model restored from epoch:', start_epoch)

    if config.evaluate:
        # Evaluate clean accuracy first because test_c mutates underlying data
        test_loss, test_acc = test(net, test_loader)
        logging.info('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
            test_loss, 100 - 100. * test_acc))

        test_c_acc = test_c(net, test_data, base_c_path, logging)
        logging.info('Mean Corruption Error: {:.3f}'.format(
            100 - 100. * test_c_acc))
        return

    scheduler = get_lr_scheduler(
        config, optimizer, len_train_loader=len(train_loader))

    with open(log_path, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    best_acc = 0
    logging.info('Beginning training from epoch: {}'.format(
        str(start_epoch + 1)))
    for epoch in range(start_epoch, config.epochs):
        begin_time = time.time()

        train_loss_ema = train(net, train_loader, optimizer,
                               scheduler, config, logging, epoch+1)
        test_loss, test_acc = test(net, test_loader, logging)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        checkpoint = {
            'epoch': epoch,
            'dataset': config.dataset,
            'model': config.model,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(config.save_folder, 'checkpoint.pth.tar')
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(
                config.save_folder, 'model_best.pth.tar'))

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_time,
                train_loss_ema,
                test_loss,
                100 - 100. * test_acc,
            ))

        logging.info(
            '\nEpoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
            ' Test Error {4:.2f}\n'
            .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                    test_loss, 100 - 100. * test_acc))

    test_c_acc = test_c(net, test_data, base_c_path, config=config, logging=logging)
    logging.info('Mean Corruption Error: {:.3f}'.format(
        100 - 100. * test_c_acc))

    with open(log_path, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
                (config.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))


if __name__ == '__main__':
    main()
