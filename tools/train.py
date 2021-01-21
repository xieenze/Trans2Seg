import time
import copy
import datetime
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.solver.loss import get_segmentation_loss
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params, get_color_pallete
from segmentron.config import cfg
from tabulate import tabulate
from IPython import embed
from PIL import Image
try:
    import apex
except:
    print('apex is not installed')

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.use_fp16 = cfg.TRAIN.APEX

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform,
                       'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}

        data_kwargs_testval = {'transform': input_transform,
                       'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TEST.CROP_SIZE}

        train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode='testval', **data_kwargs_testval)
        test_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='test', mode='testval', **data_kwargs_testval)

        self.classes = test_dataset.classes

        self.iters_per_epoch = len(train_dataset) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters, drop_last=True)

        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, cfg.TEST.BATCH_SIZE, drop_last=False)

        test_sampler = make_data_sampler(test_dataset, False, args.distributed)
        test_batch_sampler = make_batch_data_sampler(test_sampler, cfg.TEST.BATCH_SIZE, drop_last=False)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=cfg.DATASET.WORKERS,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_sampler=test_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        # create network
        self.model = get_segmentation_model().to(self.device)
        
        # print params and flops
        if get_rank() == 0:
            try:
                show_flops_params(copy.deepcopy(self.model), args.device)
            except Exception as e:
                logging.warning('get flops and params error: {}'.format(e))
        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')
        # create criterion
        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
                                               aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
                                               ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)
        # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)
        # apex
        if self.use_fp16:
            self.model, self.optimizer = apex.amp.initialize(self.model.cuda(), self.optimizer, opt_level="O1")
            logging.info('**** Initializing mixed precision done. ****')

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters, iters_per_epoch=self.iters_per_epoch)
        # resume checkpoint if needed
        self.start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            name, ext = os.path.splitext(args.resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            logging.info('Resuming training, loading {}...'.format(args.resume))
            resume_sate = torch.load(args.resume)
            self.model.load_state_dict(resume_sate['state_dict'])
            self.start_epoch = resume_sate['epoch']
            logging.info('resume train from epoch: {}'.format(self.start_epoch))
            if resume_sate['optimizer'] is not None and resume_sate['lr_scheduler'] is not None:
                logging.info('resume optimizer and lr scheduler from resume state..')
                self.optimizer.load_state_dict(resume_sate['optimizer'])
                self.lr_scheduler.load_state_dict(resume_sate['lr_scheduler'])

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class, args.distributed)

    def train(self):
        self.save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = cfg.TRAIN.EPOCHS, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.iters_per_epoch

        start_time = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0
        for (images, targets, _) in self.train_loader:
            epoch = iteration // iters_per_epoch + 1
            iteration += 1

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.optimizer.zero_grad()
            if self.use_fp16:
                with apex.amp.scale_loss(losses, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))

            if iteration % self.iters_per_epoch == 0 and self.save_to_disk:
                save_checkpoint(self.model, epoch, self.optimizer, self.lr_scheduler, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(epoch)
                self.test()
                self.model.train()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logging.info("Total training time: {} ({:.4f}s / it)".format(total_training_str,
                                                                     total_training_time / max_iters))

    def validation(self, epoch):
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.eval()
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                if cfg.DATASET.MODE == 'val' or cfg.TEST.CROP_SIZE is None:
                    output = model(image)[0]
                else:
                    size = image.size()[2:]
                    assert cfg.TEST.CROP_SIZE[0] == size[0]
                    assert cfg.TEST.CROP_SIZE[1] == size[1]
                    output = model(image)[0]

            self.metric.update(output, target)
            pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
            logging.info("[EVAL] Sample: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc * 100, mIoU * 100))
        pixAcc, mIoU = self.metric.get()
        logging.info("[EVAL END] Epoch: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(epoch, pixAcc * 100, mIoU * 100))
        synchronize()

    def test(self, vis=False):
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.eval()
        for i, (image, target, filename) in enumerate(self.test_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                if cfg.DATASET.MODE == 'test' or cfg.TEST.CROP_SIZE is None:
                    output = model(image)[0]
                else:
                    size = image.size()[2:]
                    assert cfg.TEST.CROP_SIZE[0] == size[0]
                    assert cfg.TEST.CROP_SIZE[1] == size[1]
                    output = model(image)[0]

            if vis:
                save_gt = False
                if save_gt:
                    test_path = '/mnt/lustre/xieenze/xez_space/TransparentSeg/datasets/transparent/Trans10K_cls12/test/images'
                    save_path = 'workdirs/trans10kv2/gt_img'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    gt_img = Image.open(os.path.join(test_path, filename[0])).resize((512,512))
                    gt_img.save(os.path.join(save_path, str(i) + '.png'))

                    gt_mask = target[0].data.cpu().numpy()
                    vis_gt = get_color_pallete(gt_mask, dataset='trans10kv2')
                    save_path = 'workdirs/trans10kv2/gt_mask'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    vis_gt.save(os.path.join(save_path, str(i) + '.png'))
                else:
                    vis_pred = output[0].permute(1,2,0).argmax(-1).data.cpu().numpy()
                    vis_pred = get_color_pallete(vis_pred, dataset='trans10kv2')
                    save_path = os.path.join(cfg.TRAIN.MODEL_SAVE_DIR, 'vis')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    vis_pred.save(os.path.join(save_path, str(i)+'.png'))
                print("[VIS TEST] Sample: {:d}".format(i+1))
                continue

            self.metric.update(output, target)
            pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
            logging.info("[TEST] Sample: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc * 100, mIoU * 100))

        synchronize()
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info("[TEST END]  pixAcc: {:.3f}, mIoU: {:.3f}".format(pixAcc * 100, mIoU * 100))

        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i]])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid',
                                                           showindex="always", numalign='center', stralign='center')))


if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train' if not args.test else 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)

    # create a trainer and start train
    trainer = Trainer(args)
    if args.test:
        assert 'pth' in cfg.TEST.TEST_MODEL_PATH, 'please provide test model pth!'
        logging.info('test model......')
        trainer.test(args.vis)
    else:
        trainer.train()
        trainer.test()
