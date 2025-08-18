# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/10 18:47
@Auth ： dongb
@IDE ：PyCharm
"""
import datetime
import logging
import os
import torch
import torch.nn.functional as F

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=12)
    return beijing_time.timetuple()

def logger_configuration(filename, phase, save_log=True):
    logger = logging.getLogger("RDJSCC")
    workdir = "./history/{}".format(filename)
    if phase == "TEST":
        filename = filename+"_test"
    log = workdir + "/{}.log".format(filename)
    models = workdir + "/models"
    makedirs(workdir)
    makedirs(models)
    logging.Formatter.converter = beijing
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if save_log:
        filehandler = logging.FileHandler(log)
        filehandler.setLevel(logging.INFO)
        logger.addHandler(filehandler)
    return workdir, logger

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

def psnr_calculate(img_tensor, output_imgs):
    mse = torch.mean((output_imgs * 255. - img_tensor * 255.) ** 2)
    psnr = 20 * torch.log10(255. / torch.sqrt(mse))
    return psnr

def calculate_cosine_similarity_squared_mean(tensor_pairs):
    return [torch.mean(torch.pow(F.cosine_similarity(t1, t2), 2), dim=[0, 1, 2]) for t1, t2 in tensor_pairs]
