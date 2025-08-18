# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/9 10:50
@Auth ： dongb
@IDE ：PyCharm
"""
import time

import torch
import torchvision
import yaml
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms

from models.distortion import MS_SSIM, MS_SSIM_Loss, LPIPSLoss
from utils import *
from models.RDJSCC import Jscc
from dataset.PairKitti import PairKitti
from dataset.PairCityscape import PairCityscape
from dataset.PairCifar import PairCifar

# ===================Config
with open('configs/config.yaml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
config["phase"] = "TEST"
# ===================Definitions
log_filename = "{}_{}_awgn".format(config["model_name"], config["dataset_name"])
workdir, logger = logger_configuration(log_filename, config["phase"], save_log=config["save_log"])
# ===================Datasets
path = config['dataset_path']
resize = tuple(config['resize'])
if config['dataset_name'] == 'Cifar10':
    test_dataset = PairCifar(set_type='test').test_pair()
    config["enc_shape"] = [config["enc_N"], 8, 8]
    config["S"] = 6
    config["M"] = 64
    config["pilot_path"] = './configs/Pilot_bit.pt'
    config["batchsize"] = 128
elif config['dataset_name'] == 'KITTI':
    test_dataset = PairKitti(path=path, set_type='test', resize=resize)
elif config['dataset_name'] == 'Cityscape':
    test_dataset = PairCityscape(path=path, set_type='test', resize=resize)
else:
    raise Exception("Dataset not found")
test_loader = DataLoader(dataset=test_dataset, batch_size=config["batchsize"], shuffle=False, num_workers=1, drop_last=True)
# ===================Test
def test(model, commom, test_snrs, config):
    ms_ssim_calculate = MS_SSIM(size_average=True, max_val=1, device=config["device"]).to(config["device"])
    lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True, reduction="mean").to(config["device"])
    lpips_loss.requires_grad_(False).to(config["device"])

    metrics_list = [
        "psnr_x", "ms_ssim_x", "lpips_x",
        "psnr_y", "ms_ssim_y", "lpips_y",
        "cor1", "cor2", "cor3", "cor4", "cor5", "cor6", "cor7", "cor8", "cor9", "cor10",
        "sf1", "sf2", "sf3", "sf4", "sf5", "sf6",
        "loss_H_X_LIST", "loss_H_Y_LIST"
    ]
    results = {metric: np.zeros(len(test_snrs)) for metric in metrics_list}

    for i, test_snr in enumerate(test_snrs):
        if commom == True:
            test_snr1 = test_snr
        else: test_snr1 = 2

        SNR1 = test_snr1 * torch.ones((config["batchsize"], 1), dtype=torch.float32).to(config["device"])
        SNR2 = test_snr * torch.ones((config["batchsize"], 1), dtype=torch.float32).to(config["device"])

        test_psnrs_x, test_ms_ssims_x, test_lpips_x = [AverageMeter() for _ in range(3)]
        test_psnrs_y, test_ms_ssims_y, test_lpips_y  = [AverageMeter() for _ in range(3)]
        elapsed, snrs, papr, papr_clip, cor1, cor2, cor3, cor4, cor5, cor6, cor7, cor8, cor9,cor10  = [AverageMeter() for _ in range(14)]
        sf1, sf2, sf3, sf4, sf5, sf6 = [AverageMeter() for _ in range(6)]
        loss_H_X_LIST, loss_H_Y_LIST, = [AverageMeter() for _ in range(2)]

        metrics_aver = [test_psnrs_x, test_ms_ssims_x, test_lpips_x,
                   test_psnrs_y, test_ms_ssims_y, test_lpips_y,
                   elapsed, cor1, cor2, cor3, cor4,cor5,cor6, cor7, cor8, cor9,cor10,
                   sf1, sf2, sf3, sf4, sf5, sf6,loss_H_X_LIST, loss_H_Y_LIST]

        model.eval()
        with torch.no_grad():
            start_time = time.time()
            for j, data in enumerate(iter(test_loader)):
                input_imgs, side_imgs = data
                input_imgs = input_imgs.to(config["device"])
                side_imgs = side_imgs.to(config["device"])

                x_recon, y_recon, x_enc, y_enc, x_enc1, x_enc2, y_enc1, y_enc2, r3, r3_y, sf,loss_H = model(input_imgs, side_imgs, SNR1, SNR2)

                tensor_pairs = [
                    (input_imgs, side_imgs),
                    (x_enc1, y_enc1),
                    (x_enc2, y_enc2),
                    (x_enc, y_enc),
                    (r3, r3_y),
                    (x_recon[0], y_recon[0]),
                    (x_recon[1], y_recon[1]),
                    (x_recon[2], y_recon[2]),
                    (x_recon[3], y_recon[3]),
                    (x_recon[4], y_recon[4]),
                ]
                corNums = calculate_cosine_similarity_squared_mean(tensor_pairs)
                corNum1, corNum2, corNum3, corNum4, corNum5, corNum6, corNum7, corNum8, corNum9, corNum10 = corNums

                elapsed_time = time.time() - start_time

                psnr_x = psnr_calculate(input_imgs, x_recon[4])
                ms_ssim_x = ms_ssim_calculate(x_recon[4], input_imgs)
                lpips_x = lpips_loss(x_recon[4].clip(0.0, 1.0), input_imgs).mean().to(config["device"]) #(preds, targets)

                psnr_y = psnr_calculate(side_imgs, y_recon[4])
                ms_ssim_y = ms_ssim_calculate(y_recon[4], side_imgs)
                lpips_y = lpips_loss(y_recon[4].clip(0.0, 1.0), side_imgs).mean().to(config["device"]) #(preds, targets)


                values = [psnr_x, ms_ssim_x, lpips_x,
                          psnr_y, ms_ssim_y, lpips_y,
                          elapsed_time, corNum1, corNum2, corNum3, corNum4,corNum5, corNum6, corNum7, corNum8, corNum9, corNum10,
                          sf[0], sf[1], sf[2], sf[3], sf[4], sf[5],
                          loss_H[0],loss_H[1]]
                for metric, value in zip(metrics_aver, values):
                    metric.update(value)


        metrics_values = {
            "psnr_x": test_psnrs_x.avg, "ms_ssim_x": test_ms_ssims_x.avg, "lpips_x": test_lpips_x.avg,
            "psnr_y": test_psnrs_y.avg, "ms_ssim_y": test_ms_ssims_y.avg, "lpips_y": test_lpips_y.avg,
            "cor1": cor1.avg, "cor2": cor2.avg, "cor3": cor3.avg, "cor4": cor4.avg, "cor5": cor5.avg,
            "cor6": cor6.avg, "cor7": cor7.avg, "cor8": cor8.avg, "cor9": cor9.avg, "cor10": cor10.avg,
            "sf1": sf1.avg, "sf2": sf2.avg, "sf3": sf3.avg, "sf4": sf4.avg, "sf5": sf5.avg, "sf6": sf6.avg,
            "loss_H_X_LIST": loss_H_X_LIST.avg, "loss_H_Y_LIST": loss_H_Y_LIST.avg
        }

        for metric, value in metrics_values.items():
            precision = 4 if 'sf' not in metric else 6
            results[metric][i] = round(float(value), precision)

    for metric, _ in metrics_values.items():
        logger.info("{}: {}".format(metric.upper(), results[metric].tolist()))

def main():
    # ===================Initialize Model, criterion, optimizer
    jscc = Jscc(config=config).to(config["device"])
    # ===================Test
    logger.info("======Begin to test: " + log_filename + "======")
    test_snrs = config["test_snrs"]
    save_name = "{}_{}_awgn".format(config["model_name"], "KITTI")
    workdir = "/home/caobin/dongb/xxx"
    checkpoints = torch.load(workdir + "/models/" + save_name + ".model", map_location=config["device"])

    jscc.load_state_dict(checkpoints)
    snrs = [-8, -6, -4, -2, 0, 2]
    for snr in snrs:
        logger.info("=====================test_snr:{}".format(snr))
        test(jscc, snr, test_snrs, config)
    logger.info("Model testing completed")


if __name__ == "__main__":
    main()