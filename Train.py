# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/9 10:50
@Auth ： dongb
@IDE ：PyCharm
"""
import time

import yaml
from torch.nn import MSELoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
import torch.optim.lr_scheduler
from models.distortion import MS_SSIM, MS_SSIM_Loss, LPIPSLoss
from utils import *
from models.RDJSCC import Jscc
from dataset.PairKitti import PairKitti
from dataset.PairCityscape import PairCityscape
import wandb

os.environ["WANDB_API_KEY"] = "XXX"  # write your own wandb id

# ===================Config
with open('configs/config.yaml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
# ===================Definitions
log_filename = "{}_{}_awgn".format(config["model_name"], config["dataset_name"])
workdir, logger = logger_configuration(log_filename, config["phase"], save_log=config["save_log"])
writter = SummaryWriter("history/" + log_filename + "/Tensorboard_logs")

if config["use_wandb"]:
    wandb.init(project=config["project_name"], name=log_filename)
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
    # wandb.config.update(args)  # config is a variable that holds and saves hyper parameters and inputs

# ===================Datasets
path = config['dataset_path']
resize = tuple(config['resize'])
if config['dataset_name'] == 'Cifar10':
    train_dataset = PairCifar(set_type='train').train_pair()
    val_dataset = PairCifar(set_type='val').val_pair()
    test_dataset = PairCifar(set_type='test').test_pair()
    config["enc_N"] = 12
    config["enc_shape"] = [config["enc_N"], 8, 8]
    config["S"] = 6
    config["M"] = 64
    config["pilot_path"] = './configs/Pilot_bit.pt'
    config["batchsize"] = 128
elif config['dataset_name'] == 'KITTI':
    train_dataset = PairKitti(path=path, set_type='train', resize=resize)
    val_dataset = PairKitti(path=path, set_type='val', resize=resize)
    test_dataset = PairKitti(path=path, set_type='test', resize=resize)
elif config['dataset_name'] == 'Cityscape':
    train_dataset = PairCityscape(path=path, set_type='train', resize=resize)
    val_dataset = PairCityscape(path=path, set_type='val', resize=resize)
    test_dataset = PairCityscape(path=path, set_type='test', resize=resize)
else:
    raise Exception("Dataset not found")
train_loader = DataLoader(dataset=train_dataset, batch_size=config["batchsize"], shuffle=True, num_workers=1, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=config["batchsize"], shuffle=True, num_workers=1, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config["batchsize"], shuffle=False, num_workers=1,
                         drop_last=True)
# ===================Train
def train_one_epoch(model, criterion, optimizer, epoch):
    model.train()
    train_losses, loss_H_X_LIST, loss_H_Y_LIST, train_psnrs, elapsed, snr1, snr2 = [AverageMeter() for _ in range(7)]
    metrics = [train_losses, loss_H_X_LIST, loss_H_Y_LIST, train_psnrs, elapsed, snr1, snr2]
    start_time = time.time()
    for i, data in enumerate(iter(train_loader)):
        input_imgs, side_imgs = data
        input_imgs = input_imgs.to(config["device"])
        side_imgs = side_imgs.to(config["device"])

        # Adapt to different SNR regimes ranging from -5 dB to 5 dB.
        SNR_TRAIN1 = 10 * (torch.rand(config["batchsize"], 1) - 0.8).to(config["device"])
        SNR_TRAIN2 = 10 * (torch.rand(config["batchsize"], 1) - 0.8).to(config["device"])
        x_recon, y_recon, x_enc, y_enc, x_enc1, x_enc2, y_enc1, y_enc2,r3,r3_y, sf, loss_H = model(input_imgs, side_imgs, SNR_TRAIN1,
                                                                                       SNR_TRAIN2)

        # loss = criterion(x_recon[4], input_imgs) + config["alpha"] * criterion(y_recon[4], side_imgs) + \
        #     config["lam_h"] * loss_H[0] + config["lam_h"] * loss_H[1]
        loss = criterion(x_recon[4], input_imgs) + config["alpha"] * criterion(y_recon[4], side_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed_time = time.time() - start_time
        psnr = psnr_calculate(input_imgs, x_recon[4])

        # metrics = [train_losses, train_psnrs, elapsed, snr1, snr2, cor,sf1,sf2,sf3,sf4,sf5,sf6]
        values = [loss, loss_H[0], loss_H[1], psnr, elapsed_time, SNR_TRAIN1.mean(), SNR_TRAIN2.mean(),loss_H[0],loss_H[1]]
        for metric, value in zip(metrics, values):
            metric.update(value)


    log = (' | '.join([
        f'Epoch {epoch}',
        f'Train Loss {train_losses.val:.4f} ({train_losses.avg:.4f})',
        f'loss_H_X {loss_H_X_LIST.val:.4f} ({loss_H_X_LIST.avg:.4f})',
        f'loss_H_Y {loss_H_Y_LIST.val:.4f} ({loss_H_Y_LIST.avg:.4f})',
        f'Time {elapsed.avg:.4f}',
        f'PSNR {train_psnrs.val:.4f}({train_psnrs.avg:.4f})',
        f'SNR1 {snr1.val:.4f}({snr1.avg:.4f})',
        f'SNR2 {snr2.val:.4f}({snr2.avg:.4f})',
    ]))
    logger.info(log)
    if config["use_wandb"]:
        log_dict = {"PSNR": train_psnrs.avg,"loss": train_losses.avg}
        wandb.log(log_dict)

    writter.add_scalar("train_loss", train_losses.val, global_step=epoch)
    for i in metrics:
        i.clear()
# ===================Validation
def val_epoch(model, criterion, epoch, BEST_LOSS):
    model.eval()
    val_losses, loss_H_X_LIST, loss_H_Y_LIST, val_psnrs, elapsed, snr1, snr2, cor1, cor2, cor3, cor4,cor5, cor6, cor7, cor8, cor9,cor10 = [AverageMeter() for _ in range(17)]
    metrics = [val_losses, loss_H_X_LIST, loss_H_Y_LIST, val_psnrs, elapsed, snr1, snr2, cor1, cor2, cor3, cor4,cor5, cor6, cor7, cor8, cor9,cor10]
    sf1,sf2,sf3,sf4,sf5,sf6 = [AverageMeter() for _ in range(6)]
    metrics += [sf1,sf2,sf3,sf4,sf5,sf6]
    with torch.no_grad():
        start_time = time.time()
        for i, data in enumerate(iter(val_loader)):
            input_imgs, side_imgs = data
            input_imgs = input_imgs.to(config["device"])
            side_imgs = side_imgs.to(config["device"])

            # Adapt to different SNR regimes ranging from -5 dB to 5 dB.
            SNR_TEST1 = 10 * (torch.rand(config["batchsize"], 1) - 0.8).to(config["device"])
            SNR_TEST2 = 10 * (torch.rand(config["batchsize"], 1) - 0.8).to(config["device"])
            x_recon, y_recon, x_enc_v, y_enc_v, x_enc1, x_enc2, y_enc1,y_enc2,r3,r3_y, sf,loss_H = model(input_imgs, side_imgs, SNR_TEST1, SNR_TEST2)

            tensor_pairs = [
                (input_imgs, side_imgs),
                (x_enc1, y_enc1),
                (x_enc2, y_enc2),
                (x_enc_v, y_enc_v),
                (r3, r3_y),
                (x_recon[0], y_recon[0]),
                (x_recon[1], y_recon[1]),
                (x_recon[2], y_recon[2]),
                (x_recon[3], y_recon[3]),
                (x_recon[4], y_recon[4]),
            ]

            corNums = calculate_cosine_similarity_squared_mean(tensor_pairs)
            corNum1, corNum2, corNum3, corNum4, corNum5, corNum6, corNum7, corNum8, corNum9, corNum10 = corNums

            loss = criterion(x_recon[4], input_imgs) + config["alpha"]*criterion(y_recon[4], side_imgs)+ \
            config["lam_h"] * loss_H[0] + config["lam_h"] * loss_H[1]

            elapsed_time = time.time() - start_time

            psnr = psnr_calculate(input_imgs, x_recon[4])

            values = [loss, loss_H[0],loss_H[1],psnr, elapsed_time, SNR_TEST1.mean(), SNR_TEST2.mean(), corNum1, corNum2,
                      corNum3, corNum4, corNum5, corNum6, corNum7, corNum8, corNum9, corNum10, sf[0], sf[1], sf[2], sf[3], sf[4], sf[5]]
            for metric, value in zip(metrics, values):
                metric.update(value)


        metrics_dic = [
            ('Val Loss', val_losses, 4),
            ('loss_H_X', loss_H_X_LIST, 4),
            ('loss_H_Y', loss_H_Y_LIST, 4),
            ('Time', elapsed, 4),
            ('PSNR', val_psnrs, 4),
            ('Cor1', cor1, 4), ('Cor2', cor2, 4), ('Cor3', cor3, 4), ('Cor4', cor4, 4), ('Cor5', cor5, 4),
            ('Cor6', cor6, 4), ('Cor7', cor7, 4), ('Cor8', cor8, 4), ('Cor9', cor9, 4), ('Cor10', cor10, 4),
            ('SNR1', snr1, 4), ('SNR2', snr2, 4),
            ('SF1', sf1, 6), ('SF2', sf2, 6), ('SF3', sf3, 6), ('SF4', sf4, 6), ('SF5', sf5, 6), ('SF6', sf6, 6)
        ]

        log = ' | '.join([
            f'Epoch {epoch}',
            *(f'{name} {metric.val:.{precision}f}({metric.avg:.{precision}f})' for name, metric, precision in metrics_dic),
            f'BEST_LOSS {BEST_LOSS}'
        ])
        logger.info(log)

        if config["use_wandb"]:
            log_dict = {"PSNR": val_psnrs.avg, "loss": val_losses.avg}
            wandb.log(log_dict)
        writter.add_scalar("val_loss", val_losses.avg, global_step=epoch)
        val = val_losses.avg
        for i in metrics:
            i.clear()
    return val

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
    # ===================Initialize SNR, Model, criterion, optimizer
    jscc = Jscc(config=config).to(config["device"])
    BEST_LOSS = False
    best_loss = float("inf")
    if config["metric"] == "MSE":
        criterion = MSELoss()  # MSE_Loss(lmbda=args.lmbda)
    elif config["metric"] == "MS_SSIM":
        criterion = MS_SSIM_Loss(config["device"])  # (device, lmbda=args.lmbda)
        # criterion = MSSSIMLoss().to(config["device"])
    elif config["metric"] == "LPIPS":
        criterion = LPIPSLoss().to(config["device"])
    else:
        raise Exception("Criterion not found")
    last_epoch = 0
    optimizer = torch.optim.Adam(jscc.parameters(), lr=config["learning_rate"])
    scheduler = lr_scheduler.MultiStepLR(optimizer, [100, 200],
                                   gamma=0.5)  # optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.2)
    # ===================Resume
    if config["resume"] == True:
        save_name = "{}_{}_awgn_200".format(config["model_name"], config["dataset_name"])
        checkpoints = torch.load(workdir + "/models/" + save_name + ".model", map_location=config["device"])
        jscc.load_state_dict(checkpoints["models"])
        optimizer.load_state_dict(checkpoints["optimizer"])
    # ===================Train
    logger.info("======Begin to train: " + log_filename + "======")
    for epoch in range(last_epoch, config["epochs"]):
        # print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(jscc, criterion, optimizer, epoch)
        scheduler.step()
        if epoch % 5 == 0 or epoch == (config["epochs"] - 1):
            val_loss = val_epoch(jscc, criterion, epoch, BEST_LOSS)
            BEST_LOSS = False
            # ===================Save Model
            # checkpoint = {
            #     "models": jscc.state_dict(),
            #     "optimizer": optimizer.state_dict(),
            #     "epoch": epoch,
            # }
            # if val_loss < best_loss and epoch != (config["epochs"] - 1):
            if val_loss < best_loss:
                BEST_LOSS = True
                save_model_name = "{}_{}_awgn.model".format(config["model_name"], config["dataset_name"])
                torch.save(jscc.state_dict(), workdir + "/models/" + save_model_name)
                best_loss = val_loss
            # elif epoch == (config["epochs"] - 1):
            #     save_model_name = "{}_{}_awgn_{}.model".format(config["model_name"], config["dataset_name"], config["epochs"])
            #     torch.save(checkpoint, workdir + "/models/" + save_model_name)
            #     best_loss = val_loss
    writter.close()
    logger.info("Model training completed")

    print('evaluating...')
    test_snrs = config["test_snrs"]
    save_name = "{}_{}_awgn".format(config["model_name"], config["dataset_name"])
    checkpoints = torch.load(workdir + "/models/" + save_name + ".model", map_location=config["device"])
    # jscc.load_state_dict(checkpoints["models"])
    jscc.load_state_dict(checkpoints)
    test(jscc, True, test_snrs, config)
    test(jscc, False, test_snrs, config)
    logger.info("Model testing completed")


if __name__ == "__main__":
    main()