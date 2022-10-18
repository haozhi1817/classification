"""
Author: HaoZhi
Date: 2022-08-02 13:52:57
LastEditors: HaoZhi
LastEditTime: 2022-08-02 17:15:37
Description: 
"""
import os
import yaml

import torch
from torch.utils.tensorboard import SummaryWriter

from data import build_dataloader_unbalance
from model import build_backbone, build_loss
from utils.utils import read_csv, metric

train_folder = r"D:\workspace\convnext\dataset\train"
valid_folder = r"D:\workspace\convnext\dataset\test"
valid_label_csv_path = r"d:\workspace\convnext\dataset\sample_submit.csv"
devic = 'cuda:0'
batch_size = 12
img_size = 224

backbone_name = 'convnext'
loss_name = 'ce_loss'
num_class = 3

lr = 1e-3
wd = 1e-5
num_epoch = 100
reseum_path = False

log_path = r"D:\workspace\convnext\log_convnext_unbalance"
ckpt_path = r"D:\workspace\convnext\ckpt_convnext_unbalance"

def main():
    # device
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    device = torch.device(devic)

    # dataset
    train_loader = build_dataloader_unbalance(train_folder, batch_size= batch_size, img_size= img_size, mode = 'train')
    #train_label_dict = get_label_dict(train_folder)

    valid_loader = build_dataloader_unbalance(valid_folder, batch_size= batch_size * 3, img_size= img_size, mode = 'valid')
    valid_label_dict = read_csv(valid_label_csv_path)

    num_batch = len(train_loader)

    # model
    backbone = build_backbone(model_name = backbone_name, img_size = img_size, num_class = num_class).to(device)
    loss_op = build_loss(loss_name= loss_name)

    optimizer = torch.optim.Adam(
        backbone.parameters(),
        lr=lr,
        weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max= num_epoch,
        eta_min= 1e-6,
    )

    if reseum_path:
        checkpoint = torch.load(reseum_path)
        backbone.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    writer = SummaryWriter(log_dir= log_path)

    for epoch in range(num_epoch):
        backbone.train()
        train_pred_dict = {}
        train_label_dict = {}
        for idx, (imgs, labels, pathes) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = backbone(imgs)
            loss = loss_op(logits, labels)
            preds = torch.argmax(logits, axis = -1).detach().cpu().numpy()
            for (pred, label, path) in zip(preds, labels.cpu().numpy(), pathes):
                train_pred_dict[os.path.split(path)[-1]] = pred
                train_label_dict[os.path.split(path)[-1]] = label

            writer.add_scalar(
                "lr", scheduler.get_last_lr()[0], global_step=epoch * num_batch + idx
            )
            writer.add_scalar(
                "ce_loss", loss, global_step=epoch * num_batch + idx
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                "|Epoch = {:>5d} | Steps = {:>5d} | Lr = {:.3e} | Loss = {:.5f}|".format(
                    epoch,
                    epoch * num_batch + idx,
                    scheduler.get_last_lr()[0],
                    loss.item(),
                )
            )
        scheduler.step()
        checkpoint = {
            "model_state_dict": backbone.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(
            checkpoint,
            os.path.join(ckpt_path, "model_" + str(epoch) + ".pth"
            ),
        )

        with torch.no_grad():
            backbone.eval()
            valid_pred_dict = {}
            for idx, (imgs, pathes) in enumerate(valid_loader):
                imgs = imgs.to(device)
                logits = backbone(imgs)
                preds = torch.argmax(logits, axis = -1).detach().cpu().numpy()
                for pred, path in zip(preds, pathes):
                    valid_pred_dict[os.path.split(path)[-1]] = pred

        print('===========================Epoch %s ===================================' %(epoch))
        print('Train Confusion Metric: ')
        print(metric(pred_dict= train_pred_dict, label_dict= train_label_dict))
        print('=======================================')
        print('Valid Confusion Metric: ')
        print(metric(pred_dict= valid_pred_dict, label_dict= valid_label_dict))
        print('===========================Epoch =====================================')


if __name__ == "__main__":
    main()