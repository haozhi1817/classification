'''
Author: HaoZhi
Date: 2022-08-22 16:46:59
LastEditors: HaoZhi
LastEditTime: 2022-10-14 09:38:55
Description: 
'''
import os

import torch

from data import build_dataloader
from model import build_backbone, build_loss
from utils.utils import read_csv, metric, save_result

valid_folder = r"D:\workspace\convnext\dataset\test"
valid_label_csv_path = r"d:\workspace\convnext\dataset\sample_submit.csv"
devic = "cuda:0"
batch_size = 3
img_size = 224

backbone_name = "resnet50"
num_class = 3

reseum_path = r'D:\workspace\convnext\ckpt_resnet_coswarm\model_89.pth'

result_path = r"D:\workspace\convnext\result_89"
result_csv_name = "resnet_coswarm_89.csv"


def main():
    # device

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    device = torch.device(devic)

    # dataset

    valid_loader = build_dataloader(
        valid_folder, batch_size=batch_size * 3, img_size=img_size, mode="valid"
    )
    valid_label_dict = read_csv(valid_label_csv_path)

    # model
    backbone = build_backbone(
        model_name=backbone_name, img_size=img_size, num_class=num_class
    ).to(device)

    checkpoint = torch.load(reseum_path)
    backbone.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        backbone.eval()
        valid_pred_dict = {}
        for idx, (imgs, pathes) in enumerate(valid_loader):
            imgs = imgs.to(device)
            logits = backbone(imgs)
            preds = torch.argmax(logits, axis=-1).detach().cpu().numpy()
            for pred, path in zip(preds, pathes):
                valid_pred_dict[os.path.split(path)[-1]] = pred

        # save_result(
        #     label_dict=valid_label_dict,
        #     pred_dict=valid_pred_dict,
        #     csv_path=os.path.join(result_path, result_csv_name),
        # )

        print("=======================================")
        print("Valid Confusion Metric: ")
        print(metric(pred_dict=valid_pred_dict, label_dict=valid_label_dict))
        print("=======================================")


if __name__ == "__main__":
    main()
