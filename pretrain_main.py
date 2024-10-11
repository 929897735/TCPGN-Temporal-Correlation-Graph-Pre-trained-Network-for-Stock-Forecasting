from model.model import TCGPN
from utils.utils import DataGenerator
from utils.train_utils import train_pretrain

import numpy as np  
import os
import toml
import torch

settings = toml.load("./config/config.toml")


if __name__ == "__main__":
    dataset_name='CSI300'
    data_path= f'./data/{dataset_name}/'

    features=torch.load(os.path.join(data_path,'features.pt'))
    labels=torch.load(os.path.join(data_path,'labels.pt'))

    (turnover_pos,industry_pos,vali_label_pos)= np.load(os.path.join(data_path,'idx.npy'))


    reg_capital = torch.load(os.path.join(data_path,'reg.pt'))

    os.environ["CUDA_VISIBLE_DEVICES"] = settings["cuda_num"]

    len_time_step=features.shape[1]


    train_date_begin = settings["data_config"]["time_step"] - 1
    train_date_end =  int(len_time_step*0.80)

    val_date_begin = train_date_end
    val_date_end = len_time_step

    model = TCGPN(settings=settings["model_config"])

    train_data_generator = DataGenerator(
        date_begin=train_date_begin,
        date_end=train_date_end,
        features=features,
        labels=labels,
        time_step=settings["data_config"]["time_step"],
        reg_capital=reg_capital,
        stock_sample_time=settings["data_config"]["stock_sample_time"],
        stock_sample_num=settings["data_config"]["stock_sample_num"],
        mask_sample_time=settings["data_config"]["mask_sample_time"],
        temporal_mask_ratio=settings["data_config"]["temporal_mask_ratio"],
        graph_mask_ratio=settings["data_config"]["graph_mask_ratio"],
        graph_type=settings["data_config"]["graph_type"],
        turnover_pos=turnover_pos,
        industry_pos=industry_pos,
        vali_label_pos=vali_label_pos,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data_generator, batch_size=settings["batch_size"], shuffle=False
    )

    if settings["use_val"]:
        val_data_generator = DataGenerator(
            date_begin=val_date_begin,
            date_end=val_date_end,
            features=features,
            labels=labels,
            time_step=settings["data_config"]["time_step"],
            reg_capital=reg_capital,
            stock_sample_time=settings["data_config"]["stock_sample_time"],
            stock_sample_num=settings["data_config"]["stock_sample_num"],
            mask_sample_time=settings["data_config"]["mask_sample_time"],
            temporal_mask_ratio=settings["data_config"]["temporal_mask_ratio"],
            graph_mask_ratio=settings["data_config"]["graph_mask_ratio"],
            graph_type=settings["data_config"]["graph_type"],
            turnover_pos=turnover_pos,
            industry_pos=industry_pos,
            vali_label_pos=vali_label_pos,
            use_rand=False
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_data_generator, batch_size=settings["batch_size"], shuffle=False
        )
    else:
        val_loader = None


    model = train_pretrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        settings=settings["train_config"],
        use_val=settings["use_val"],
        vali_label_pos=vali_label_pos,
    )
