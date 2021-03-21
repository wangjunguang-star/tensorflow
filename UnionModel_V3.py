import os
import numpy as np
import pandas as pd
import sklearn
import torch
import tensorwatch as tw
from torchvision.models import AlexNet
from torchviz import make_dot

import random
import matplotlib
from utils import *

from data_utils import MultiLabelDataset, LoadData, TensorDataset
from nn_models import KH_MLP_MLT, CustomMSELoss, KH_MLP,  EarlyStopping
from nn_models import MODEL as MODEL
from nn_models import MultiTaskLossWrapper

torch.manual_seed(2)
torch.set_num_threads(6) 

def train(epochs, train_data_loader, test_data_loader
    , x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, is_transfer=False, l2=0.001):

    if not is_transfer:
        print("/******************* init new model **********************/ ")
        model = MODEL(input_sz1=9, input_sz2=2, output_sz=5)
        # tw.draw_model(model, [1, 8, 2])
    else:
        print("/******************* load pre-trained model *******************/")
        model = MODEL(input_sz1=9, input_sz2=2, output_sz=5)
        model.load_state_dict(torch.load("/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/models/m.pt"))
    print(model)
    # print([x for x in model.temp_head.parameters()])
    model_wrapper = MultiTaskLossWrapper(5, model)
    
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=5*1e-5, weight_decay=l2)
    # optimizer = torch.optim.Adam([{"params": model.pn_head.parameters(), "lr":0.1*1e-4}
    #                             , {"params": model.fc1.parameters()}
    #                             , {"params": model.fc2.parameters()}
    #                             , {"params": model.fc3.parameters()}
    #                             , {"params": model.fc4.parameters()}
    #                             , {"params": model.temp_head.parameters()}
    #                             , {"params": model.thc_head.parameters()}
    #                             , {"params": model.nox_head.parameters()}
    #                             , {"params": model.cov_head.parameters()}]
    #                             , lr=0.1*1e-3, weight_decay=l2)
    loss_mse = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_losses = []
    test_losses = []
    for e in range(epochs):
        model.train()
        for x_, y_ in train_data_loader:
            optimizer.zero_grad()
            x_params = x_[:, 2:]
            x_khs = x_[:, 0:2]
            loss, log_vars = model_wrapper.forward(x_params, x_khs, y_)
            loss.backward()
            optimizer.step()
        model.eval()
        p_train = model(x_train_tensor[:, 2:], x_train_tensor[:, 0:2])
        # print(p_train.shape, y_train_tensor.shape)
        train_lss = loss_mse(p_train, y_train_tensor)
        p_test = model(x_test_tensor[:, 2:], x_test_tensor[:, 0:2])
        test_lss = loss_mse(p_test, y_test_tensor)

        train_losses.append(train_lss)
        test_losses.append(test_lss)
        if e% 10 == 0:
            print("%d epoch, train loss: %f, test loss: %f"%(e, train_lss.item(), test_lss.item()))
    
        # early_stopping(test_lss.item(), model)
        # if early_stopping.early_stop:
        #     #print("Early Stopping")
        #     break
    
    torch.save(model.state_dict(), "/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/models/union_model.pt")
    print("%d epoch, train loss: %f, test loss: %f"%(e, train_lss.item(), test_lss.item()))
    
    return train_losses, test_losses, p_train, p_test

def tmp_plot(y_test, p_test, super_title, save_path):
    temps_true = y_test.detach().numpy()[:, 0].tolist()
    pn_true = y_test.detach().numpy()[:, 1].tolist()
    nox_true = y_test.detach().numpy()[:, 2].tolist()
    thc_true = y_test.detach().numpy()[:, 3].tolist()
    cov_true = y_test.detach().numpy()[:, 4].tolist()
    
    temps_pred = p_test.detach().numpy()[:, 0].tolist()
    pn_pred = p_test.detach().numpy()[:, 1].tolist()
    nox_pred = p_test.detach().numpy()[:, 2].tolist()
    thc_pred = p_test.detach().numpy()[:, 3].tolist()
    cov_pred = p_test.detach().numpy()[:, 4].tolist()

    # plt_sort(temps_true, temps_pred, "%s temp "%indicator)
    # plt_sort(pn_true, pn_pred, "%s pn "%indicator)
    # plt_sort(nox_true, nox_pred, "%s nox "%indicator)
    # plt_sort(thc_true, thc_pred, "%s thc "%indicator)
    plot_sort(t=[temps_true, temps_pred], pn=[pn_true, pn_pred], nox=[nox_true, nox_pred], thc=[thc_true, thc_pred], cov=[cov_true, cov_pred]
                , super_title=super_title, save_path=save_path)

"""
联合建模探索
使用全空间数据建模，在传统数据上测试
"""
if __name__ == "__main__":

    # 加载数据
    load_data = LoadData(path="/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/datas_v2/1000x3.5.csv")
    load_data.load_data()
    ds_1000_35 = load_data.norm_data()
    test_ds_1000_35 = load_data.norm_test_data("/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/datas_v2/Test_1000_3.5.txt")
    print("1000x3.5 dataset size: ", ds_1000_35.shape)
    print("1000x3.5 test dataset size: ", test_ds_1000_35.shape)

    load_data_1 = LoadData(path="/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/datas_v2/1300x2.5.csv")
    load_data_1.load_data()
    ds_1300_25 = load_data_1.norm_data()
    test_ds_1300_25 = load_data_1.norm_test_data("/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/datas_v2/Test_1300_2.5.txt")
    print("1300x2.5 dataset size: ", ds_1300_25.shape)
    print("1300x2.5 test dataset size: ", test_ds_1300_25.shape)

    load_data_2 = LoadData(path="/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/datas_v2/1000x2.5.csv")
    load_data_2.load_data()
    ds2 = load_data_2.norm_data()#.head(200)
    ds_1000_25 = sklearn.utils.shuffle(ds2, random_state=1236)
    test_ds_1000_25 = load_data_2.norm_test_data("/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/datas_v2/Test_1000_2.5.txt")
    print("1000x2.5 dataset size： ", ds_1000_25.shape)
    print("1000x2.5 test dataset size： ", test_ds_1000_25.shape)

    print("******************************************************************************************************************************************")

    # 训练部分
    test_single = False
    if test_single:
        """ 使用1000x3.5的数据预训练, 注意增压器对燃烧的影响 """
        mlt_ds_1 = MultiLabelDataset()
        train_data_loader, _ = mlt_ds_1.split_train_and_test(ds_1000_35, 10, 0.99)
        x_train_tensor_1000_35, y_train_tensor_1000_35, _, _ = mlt_ds_1.get_train_tensor()
        _, test_data_loader = mlt_ds_1.split_train_and_test(test_ds_1000_35, 10, 0.01)
        _, _, x_test_tensor_1000_35, y_test_tensor_1000_35 = mlt_ds_1.get_train_tensor()
        print(x_train_tensor_1000_35.shape)
        print(x_test_tensor_1000_35.shape)
        train_losses_, test_losses_, p_train, p_test = train(4500
                                                            , train_data_loader, test_data_loader
                                                            , x_train_tensor_1000_35, y_train_tensor_1000_35
                                                            , x_test_tensor_1000_35, y_test_tensor_1000_35
                                                            , is_transfer=False
                                                            , l2=0.001)
        plt_loss(train_losses_, test_losses_, title="1000x3.5 loss")
        tmp_plot(y_test_tensor_1000_35, p_test
                , super_title="1000x3.5"
                , save_path="/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/pics/1000x3.5.png")

        # """ 使用1300x2.5的数据预训练 """
        # mlt_ds_2 = MultiLabelDataset()
        # train_data_loader, _ = mlt_ds_2.split_train_and_test(ds_1300_25, 10, 0.98)
        # x_train_tensor_1300_25, y_train_tensor_1300_25, _, _ = mlt_ds_2.get_train_tensor()
        # _, test_data_loader = mlt_ds_2.split_train_and_test(test_ds_1300_25, 10, 0.02)
        # _, _, x_test_tensor_1300_25, y_test_tensor_1300_25 = mlt_ds_2.get_train_tensor()
        # print(x_train_tensor_1300_25.shape)
        # print(x_test_tensor_1300_25.shape)
        # train_losses_, test_losses_, p_train, p_test = train(1500
        #                                                     , train_data_loader, test_data_loader
        #                                                     , x_train_tensor_1300_25, y_train_tensor_1300_25
        #                                                     , x_test_tensor_1300_25, y_test_tensor_1300_25
        #                                                     , is_transfer=False)
        # plt_loss(train_losses_, test_losses_, title="1300x2.5 loss")
        # tmp_plot(y_test_tensor_1300_25, p_test
        #         , super_title="1300x2.5"
        #         , save_path="/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/pics/1300x2.5.png")
        # tmp_plot(y_train_tensor_1300_25, p_train
        #         , super_title="1300x2.5"
        #         , save_path="/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/pics/train 1300x2.5.png")

        # """ 使用1000x2.5的数据预训练 """
        # mlt_ds = MultiLabelDataset()
        # train_data_loader, _ = mlt_ds.split_train_and_test(ds_1000_25, 10, 0.99)
        # x_train_tensor_1000_25, y_train_tensor_1000_25, _, _ = mlt_ds.get_train_tensor()
        # _, test_data_loader = mlt_ds.split_train_and_test(test_ds_1000_25, 10, 0.02)
        # _, _, x_test_tensor_1000_25, y_test_tensor_1000_25 = mlt_ds.get_train_tensor()
        # print(x_train_tensor_1000_25.shape)
        # print(x_test_tensor_1000_25.shape)
        
        # train_losses_, test_losses_, p_train, p_test = train(2500
        #                                                     , train_data_loader, test_data_loader
        #                                                     , x_train_tensor_1000_25, y_train_tensor_1000_25
        #                                                     , x_test_tensor_1000_25, y_test_tensor_1000_25
        #                                                     , is_transfer=False
        #                                                     , l2=0.001)
        # plt_loss(train_losses_, test_losses_, title="200 - 1000x2.5 loss")
        # tmp_plot(y_test_tensor_1000_25, p_test
        #         , super_title="200 - 1000x2.5"
        #         , save_path="/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/pics/200 - 1000x2.5.png")
        # # tmp_plot(y_test_tensor_1000_25, p_test
        # #         , super_title="1000x2.5"
        # #         , save_path="/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/pics/1000x2.5.png")

    print("**************************************************************************************************************************")

    # # 综合建模
    if not test_single:
        load_data = LoadData(path="/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/datas_v2/1000x2.5.csv")
        load_data.load_data()
        ds = load_data.norm_data()#.head(200)
        print(load_data.data_scale_info)
        test_ds = load_data.norm_test_data("/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/datas_v2/Test_1000_2.5.txt")
        mlt_ds = MultiLabelDataset()
        train_data_loader, _ = mlt_ds.split_train_and_test(ds, 11, 0.999)
        x_train_tensor, y_train_tensor, _, _ = mlt_ds.get_train_tensor()
        mtl_ds_test = MultiLabelDataset()
        _, test_data_loader = mtl_ds_test.split_train_and_test(test_ds, 11, 0.01)
        _, _, x_test_tensor, y_test_tensor = mtl_ds_test.get_train_tensor()
        print(x_train_tensor.shape)
        print(x_test_tensor.shape)
        train_losses_, test_losses_, p_train, p_test = train(720
                                                            , train_data_loader, test_data_loader
                                                            , x_train_tensor, y_train_tensor
                                                            , x_test_tensor, y_test_tensor
                                                            , is_transfer=False)
        plt_loss(train_losses_, test_losses_, title="1000x2.5 loss")
        tmp_plot(y_test_tensor, p_test
                , super_title="1000x2.5 pred 1000x2.5"
                , save_path="/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/pics/1000x2.5 pred 1000x2.5 .png")
        
        # print(p_test[:, -1], y_test_tensor[:, -1])
        # import netron
        # model_path = "/home/nvidia/Documents/znbd_k7/program/wangjunguang/KH_25_model_VIP/NN/models/union_model.pt"
        # netron.start(model_path)

    #     x_train_tensor = torch.cat([x_train_tensor_1000_35, x_train_tensor_1000_25], dim=0)
    #     y_train_tensor = torch.cat([y_train_tensor_1000_35, y_train_tensor_1000_25], dim=0)
    #     train_data_loader = torch.utils.data.DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    #     test_data_loader = torch.utils.data.DataLoader(TensorDataset(x_test_tensor_1000_35, y_test_tensor_1000_35), batch_size=64, shuffle=True)
    #     train_losses_, test_losses_, p_train, p_test = train(15000
    #                                                         , train_data_loader, test_data_loader
    #                                                         , x_train_tensor, y_train_tensor
    #                                                         , x_test_tensor_1000_35, y_test_tensor_1000_35
    #                                                         , is_transfer=False)
    #     plt_loss(train_losses_, test_losses_, title="1000x2.5 & 1000x3.5  pred 1000x3.5 loss")
    #     tmp_plot(y_test_tensor_1000_35, p_test, indicator="1000x2.5 & 1000x3.5 pred 1000x3.5")

