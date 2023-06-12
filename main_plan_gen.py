import torch
from pose_data_loader import PoseDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
# from models.RawTransformerAndLSTM import LSTMTaggerSep, TransformerTraggerSeq, KimoreFusionModel
from models.PlanGenerateNet import LSTMTaggerSep, TransformerTraggerSeq, KimoreFusionModel
import torch.optim as optim
from loss import Shrinkage_loss, nn
from pck_metrics import keypoint_pck_accuracy


def point_change(raw_xyz, dim=3):
    raw_xyz = raw_xyz.cpu()
    raw_xyz = raw_xyz.detach().numpy()
    d = raw_xyz.shape[0] // 3
    xyz = raw_xyz.reshape(d, dim)
    return xyz


def run(mode='train'):
    path_root = "../Kimore"
    device = torch.device("mps")
    if mode == 'train':
        train_dataset = PoseDataset(path_root, mode=mode)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
    elif mode == 'test':
        test_dataset = PoseDataset(path_root, mode=mode)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    else:
        print('Mode error')
        return False
    
    # max_epoch = 200
    max_epoch = 20
    num_layer = 4
    input_embeddings = 256
    hidden_embeddings = 128
    criterion_MSE = torch.nn.MSELoss().to(device)
    # model = LSTMTaggerSep(input_embeddings, hidden_embeddings, lstm_num_layers = num_layer)
    model = KimoreFusionModel(input_embeddings, hidden_embeddings, num_layers = num_layer)
    model = model.to(device)

    learning_rate = 0.001
    momentum = 0.95
    weight_decay = 1e-4
    scheduler_gamma = 0.95
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, scheduler_gamma)

    shrinkage_a = 5
    shrinkage_c = 0.2
    # criterion_1 = Shrinkage_loss(shrinkage_a, shrinkage_c).to(device)
    criterion = nn.MSELoss().to(device)
    best_loss = 1000

    point_dim = 3

    if mode == 'train':
        model.train()
    elif mode == 'test':
        state_dict = torch.load('./trained_param/best_performance_param.pth')
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print("Mode error, neither train nor test")
        return False
    
    if mode == 'test':
        max_epoch = 1

    for epoch in range(max_epoch):
        if mode == 'train':
            data_iter = iter(train_dataloader)
        elif mode == 'test':
            data_iter = iter(test_dataloader)
        else:
            return False
        
        sum_loss = 0.0
        print("The epoch is: ")
        print(epoch)
        for i, data in enumerate(data_iter):
            if mode == 'test' and i == 5:
                continue

            isFirstPose = True
            human_pose_numpy, robot_force_numpy, robot_pose_numpy, human_xyz_numpy, human_quat_numpy, robot_xyz_numpy, robot_velocity_numpy = data
            robot_xyz_prev_pose = torch.from_numpy(np.zeros([1, robot_xyz_numpy.shape[2]]))
            robot_velocity_prev_pose = torch.from_numpy(np.zeros([1, robot_velocity_numpy.shape[2]]))
            robot_force_prev_pose = torch.from_numpy(np.zeros([1, robot_force_numpy.shape[2]]))

            ground_truth_xyz_list = []
            ground_truth_vel_list = []
            ground_truth_for_list = []

            output_pred_xyz_list = []
            output_pred_vel_list = []
            output_pred_for_list = []

            pred_xyz_points_list = []
            gt_pred_xyz_points_list = []
            pred_vel_points_list = []
            gt_pred_vel_points_list = []

            loss = torch.tensor(0.0, device = device, requires_grad=True)
            loss_xyz = torch.tensor(0.0, device = device, requires_grad=True)
            loss_vel = torch.tensor(0.0, device = device, requires_grad=True)
            loss_for = torch.tensor(0.0, device = device, requires_grad=True)

            print(f"human pose data number: {human_pose_numpy.shape[1]}")

            for j in range(human_pose_numpy.shape[1]):
                human_xyz_numpy_ = human_xyz_numpy[0]
                human_xyz_single_ = human_xyz_numpy_[j]
                human_xyz_single = human_xyz_single_[None, :]

                robot_xyz_numpy_ = robot_xyz_numpy[0]
                robot_velocity_numpy_ = robot_velocity_numpy[0]
                robot_force_numpy_ = robot_force_numpy[0]

                ground_truth_xyz = robot_xyz_numpy_[j]
                ground_truth_velocity = robot_velocity_numpy_[j]
                ground_truth_force = robot_force_numpy_[j]

                ground_truth_xyz = ground_truth_xyz.float().to(device)
                ground_truth_velocity = ground_truth_velocity.float().to(device)
                ground_truth_force = ground_truth_force.float().to(device)

                optimizer.zero_grad()

                if isFirstPose == True:
                    isFirstPose = False
                else:
                    robot_force_prev_pose = robot_force_numpy_[j - 1]
                    robot_force_prev_pose = robot_force_prev_pose[None, :]
                    robot_xyz_prev_pose = robot_xyz_numpy_[j - 1]
                    # robot_xyz_prev_pose = output_pred_xyz_list[j - 1].cpu()
                    robot_xyz_prev_pose = robot_xyz_prev_pose[None, :]
                    robot_velocity_prev_pose = robot_velocity_numpy_[j - 1]
                    # robot_velocity_prev_pose = output_pred_vel_list[j - 1].cpu()
                    robot_velocity_prev_pose = robot_velocity_prev_pose[None, :]

                robot_xyz_velocity_merge = torch.cat([human_xyz_single, robot_force_prev_pose, robot_xyz_prev_pose, robot_velocity_prev_pose], dim = 1)
                robot_xyz_velocity_merge = robot_xyz_velocity_merge[None, :, :]
                robot_xyz_velocity_merge = robot_xyz_velocity_merge.float().to(device)
                # human_robot_xyz_merge = robot_xyz_velocity_merge[None, :]
                human_xyz_single = human_xyz_single[None, :]
                preds_xyz, preds_vel = model(robot_xyz_velocity_merge)

                preds_xyz = preds_xyz.squeeze()
                preds_vel = preds_vel.squeeze()

                output_pred_xyz_list.append(preds_xyz)
                output_pred_vel_list.append(preds_vel)

                ground_truth_xyz_list.append(ground_truth_xyz)
                ground_truth_vel_list.append(ground_truth_velocity)

            N = 0
            
            for j in range(len(output_pred_xyz_list)):
                N += 1
                loss_xyz = loss_xyz + criterion(output_pred_xyz_list[j], ground_truth_xyz_list[j])
                loss_vel = loss_vel + criterion(output_pred_vel_list[j], ground_truth_vel_list[j])
                pred_xyz_points_list.append(output_pred_xyz_list[j].cpu().detach().numpy())
                pred_vel_points_list.append(output_pred_vel_list[j].cpu().detach().numpy())
                gt_pred_xyz_points_list.append(ground_truth_xyz_list[j].cpu().detach().numpy())
                gt_pred_vel_points_list.append(ground_truth_vel_list[j].cpu().detach().numpy())

            loss = loss_xyz + loss_vel #+ 0.02 * loss_for
            if mode == 'train':
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            sum_loss += loss.item()

            tmp = pred_xyz_points_list[0].shape[0]
            K = int(tmp // point_dim)
            pred_xyz_topck = np.array(pred_xyz_points_list).reshape(N, K, point_dim)
            pred_vel_topck = np.array(pred_vel_points_list).reshape(N, K, point_dim)
            gt_pred_xyz_topck = np.array(gt_pred_xyz_points_list).reshape(N, K, point_dim)
            gt_pred_vel_topck = np.array(gt_pred_vel_points_list).reshape(N, K, point_dim)
            mask = np.ones((N, K), dtype=bool)
            normalize = np.ones((N, point_dim), dtype=float)

            xyz_acc, xyz_avg_acc, xyz_cnt = keypoint_pck_accuracy(
                pred=pred_xyz_topck, 
                gt=gt_pred_xyz_topck, 
                thr=0.1, 
                mask=mask, 
                normalize=normalize
            )
            xyz_pck_res = f"xyz_acc: {xyz_acc}\nxyz_avg_acc: {xyz_avg_acc}\nxyz_cnt: {xyz_cnt}"
            print(xyz_pck_res)

            vel_acc, vel_avg_acc, vel_cnt = keypoint_pck_accuracy(
                pred=pred_vel_topck, 
                gt=gt_pred_vel_topck, 
                thr=0.1, 
                mask=mask, 
                normalize=normalize
            )
            vel_pck_res = f"vel_acc: {vel_acc}\nvel_avg_acc: {vel_avg_acc}\nvel_cnt: {vel_cnt}"
            print(vel_pck_res)
            print("------***------")

            
        sum_loss = sum_loss / 100.0
        if mode == 'train':
            if sum_loss < best_loss:
                best_loss = sum_loss
                torch.save(model.state_dict(), "./trained_param/best_performance_param.pth")

            print(f"Training: sum loss {sum_loss}, xyz_loss: {loss_xyz}, vel_loss: {loss_vel}")
        else:
            print(f"Testing: sum loss {sum_loss}, xyz_loss: {loss_xyz}, vel_loss: {loss_vel}")
        print("\n")
    

if __name__ == "__main__":
    run()
    run('test')