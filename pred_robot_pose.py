import torch
from pose_data_loader import PoseDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PlanGenerateNet import LSTMTagger, LSTMTaggerSep, LSTMPreSep
import torch.optim as optim
from loss import Shrinkage_loss, nn
from pck_metrics import keypoint_pck_accuracy

def main():
    path_root = "../Kimore"
    device = torch.device("cuda:0")
    num_layer = 4
    input_embeddings = 256
    hidden_embeddings = 128
    train_dataset = PoseDataset(path_root)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)
    model = LSTMPreSep(input_embeddings, hidden_embeddings, lstm_num_layers = num_layer)
    # model = TransformerTraggerSeq(input_embeddings, hidden_embeddings, num_layers = num_layer)
    model.load_state_dict(torch.load("./trained_param/best_performance_param.pth"),  strict = True)
    # model = torch.load("./trained_param/best_performance.pth")
    model.eval()
    model = model.to(device)
    
    title = "num,force,x,y,z,vx,vy,vz\n"
    train_iter = iter(train_dataloader)

    with torch.no_grad():
        for i, data in enumerate(train_iter):
            if i == 74:
                continue
            human_pose_numpy, robot_force_numpy, robot_pose_numpy, human_xyz_numpy, human_quat_numpy, robot_xyz_numpy, robot_velocity_numpy = data
            robot_xyz_prev_pose = torch.from_numpy(np.zeros([1, robot_xyz_numpy.shape[2]]))
            robot_velocity_prev_pose = torch.from_numpy(np.zeros([1, robot_velocity_numpy.shape[2]]))
            robot_force_prev_pose = torch.from_numpy(np.zeros([1, robot_force_numpy.shape[2]]))
            output_pred_list = []

            # human_pose_numpy
            for j in range(human_pose_numpy.shape[1]):
                robot_12_pose = {}
                human_xyz_numpy_ = human_xyz_numpy[0]
                #print(human_xyz_numpy_.shape)
                human_xyz_single_ = human_xyz_numpy_[j]
                human_xyz_single = human_xyz_single_[None, :]

                robot_xyz_numpy_ = robot_xyz_numpy[0]
                robot_velocity_numpy_ = robot_velocity_numpy[0]
                robot_force_numpy_ = robot_force_numpy[0]
            
                robot_force_prev_pose = robot_force_numpy_[j - 1]
                robot_force_prev_pose = robot_force_prev_pose[None, :]
                robot_xyz_prev_pose = robot_xyz_numpy_[j - 1]
                robot_xyz_prev_pose = robot_xyz_prev_pose[None, :]
                robot_velocity_prev_pose = robot_velocity_numpy_[j - 1]
                robot_velocity_prev_pose = robot_velocity_prev_pose[None, :]
                ground_truth_force = robot_force_numpy_[j]
                # print(ground_truth_force)

                robot_xyz_velocity_merge = torch.cat([human_xyz_single, robot_force_prev_pose, robot_xyz_prev_pose, robot_velocity_prev_pose], dim = 1)
                robot_xyz_velocity_merge = robot_xyz_velocity_merge[None, :, :]
                robot_xyz_velocity_merge = robot_xyz_velocity_merge.float().to(device)
                # print(robot_xyz_velocity_merge)  # The input here is the same in every time

                preds_xyz, preds_vel = model(robot_xyz_velocity_merge)

                preds_xyz = preds_xyz.squeeze()
                preds_vel = preds_vel.squeeze()
                # print(preds_xyz)
                # print(preds_vel)
                # print(ground_truth_force)

                g = 0
                for k in range(12):
                    if k < 6:
                        robot_num = f"a{k+1}"
                    else:
                        robot_num = f"g{k-6+1}"
                    robot_pose = []
                    robot_pose.append(float(ground_truth_force[k]))
                    for s in range(3):
                        robot_pose.append(float(preds_xyz[g+s]))
                        robot_pose.append(float(preds_vel[g+s]))
                    # print(robot_pose)
                    robot_12_pose[robot_num] = robot_pose
                    g += 3

                output_pred_list.append(robot_12_pose)

            # write_dict = {'a1': [], 'a2': [], 'a3': [], 'a4': [], 'a5': [], 'a6': [], 'g1': [], 'g2': [], 'g3': [], 'g4': [], 'g5': [], 'g6': []}
            write_dict = {}
            for k in range(12):
                if k < 6:
                    write_dict[f"a{k+1}"] = []
                else:
                    write_dict[f"g{k-6+1}"] = [] 

            for j in range(len(output_pred_list)):
                write_dict['a1'].append(output_pred_list[j]['a1'])
                write_dict['a2'].append(output_pred_list[j]['a2'])
                write_dict['a3'].append(output_pred_list[j]['a3'])
                write_dict['a4'].append(output_pred_list[j]['a4'])
                write_dict['a5'].append(output_pred_list[j]['a5'])
                write_dict['a6'].append(output_pred_list[j]['a6'])
                write_dict['g1'].append(output_pred_list[j]['g1'])
                write_dict['g2'].append(output_pred_list[j]['g2'])
                write_dict['g3'].append(output_pred_list[j]['g3'])
                write_dict['g4'].append(output_pred_list[j]['g4'])
                write_dict['g5'].append(output_pred_list[j]['g5'])
                write_dict['g6'].append(output_pred_list[j]['g6'])


            for k in range(12):
                if k < 6:
                    robot_name = f'a{k+1}'
                    file_name = f"./pred_data/datarehab_{i}_{robot_name}.csv"
                else:
                    robot_name = f'g{k-6+1}'
                    file_name = f"./pred_data/datarehab_{i}_{robot_name}.csv"
                with open(file_name, 'w') as f:
                    contant = title
                    for j in range(len(write_dict[robot_name])):
                        pose = write_dict[robot_name][j]
                        template = f"{j},{pose[0]},{pose[1]},{pose[2]},{pose[3]},{pose[4]},{pose[5]},{pose[6]}\n"
                        contant += template
                    f.write(contant)
                    #  print(file_name + " done")
            print(f'No.{i} data has beed done!!!')


if __name__ == "__main__":
    main()