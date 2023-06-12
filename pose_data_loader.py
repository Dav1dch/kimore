import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from models.RawTransformerAndLSTM import get_net
from torch import nn

trian_lines = 69
test_lines = 30

def read_lines(data_root):
    txt_file = data_root.split('/')[-1]
    txt_file_merge = []
    if 'test' in txt_file:
        lines = test_lines
    elif 'train' in txt_file:
        lines = trian_lines
    else:
        lines = -1
    with open(txt_file) as f:
        for i in range(0, lines):
            txt_single_line = f.readline().strip("\n")
            txt_file_merge.append(txt_single_line.split(" "))
           # break
    print('')
    return txt_file_merge

def concat_data(file_lists, i):
    pose_first_file = True
    robot_first_file = True

    human_pose_numpy = np.array([0])
    human_xyz_numpy = np.array([0])
    human_quat_numpy = np.array([0])

    robot_pose_numpy = np.array([0])
    robot_xyz_numpy = np.array([0])
    robot_quat_numpy = np.array([0])
    robot_force_numpy = np.array([0])

    file_list = file_lists[i]
    human_pose_list = file_list[0:25]
    robot_joint_list = file_list[25:]

    for file in human_pose_list:
        temp_csv = pd.read_csv(file, header=None)
        temp_csv = np.array(temp_csv)
        temp_human_xyz = temp_csv[:, 0:3]
        temp_human_quat = temp_csv[:, 4:8]

        if pose_first_file == True:
            pose_first_file = False
            human_pose_numpy = temp_csv
            human_xyz_numpy = temp_human_xyz
            human_quat_numpy = temp_human_quat

        else:
            human_pose_numpy = np.concatenate([human_pose_numpy, temp_csv], axis = 1)
            human_xyz_numpy = np.concatenate([human_xyz_numpy, temp_human_xyz], axis=1)
            human_quat_numpy = np.concatenate([human_quat_numpy, temp_human_quat], axis = 1)
        #break

    for robot_file in robot_joint_list:
        temp_robot_csv = pd.read_csv(robot_file)
        temp_robot_csv = np.array(temp_robot_csv)
        temp_robot_csv = np.delete(temp_robot_csv, 0, axis=1)
        temp_robot_xyz = temp_robot_csv[:, 2:5]
        temp_robot_quat = temp_robot_csv[:, 5:8]
        temp_robot_force = temp_robot_csv[:, 1].reshape(-1, 1)
        #temp_robot_csv_1 = temp_robot_csv[0]
        #temp_robot_csv = np.insert(temp_robot_csv, 0, temp_robot_csv_1, axis = 0)

        if robot_first_file == True:
            robot_first_file = False
            robot_pose_numpy = temp_robot_csv
            robot_xyz_numpy = temp_robot_xyz
            robot_quat_numpy = temp_robot_quat
            robot_force_numpy = temp_robot_force

        else:
            robot_pose_numpy = np.concatenate([robot_pose_numpy, temp_robot_csv], axis = 1)
            robot_xyz_numpy = np.concatenate([robot_xyz_numpy, temp_robot_xyz], axis = 1)
            robot_quat_numpy = np.concatenate([robot_quat_numpy, temp_robot_quat], axis = 1)
            robot_force_numpy = np.concatenate([robot_force_numpy, temp_robot_force], axis = 1)
        #break

    return human_pose_numpy, robot_force_numpy, robot_pose_numpy, human_xyz_numpy, human_quat_numpy, robot_xyz_numpy, robot_quat_numpy


class PoseDataset(Dataset):
    def __init__(self, data_root, mode='train'):
        if mode == 'train':
            txt_file = os.path.join(data_root, "train_file.txt")
        elif mode == 'test':
            txt_file = os.path.join(data_root, "test_file.txt")
        else:
            txt_file = '' 
        
        self.file_list = read_lines(txt_file)
        self.human_joints_num = 25
        #self.file_index = i

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        human_pose_numpy, robot_force_numpy, robot_pose_numpy, human_xyz_numpy, human_quat_numpy, robot_xyz_numpy, robot_quat_numpy = concat_data(self.file_list, item)
        human_pose_numpy = torch.from_numpy(human_pose_numpy.astype(np.float32))
        robot_pose_numpy = torch.from_numpy(robot_pose_numpy.astype(np.float32))
        human_xyz_numpy = torch.from_numpy(human_xyz_numpy.astype(np.float32))
        human_quat_numpy = torch.from_numpy(human_quat_numpy.astype(np.float32))
        robot_xyz_numpy = torch.from_numpy(robot_xyz_numpy.astype(np.float32))
        robot_quat_numpy = torch.from_numpy(robot_quat_numpy.astype(np.float32))
        robot_force_numpy = torch.from_numpy(robot_force_numpy.astype(np.float32))

        return human_pose_numpy, robot_force_numpy, robot_pose_numpy, human_xyz_numpy, human_quat_numpy, robot_xyz_numpy, robot_quat_numpy


if __name__ == "__main__":
    data_root = "../kimoreData"
    dataset = PoseDataset(data_root)
    human_pose_numpy, robot_force_numpy, robot_pose_numpy, human_xyz_numpy, human_quat_numpy, robot_xyz_numpy, robot_quat_numpy = dataset[5]

    loss = nn.MSELoss(reduction='none')

    net = get_net()
    for i in range(human_xyz_numpy.shape[0]):
        human_xyz_i = human_xyz_numpy[i]
        robot_xyz_i = robot_xyz_numpy[i]
        preds = net(human_xyz_i)
        l = loss(preds, robot_xyz_i)