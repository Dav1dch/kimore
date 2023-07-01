import torch
import torch.nn.functional as F
from torch import autograd

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
from pck_metrics_torch import keypoint_pck_accuracy_torch
from ReLoss.reloss.robot_pose import ReLoss
from ReLoss.example.train_reloss.spearman import spearman, spearman_diff


def calc_gradient_penalty(loss_module, logits, targets, mask):
    logits = autograd.Variable(logits, requires_grad=True)
    loss = loss_module(logits, targets, mask)
    gradients = autograd.grad(outputs=loss,
                              inputs=logits,
                              grad_outputs=torch.ones(loss.size(),
                                                      device=loss.device),
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return penalty

def point_change(raw_xyz, dim=3):
    raw_xyz = raw_xyz.cpu()
    raw_xyz = raw_xyz.detach().numpy()
    d = raw_xyz.shape[0] // 3
    xyz = raw_xyz.reshape(d, dim)
    return xyz


def run(mode='train'):
    path_root = "../Kimore"
    device = torch.device('cuda')
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
    hidden_embeddings = 512
    criterion_MSE = torch.nn.MSELoss().to(device)
    # model = LSTMTaggerSep(input_embeddings, hidden_embeddings, lstm_num_layers = num_layer)
    model = KimoreFusionModel(input_embeddings, hidden_embeddings, num_layers = num_layer)
    model = model.to(device)

    loss_module = ReLoss()
    print(loss_module.logits_fcs)
    loss_module = loss_module.to(device)
    reoptimizer = torch.optim.Adam(loss_module.parameters(),
                                 0.005,
                                 weight_decay=1e-4)

    learning_rate = 0.001
    momentum = 0.95
    weight_decay = 5e-4
    scheduler_gamma = 0.95
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, scheduler_gamma)

    shrinkage_a = 5
    shrinkage_c = 0.2
    # criterion_1 = Shrinkage_loss(shrinkage_a, shrinkage_c).to(device)
    criterion = nn.MSELoss().to(device)
    best_loss = 1000

    point_dim = 3
    torch.autograd.set_detect_anomaly = True

    if mode == 'train':
        model.train()
        loss_module.train()
    elif mode == 'test':
        state_dict = torch.load('./trained_param/best_performance_param.pth')
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print("Mode error, neither train nor test")
        return False
    
    if mode == 'test':
        max_epoch = 1
    
    avg_arr = []

    for epoch in tqdm(range(max_epoch)):
        if mode == 'train':
            data_iter = iter(train_dataloader)
        elif mode == 'test':
            data_iter = iter(test_dataloader)
        else:
            return False
        
        sum_loss = 0.0
        avg_acc = 0.0

        # print("The epoch is: ")
        # print(epoch)
        with torch.enable_grad():
            for i, data in enumerate(data_iter):

                
                human_xyz_numpy, human_quat_numpy, robot_xyz_numpy, robot_vel_numpy,robot_force_numpy = data

                ground_truth_xyz_list = []
                ground_truth_vel_list = []

                output_pred_xyz_list = []
                output_pred_vel_list = []

                pred_xyz_points_list = []
                gt_pred_xyz_points_list = []
                pred_vel_points_list = []
                gt_pred_vel_points_list = []

                losssum = 0
                n = 0
                losses_xyz = []
                losses_vel = []

                loss = torch.tensor(0.0, device = device, requires_grad=True)
                loss_xyz = torch.tensor(0.0, device = device, requires_grad=True)
                loss_vel = torch.tensor(0.0, device = device, requires_grad=True)

                print(f"human pose data number: {robot_xyz_numpy.shape[1]}")
                optimizer.zero_grad()
                model.init_state()
                tmp_pred = []
                tmp_gt = []

                for j in range(robot_xyz_numpy.shape[1]):
                    human_xyz_numpy_ = human_xyz_numpy[0]
                    human_xyz_single_ = human_xyz_numpy_[j]
                    human_xyz_single = human_xyz_single_[None, :]

                    human_xyz_ = human_xyz_numpy[0, j].unsqueeze(0)
                    human_quat_ = human_quat_numpy[0, j].unsqueeze(0)
                    human_input = torch.cat((human_xyz_, human_quat_), dim=1).to(device)
                    human_input = human_input.unsqueeze(0)

                    # robot_xyz_numpy_ = robot_xyz_numpy[0]
                    # robot_velocity_numpy_ = robot_vel_numpy[0]
                    # robot_force_numpy_ = robot_force_numpy[0]

                    ground_truth_xyz_ = robot_xyz_numpy[0, j]
                    ground_truth_velocity_ = robot_vel_numpy[0, j]
                    ground_truth_force_ = robot_force_numpy[0, j]

                    ground_truth_xyz = F.normalize(ground_truth_xyz_.float().to(device), dim=-1)
                    ground_truth_velocity = ground_truth_velocity_.float().to(device)
                    ground_truth_force = ground_truth_force_.float().to(device)




                    preds_xyz = model(human_input)
    # , preds_vel


                    preds_xyz = preds_xyz.squeeze()
                    # preds_vel = preds_vel.squeeze()
                    # print('preds ', preds_xyz)
                    # print('gt ',ground_truth_xyz)

                    output_pred_xyz_list.append(preds_xyz)
                    tmp_pred.append(preds_xyz.detach())
                    # output_pred_vel_list.append(preds_vel)

                    ground_truth_xyz_list.append(ground_truth_xyz)
                    tmp_gt.append(ground_truth_xyz.detach())
                    ground_truth_vel_list.append(ground_truth_velocity)
                    losses_xyz.append(criterion(output_pred_xyz_list[j].view(-1, 3), ground_truth_xyz_list[j].view(-1, 3)))
                    # losses_vel.append(criterion(output_pred_vel_list[j], ground_truth_vel_list[j]))
                    pred_xyz_points_list.append(output_pred_xyz_list[j].cpu().detach().numpy())
                    # pred_vel_points_list.append(output_pred_vel_list[j].cpu().detach().numpy())
                    gt_pred_xyz_points_list.append(ground_truth_xyz_list[j].cpu().detach().numpy())
                    gt_pred_vel_points_list.append(ground_truth_vel_list[j].cpu().detach().numpy())
                    if mode == 'train':
                        if j != 0 and (j % 128 == 0 or j == human_xyz_numpy.shape[1] - 1):
                            n += 1
                            a = torch.vstack(losses_xyz)
                            a_ = torch.where(torch.isnan(a), torch.full_like(a, 0), a)

                            # b = torch.vstack(losses_vel)
                            # b = torch.where(torch.isnan(b), torch.full_like(b, 0), b)

                            loss = torch.mean(a_)
                            # print(loss)
                            # + torch.mean(b)
                            losssum += loss.item()

                            # loss1.backward()
                            loss.backward()
                            optimizer.step()
                            # scheduler.step()
                            optimizer.zero_grad()



                            # loss1 = loss.detach_().requires_grad_(True)

                            relosses = []
                            remetrics = []
                            repenalty = []
                            # for k in range(len(tmp_pred)):
                            #     preds = tmp_pred[k].view(1, 12, point_dim)
                            #     gts = tmp_gt[k].view(1, 12, point_dim)
                            #     mask = torch.ones((1, 12), dtype=bool).to('cuda')
                            #     normalize = torch.ones((1, point_dim), dtype=torch.float32).to('cuda')

                            #     xyz_acc, xyz_avg_acc, xyz_cnt = keypoint_pck_accuracy_torch(
                            #         pred=preds, 
                            #         gt=gts, 
                            #         thr=0.05, 
                            #         mask=mask, 
                            #         normalize=normalize
                            #     )
                            #     loss_re = loss_module(preds, gts, mask)
                            #     penalty = calc_gradient_penalty(loss_module, preds, gts, mask)
                            #     relosses.append(loss_re)
                            #     remetrics.append(xyz_avg_acc)
                            #     repenalty.append(penalty)
                            # penalty = sum(repenalty) / a_.shape[0]
                            # relosses = torch.stack(relosses)
                            # metrics = torch.tensor(remetrics, device=relosses.device)

                            # diff_spea = spearman_diff(-relosses.unsqueeze(0), metrics.unsqueeze(0))


                            # obj = -diff_spea + 10 * penalty
                            # reoptimizer.zero_grad()
                            # obj.backward()
                            # reoptimizer.step()

                            losses_xyz = []
                            losses_vel = []
                            tmp_pred = []
                            tmp_gt = []

                        
                    

                N = 0
                N = len(output_pred_xyz_list)
                

                sum_loss += losssum / max(n,1)

                tmp = pred_xyz_points_list[0].shape[0]
                K = int(tmp // point_dim)
                np.save(f'pred/pred_{epoch}_{str(i).zfill(2)}.npy',pred_xyz_points_list)
                np.save(f'gt/gt_{epoch}_{str(i).zfill(2)}.npy',gt_pred_xyz_points_list)
                pred_xyz_topck = np.array(pred_xyz_points_list).reshape(N, K, point_dim)
                # pred_vel_topck = np.array(pred_vel_points_list).reshape(N, K, point_dim)
                gt_pred_xyz_topck = np.array(gt_pred_xyz_points_list).reshape(N, K, point_dim)
                # gt_pred_vel_topck = np.array(gt_pred_vel_points_list).reshape(N, K, point_dim)
                mask = np.ones((N, K), dtype=bool)
                normalize = np.ones((N, point_dim), dtype=float)

                xyz_acc, xyz_avg_acc, xyz_cnt = keypoint_pck_accuracy(
                    pred=pred_xyz_topck, 
                    gt=gt_pred_xyz_topck, 
                    thr=0.1, 
                    mask=mask, 
                    normalize=normalize
                )
                xyz_pck_res = f"i: {i}\n xyz_acc: {xyz_acc}\nxyz_avg_acc: {xyz_avg_acc}\nxyz_cnt: {xyz_cnt}"
                print(xyz_pck_res)
                # print(f"\nxyz_avg_acc: {xyz_avg_acc}")
                avg_acc += xyz_avg_acc
                if len(avg_arr) == 0:
                    avg_arr = xyz_acc
                else:
                    avg_arr += xyz_acc


                # vel_acc, vel_avg_acc, vel_cnt = keypoint_pck_accuracy(
                #     pred=pred_vel_topck, 
                #     gt=gt_pred_vel_topck, 
                #     thr=0.1, 
                #     mask=mask, 
                #     normalize=normalize
                # )



                # vel_pck_res = f"vel_acc: {vel_acc}\nvel_avg_acc: {vel_avg_acc}\nvel_cnt: {vel_cnt}"
                # print(vel_pck_res)
                # print("------***------")

            
        sum_loss = sum_loss / 69.0
        if mode == 'train':
            if sum_loss < best_loss:
                best_loss = sum_loss
                torch.save(model.state_dict(), "./trained_param/best_performance_param_.pth")
            torch.save(loss_module.state_dict(), './trained_param/reloss.ckpt')

            print(f"Training: sum loss {sum_loss}, xyz_loss: {loss_xyz}, vel_loss: {loss_vel}, avg_acc: {avg_acc / 69}")
        else:
            print(f"Training: sum loss {sum_loss}, xyz_loss: {loss_xyz}, vel_loss: {loss_vel}, avg_acc: {avg_acc / 30}")
            print(f"Testing: sum loss {sum_loss}, xyz_loss: {loss_xyz}, vel_loss: {loss_vel}")
            print(avg_arr)
            print(avg_arr / 30)
        print("\n")
    

if __name__ == "__main__":
    run()
    # run('test')
