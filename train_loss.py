import os
import sys
import torch
import torch.nn as nn
from torch import autograd
from ReLoss.example.train_reloss.spearman import spearman, spearman_diff
from ReLoss.example.train_reloss.utils import accuracy, AverageMeter
from ReLoss.reloss.robot_pose import ReLoss
from pck_metrics_torch import keypoint_pck_accuracy_torch


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


def train_epoch(train_loader, loss_module, optimizer):
    loss_module.train()
    point_dim = 3
    for idx, (logits_batch, targets_batch) in enumerate(train_loader):
        logits_batch = logits_batch.cuda()
        targets_batch = targets_batch.cuda()
        losses = []
        metrics = []
        penalty = []
        for logits, targets in zip(logits_batch, targets_batch):
            # calculate loss and metric for each batch
            # augmentation - randomly modify a portion of targets
            # targets = logits.max(dim=1)[1]
            # correct_num = int(torch.rand(1).item() * targets.shape[0])
            # targets[correct_num:].random_(logits.shape[1])

            # loss = loss_module(logits, targets)

            # preds = tmp_pred[k].view(1, 12, point_dim)
            # gts = tmp_gt[k].view(1, 12, point_dim)
            preds = logits
            gts = targets
            mask = torch.ones((len(preds), 12), dtype=bool).to('cuda')
            normalize = torch.ones((len(preds), point_dim), dtype=torch.float32).to('cuda')

            xyz_acc, xyz_avg_acc, xyz_cnt = keypoint_pck_accuracy_torch(
                pred=preds, 
                gt=gts, 
                thr=0.1, 
                mask=mask, 
                normalize=normalize
            )

            loss = loss_module(preds, gts, mask)

            # penalty = calc_gradient_penalty(loss_module, preds, gts, mask)
            # top1, top5 = accuracy(logits, targets, topk=(1, 5))

            losses.append(loss)
            metrics.append(xyz_avg_acc)

            penalty_ = calc_gradient_penalty(loss_module, logits, targets, mask)
            penalty.append(penalty_)

        penalty = sum(penalty) / logits_batch.shape[0]
        losses = torch.stack(losses)
        metrics = torch.tensor(metrics, device=losses.device)

        diff_spea = spearman_diff(-losses.unsqueeze(0), metrics.unsqueeze(0))
        spea = spearman(-losses.unsqueeze(0).detach(),
                        metrics.unsqueeze(0).detach())

        obj = -diff_spea + 10 * penalty
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

        if idx % 10 == 0:
            print(
                f'Train: [{idx}/{len(train_loader)}] diff_spea {diff_spea.item():.4f} spea {spea.item():.4f}'
            )
            print(f'loss_value   {losses[:5].detach().cpu()}')
            print(f'metric_value {metrics[:5].detach().cpu()}')


def val_epoch(val_loader, loss_module):
    loss_module.eval()
    spea_meter = AverageMeter()
    ce_spea_meter = AverageMeter()
    ce = nn.CrossEntropyLoss()
    criterion = nn.MSELoss().to('cuda')
    with torch.no_grad():
        for idx, (logits_batch, targets_batch) in enumerate(val_loader):
            logits_batch = logits_batch.cuda()
            targets_batch = targets_batch.cuda()
            losses = []
            metrics = []
            ce_losses = []
            for logits, targets in zip(logits_batch, targets_batch):

                # ce_loss = ce(logits, targets)
                mse_loss = criterion(logits, targets)
                # top1, top5 = accuracy(logits, targets, topk=(1, 5))
                mask = torch.ones((len(logits), 12), dtype=bool).to('cuda')
                loss = loss_module(logits, targets, mask)
                normalize = torch.ones((len(logits), 3), dtype=torch.float32).to('cuda')
                xyz_acc, xyz_avg_acc, xyz_cnt = keypoint_pck_accuracy_torch(
                        pred=logits, 
                        gt=targets, 
                        thr=0.1, 
                        mask=mask, 
                        normalize=normalize
                    )
                losses.append(loss)
                metrics.append(xyz_avg_acc)
                ce_losses.append(mse_loss)

            losses = torch.stack(losses)
            ce_losses = torch.stack(ce_losses)
            metrics = torch.tensor(metrics, device=losses.device)

            diff_spea = spearman_diff(-losses.unsqueeze(0),
                                  metrics.unsqueeze(0)).item()
            spea = spearman(-losses.unsqueeze(0), metrics.unsqueeze(0)).item()
            ce_spea = spearman(-ce_losses.unsqueeze(0),
                               metrics.unsqueeze(0)).item()

            spea_meter.update(spea, losses.shape[0])
            ce_spea_meter.update(ce_spea, losses.shape[0])

            if idx % 10 == 0:
                print(
                    f'Val: [{idx}/{len(val_loader)}] diff_spea {diff_spea:.4f} spea {spea:.4f} ce_spea {ce_spea:.4f}'
                )

    print(f'Val: spea {spea_meter.avg:.4f} ce_spea {ce_spea_meter.avg:.4f}')
    return spea_meter.avg


def main(logits_batch_size=128):
    """
    Args:
        @logits_batch_size: the metric ACC is calculated on a 
            batch of samples, logits_batch_size defines the size
            of each batch.
    """
    """
    TODO: you should load your stored logits and targets here.
    train_logits = None  # shape: [N, C]
    train_targets = None  # shape: [N,]
    val_logits = None  # shape: [N, C]
    val_targets = None  # shape: [N,]
    """
    import numpy as np
    train_logits = torch.Tensor(np.load('./train_pred.npy'))
    train_targets = torch.Tensor(np.load('./train_gt.npy'))
    val_logits = torch.Tensor(np.load('./val_pred.npy'))
    val_targets = torch.Tensor(np.load('./val_gt.npy'))

    def batch_data(logits, targets):
        """
        Reshape logits [N, C] to [L, B, C], 
            where B is logits_batch_size and L x B = N
        """
        # drop last
        length = logits.shape[0] // logits_batch_size * logits_batch_size
        logits = logits[:length]
        targets = targets[:length]
        # reshape
        logits = logits.view(-1, logits_batch_size, 12, 3)
        targets = targets.view(-1, logits_batch_size, 12, 3)
        return logits, targets

    train_logits, train_targets = batch_data(train_logits, train_targets)
    print(train_logits.shape)
    print(train_targets.shape)
    val_logits, val_targets = batch_data(val_logits, val_targets)

    dataset_train = torch.utils.data.TensorDataset(train_logits, train_targets)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=256,
                                                   shuffle=True,
                                                   drop_last=True)
    dataset_val = torch.utils.data.TensorDataset(val_logits, val_targets)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=256,
                                                 shuffle=False,
                                                 drop_last=True)

    # initialize reloss module
    loss_module = ReLoss()
    loss_module.cuda()
    optimizer = torch.optim.Adam(loss_module.parameters(),
                                 0.01,
                                 weight_decay=1e-3)

    # train reloss
    best_spearman = -1
    total_epochs = 10
    for epoch in range(total_epochs):
        print(f'epoch: {epoch}')
        train_epoch(dataloader_train, loss_module, optimizer)
        spearman = val_epoch(dataloader_val, loss_module)

        if spearman > best_spearman:
            # save the best checkpoint
            torch.save(loss_module.state_dict(), 'loss_module_best.ckpt')
            best_spearman = spearman


if __name__ == '__main__':
    main()
