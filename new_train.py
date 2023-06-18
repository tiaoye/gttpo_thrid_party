import torch
import torch.nn as nn
from tqdm import tqdm
from new_utils import relative_to_abs
import random

def train(model, train_loader, params, optimizer, e, device):
    """
    Run training for one epoch

    :param model: torch model, GTPPO model
    :param train_loader: torch dataloader
    :param e: epoch number
    :param params: dict of hyperparameters
    :param device: torch device
    :param optimizer: torch optimizer
    :return: train_ADE, train_FDE, train_loss for one epoch
    """
    train_loss = 0
    train_ADE = []
    train_FDE = []
    model.train()
    # outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
    for batch, data in enumerate(tqdm(train_loader,desc="Epoch {}".format(e))):
        # put data to device
        for key in data.keys():
            data[key] = data[key].to(device)
        obs_traj = data['obs_traj']
        gt_traj = data['pred_traj']
        ped_mask = data['ped_mask']
        adj = data['adj']
        # Forward pass
        decoder_hidden, latent_distrib = model.encode(obs_traj, gt_traj, ped_mask, adj)
        
        # for intention classification
        if params["use_intention_cluster"]:
            intention = model.intention_classifier(decoder_hidden)
            gt_intention = data['intention_cluster_label']
            intention_loss = model.intention_classifier.cal_loss(intention, gt_intention, ped_mask)

        predict_traj_steps = []

        # Predict future trajectory
        if params["teaching_mode"] == 'recursive':
            for i in range(params['pred_len']):
                last_step = (obs_traj[:, :, -1, :] - obs_traj[:, :, -2, :]) if i == 0 else predict_traj_steps[-1]  # (batch, num_ped, 2)
                decoder_hidden, one_step = model.decode(decoder_hidden, last_step)
                predict_traj_steps.append(one_step)

        elif params["teaching_mode"] == 'teacher_forcing':
            relative_gt_traj = torch.zeros_like(gt_traj) # (batch, num_ped, pred_len, 2)
            relative_gt_traj[:,:,1:,:] = gt_traj[:,:,1:,:] - gt_traj[:,:,:-1,:]
            relative_gt_traj[:,:,0,:] = gt_traj[:,:,0,:] - obs_traj[:,:,-1,:]
            # use teacher forcing
            if random.random() < params["teacher_forcing_ratio"]:
                for i in range(params['pred_len']):
                    last_step = (obs_traj[:, :, -1, :] - obs_traj[:, :, -2, :]) if i == 0 else relative_gt_traj[:,:,i-1,:]  # (batch, num_ped, 2)
                    decoder_hidden, one_step = model.decode(decoder_hidden, last_step)
                    predict_traj_steps.append(one_step)
            # predict recursively
            else:
                for i in range(params['pred_len']):
                    last_step = (obs_traj[:, :, -1, :] - obs_traj[:, :, -2, :]) if i == 0 else predict_traj_steps[-1]  # (batch, num_ped, 2)
                    decoder_hidden, one_step = model.decode(decoder_hidden, last_step)
                    predict_traj_steps.append(one_step)

        elif params["teaching_mode"] == 'mixed':
            relative_gt_traj = torch.zeros_like(gt_traj) # (batch, num_ped, pred_len, 2)
            relative_gt_traj[:,:,1:,:] = gt_traj[:,:,1:,:] - gt_traj[:,:,:-1,:]
            relative_gt_traj[:,:,0,:] = gt_traj[:,:,0,:] - obs_traj[:,:,-1,:]
            last_step = obs_traj[:, :, -1, :] - obs_traj[:, :, -2, :]
            for i in range(params['pred_len']):
                decoder_hidden, one_step = model.decode(decoder_hidden, last_step)
                predict_traj_steps.append(one_step)
                # use teacher forcing
                if random.random() < params["teacher_forcing_ratio"]:
                    last_step = relative_gt_traj[:,:,i,:]  # (batch, num_ped, 2)
                # predict recursively
                else:    
                    last_step = one_step




        predict_traj = torch.stack(predict_traj_steps, dim=2)  # (batch, num_ped, pred_len, 2)
        predict_traj = relative_to_abs(predict_traj, obs_traj[:, :, -1, :])  # (batch, num_ped, pred_len, 2)

        losses = model.cal_loss(predict_traj,gt_traj,ped_mask,latent_distrib=latent_distrib)  # L2 loss + KL loss
        loss = losses['trajectory_loss'] + losses['KL_loss'] * params["KL_loss_ratio"]
        # add intention loss
        if params["use_intention_cluster"]:
            loss += intention_loss * params["intention_loss_ratio"]
            losses['intention_loss'] = intention_loss

        losses['total_loss'] = loss
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss
            ade = (((gt_traj - predict_traj) ** 2).sum(dim=3) ** 0.5) # (batch, num_ped, pred_len)
            fde = (ade[:,:,-1] * ped_mask).sum() / ped_mask.sum() # 1
            ade = (ade.mean(dim=-1) * ped_mask).sum() / ped_mask.sum() # 1
            train_ADE.append(ade)
            train_FDE.append(fde)

    train_ADE = torch.stack(train_ADE).mean()
    train_FDE = torch.stack(train_FDE).mean()

    return train_ADE.item(), train_FDE.item(), losses
