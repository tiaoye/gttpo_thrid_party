import torch
import torch.nn as nn
from tqdm import tqdm
from new_utils import relative_to_abs
def evaluate(model, val_loader, params, e, device):
    """
    :param model: torch model
    :param val_loader: torch dataloader
    :param params: dict with hyperparameters
    :param e: epoch number
    :param device: torch device
    :return: val_ADE, val_FDE for one epoch
    """

    model.eval()
    val_ADE = []
    val_FDE = []
    with torch.no_grad():
        for batch, data in enumerate(tqdm(val_loader,desc="Val Epoch {}".format(e))):
            # put data to device
            for key in data.keys():
                data[key] = data[key].to(device)
            obs_traj = data['obs_traj']
            gt_traj = data['pred_traj']
            ped_mask = data['ped_mask']
            adj = data['adj']
            # Forward pass
            sample_num = params['pred_num']
            decoder_hidden, latent_distrib = model.encode(obs_traj, gt_traj, ped_mask, adj, sample_num=sample_num)
            predict_traj_steps = []
            
            # Predict future trajectory
            for i in range(params['pred_len']):
                last_step = (obs_traj[:, :, -1, :] - obs_traj[:, :, -2, :]).repeat(sample_num,1,1) if i == 0 else predict_traj_steps[-1]  # (sample_num*batch, num_ped, 2)
                decoder_hidden, one_step = model.decode(decoder_hidden, last_step)
                predict_traj_steps.append(one_step) # (sample_num*batch, num_ped, 2)

            predict_traj = torch.stack(predict_traj_steps, dim=2)  # (sample_num*batch, num_ped, pred_len, 2)
            predict_traj = relative_to_abs(predict_traj, obs_traj[:, :, -1, :].repeat(sample_num,1,1))  # (sample_num*batch, num_ped, pred_len, 2)

            ade = (((gt_traj.repeat(sample_num,1,1,1) - predict_traj) ** 2).sum(dim=3) ** 0.5) # (sample_num*batch, num_ped, pred_len)
            batch_size = ade.shape[0] // sample_num
            ade = ade.reshape(sample_num, batch_size, ade.shape[1], ade.shape[2]) * ped_mask[:,:,None] # (sample_num, batch, num_ped, pred_len)
            fde, _ = torch.min(ade[:,:,:,-1], dim=0) # (batch, num_ped)
            fde = fde.sum() / ped_mask.sum() # 1
            ade, _ = torch.min(ade.mean(dim=-1), dim=0) # (batch, num_ped)
            ade = ade.sum() / ped_mask.sum() # 1
            val_ADE.append(ade)
            val_FDE.append(fde)

        val_ADE = torch.stack(val_ADE).mean()
        val_FDE = torch.stack(val_FDE).mean()

    return val_ADE.item(), val_FDE.item()
