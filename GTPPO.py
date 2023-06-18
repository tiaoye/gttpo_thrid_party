import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from new_test import evaluate
from new_train import train

from dataset import TrajectoryDataset, gtppo_collate
from my_logger import Logger
import time
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def make_mlp(dim_list, activation='relu', batch_norm=False, dropout=0):
    layers = []
    for i,(dim_in, dim_out) in enumerate(zip(dim_list[:-1], dim_list[1:])):
        layers.append(nn.Linear(dim_in, dim_out))
        if i == len(dim_list) - 2:
            # the last layer does not have activation and batch norm
            break
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class TAEncoderDecoder(nn.Module):
    def __init__(self,params):
        super(TAEncoderDecoder, self).__init__()
        # encoder
        self.encoder_gru = nn.GRU(input_size=params['gru_input_size'], hidden_size=params['gru_hidden_size'], num_layers=params['gru_num_layers'], batch_first=True)
        self.input_embedding = nn.Linear(2, params['gru_input_size'])
        self.attention_linear = nn.Linear(params['gru_hidden_size'], 1)

        # decoder
        self.decoder_gru = nn.GRU(input_size=params['gru_input_size'], hidden_size=params['gru_hidden_size'], num_layers=params['gru_num_layers'], batch_first=True)
        self.decoder_init_linear = make_mlp([params['gru_hidden_size']+params['gru_hidden_size']+params['latent_dim']*4, params['gru_hidden_size']*2,params['gru_hidden_size']])
        self.output_embedding = nn.Linear(params['gru_hidden_size'], 2)
        # dropout
        self.dropout = nn.Dropout(p=params['dropout'])

    def encode(self, trajectory):
        '''
        Args:
            trajectory: (batch_size, ped_num ,obs_len, 2)
        Returns:
            hidden_states: (batch_size, ped_num, obs_len-1, gru_hidden_size)
            output: (batch_size, ped_num, gru_hidden_size)
        '''
        batch_size, ped_num, obs_len, _ = trajectory.shape
        relative_pos = trajectory[:, :, 1:, :] - trajectory[:, :, :-1, :] # (batch_size, ped_num, obs_len-1, 2)
        relative_pos = self.input_embedding(relative_pos.reshape(batch_size*ped_num, obs_len-1, 2)) # (batch_size*ped_num, obs_len-1, gru_input_size)

        hidden_states, _ = self.encoder_gru(relative_pos) # (batch_size*ped_num, obs_len-1, gru_hidden_size)
        hidden_states = hidden_states.reshape(batch_size, ped_num, obs_len-1, -1) # (batch_size, ped_num, obs_len-1, gru_hidden_size)

        mu = torch.tanh(self.attention_linear(hidden_states)).squeeze(-1) # (batch_size, ped_num, obs_len-1)
        att = F.softmax(mu, dim=-1) # (batch_size, ped_num, obs_len-1)
        att = self.dropout(att)
        output = torch.sum(att.unsqueeze(-1) * hidden_states, dim=2) # (batch_size, ped_num, gru_hidden_size)
        return hidden_states, output
    
    def deocder_hidden_init(self,encoder_output,graph_attention_output,latent,sample_num=1):
        '''
        Args:
            encoder_output: (batch_size, ped_num, gru_hidden_size)
            graph_attention_output: (batch_size, ped_num, gru_hidden_size)
            latent: (sample_num*batch_size, ped_num, latent_dim*4)
            sample_num: int, number of samples, in training, sample_num=1, in testing, sample_num=sample_num.

        Returns:
            decoder_hidden: (1, sample_num*batch_size*ped_num, gru_hidden_size)
        '''
        if sample_num != 1:
            encoder_output = encoder_output.repeat(sample_num,1,1) # (sample_num*batch_size, ped_num, gru_hidden_size)
            graph_attention_output = graph_attention_output.repeat(sample_num,1,1) # (sample_num*batch_size, ped_num, gru_hidden_size)
        decoder_hidden = torch.cat([encoder_output,graph_attention_output,latent],dim=-1) # (sample_num*batch_size, ped_num, gru_hidden_size+gru_hidden_size+latent_dim*4)
        decoder_hidden = self.decoder_init_linear(decoder_hidden) # (sample_num*batch_size, ped_num, gru_hidden_size)
        decoder_hidden = decoder_hidden.unsqueeze(0) # (1, sample_num*batch_size, ped_num, gru_hidden_size)
        decoder_hidden = decoder_hidden.reshape(1, -1, self.decoder_gru.hidden_size) # (1, sample_num*batch_size*ped_num, gru_hidden_size)
        return decoder_hidden

    def  decode(self,decoder_hidden,relative_pos):
        '''
        Args:
            decoder_hidden: (1, sample_num*batch_size*ped_num, gru_hidden_size)
            relative_pos: (sample_num*batch_size, ped_num, 2)
        Returns:
            decoder_hidden: (1, sample_num*batch_size*ped_num, gru_hidden_size)
            output: relative pos (sample_num*batch_size, ped_num, 2)
        '''
        batch_size, ped_num, _ = relative_pos.shape
        relative_pos = self.input_embedding(relative_pos.reshape(batch_size*ped_num, 1, 2)) # (sample_num*batch_size*ped_num, 1, gru_input_size)
        output, decoder_hidden = self.decoder_gru(relative_pos, decoder_hidden) # (sample_num*batch_size*ped_num, 1, gru_hidden_size) (1, sample_num*batch_size*ped_num, gru_hidden_size)
        output = self.output_embedding(output.reshape(batch_size,ped_num, -1)) # (sample_num,batch_size,ped_num, 2)
        return decoder_hidden, output

class GraphAttentionLayer(nn.Module):
    def __init__(self,params):
        super(GraphAttentionLayer, self).__init__()
        self.relu = nn.LeakyReLU(params['leakyrelu_input_slope'])
        self.linear1 = nn.Linear(params['gru_hidden_size'], params['graph_attention_hidden_size'], bias=False)
        self.linear2 = nn.Linear(2*params['graph_attention_hidden_size'], 1, bias=False)
        ## init weights following the GAT paper
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1.414)
        if params['social_attention_mode'] == 'SSA':
            self.ssa = nn.Linear(1,1)
        self.mode = params['social_attention_mode']
        self.dropout = nn.Dropout(p=params['dropout'])

    def forward(self, agents, adj, mask):
        '''
        Args:
            agents: hidden state (batch_size, ped_num, obs_len-1, gru_hidden_size)
            adj: cosine values of the angle between velocity orientation of agent and the vector joining agent  and neighbor (batch_size, ped_num, ped_num, obs_len-1)
            mask: (batch_size, ped_num)
        Returns:
            agents: (batch_size, ped_num, obs_len-1, gru_hidden_size)
        '''
        batch_size, ped_num, seq_len, _ = agents.shape
        Wagents = self.linear1(agents) # (batch_size, ped_num, obs_len-1, graph_attention_hidden_size)
        # the dim 1 is agent, dim 2 is neighbors of the agent
        att1 = Wagents.unsqueeze(2).repeat(1,1,ped_num,1,1) # (batch_size, ped_num, ped_num, obs_len-1, graph_attention_hidden_size)
        att2 = Wagents.unsqueeze(1).repeat(1,ped_num,1,1,1) # (batch_size, ped_num, ped_num, obs_len-1, graph_attention_hidden_size)
        att = torch.cat([att1, att2], dim=-1) # (batch_size, ped_num, ped_num, obs_len-1, 2*graph_attention_hidden_size)
        att = self.relu(self.linear2(att).squeeze(-1)) # (batch_size, ped_num, ped_num, obs_len-1)
        att = torch.masked_fill(att, mask[:,None,:,None], -1e9) # (batch_size, ped_num, ped_num, obs_len-1
        att = F.softmax(att, dim=2) # (batch_size, ped_num, ped_num, obs_len-1)
        att = self.dropout(att)

        if self.mode == 'SSA':
            adj = torch.sigmoid(self.ssa(adj.unsqueeze(-1)).squeeze(-1)) # (batch_size, ped_num, ped_num, obs_len-1)
        elif self.mode == 'HSA':
            adj = (adj > 0.0)*1.0   # (batch_size, ped_num, ped_num, obs_len-1)
        else:
            adj = torch.ones_like(adj) # (batch_size, ped_num, ped_num, obs_len-1)
        weights = att * adj # (batch_size, ped_num, ped_num, obs_len-1)
        agents = torch.sigmoid(torch.sum(weights.unsqueeze(-1) * agents.unsqueeze(1).repeat(1,ped_num,1,1,1), dim=2)) # (batch_size, ped_num, obs_len-1, gru_hidden_size)
        return agents

class GraphAttention(nn.Module):
    def __init__(self,params):
        super(GraphAttention, self).__init__()
        self.layers = nn.ModuleList([GraphAttentionLayer(params) for _ in range(params['graph_attention_num_layers'])])
        self.gru = nn.GRU(input_size=params['gru_hidden_size'], hidden_size=params['gru_hidden_size'], num_layers=params['gru_num_layers'], batch_first=True)
        self.dropout = nn.Dropout(p=params['dropout'])
    def forward(self, agents, adj, mask):
        '''
        Args:
            agents: hidden state (batch_size, ped_num, obs_len-1, gru_hidden_size)
            adj: cosine values of the angle between velocity orientation of agent and the vector joining agent  and neighbor (batch_size, ped_num, ped_num, obs_len-1)
            mask: (batch_size, ped_num)
        Returns:
            output: (batch_size, ped_num, gru_hidden_size)
        '''
        for layer in self.layers:
            agents = layer(agents, adj, mask)
        agents = self.dropout(agents)
        batch_size, ped_num, seq_len, _ = agents.shape
        agents, _ = self.gru(agents.reshape(batch_size*ped_num, seq_len, -1)) # (batch_size*ped_num, obs_len-1, gru_hidden_size)
        return agents.reshape(batch_size, ped_num, seq_len, -1)[:,:,-1,:] # (batch_size, ped_num, gru_hidden_size)
    
class LatentPredictor(nn.Module):
    def __init__(self,params):
        super(LatentPredictor, self).__init__()
        self.latent_dim = params['latent_dim']
        self.embedding_layers = nn.ModuleList([nn.Linear(2, params['gru_input_size']) for _ in range(3)])
        self.grus = nn.ModuleList([nn.GRU(input_size=params['gru_input_size'], hidden_size=params['gru_hidden_size'], num_layers=params['gru_num_layers'], batch_first=True) for _ in range(3)])
        self.latent_encoders = nn.ModuleList([nn.Linear(params['gru_hidden_size'], self.latent_dim*2) for _ in range(3)])
        self.dropout = nn.Dropout(p=params['dropout'])

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, trajectory, ground_truth=None, sample_num=1):
        '''
        Args:
            trajectory: (batch_size, ped_num, obs_len, 2)
            ground_truth: (batch_size, ped_num, pred_len, 2)
            sample_num: number of samples to be generated, only for validation. batch will be duplicated sample_num times
        Return:
            z: latent variable (sample_num*batch_size, ped_num, latent_dim*4)
            mu_gt, logvar_gt, mu_history, logvar_history: (batch_size, ped_num, latent_dim*3)
        '''
        if self.training:
            batch_size, ped_num, obs_len, _ = trajectory.shape
            batch_size, ped_num, pred_len, _ = ground_truth.shape
            trajectory = torch.cat([trajectory, ground_truth], dim=2) # (batch_size, ped_num, obs_len+pred_len, 2)
            velocity = trajectory[:,:,1:,:] - trajectory[:,:,:-1,:] # (batch_size, ped_num, obs_len+pred_len-1, 2)
            acceleration = velocity[:,:,1:,:] - velocity[:,:,:-1,:] # (batch_size, ped_num, obs_len+pred_len-2, 2)
            embeddings = [ 
                self.embedding_layers[i](x) for i,x in enumerate([trajectory, velocity, acceleration]) ] # (batch_size, ped_num, obs_len+pred_len(-0)(-1)(-2), gru_input_size)
            gru_output = [ 
                self.grus[i](embeddings[i].reshape(batch_size*ped_num, obs_len+pred_len-i, -1))[0].reshape(batch_size, ped_num, obs_len+pred_len-i, -1) for i in range(3) ] # (batch_size, ped_num, obs_len+pred_len(-0)(-1)(-2), gru_hidden_size)
            gru_output_gt = [ 
                x[:,:,-1] for x in gru_output ] # (batch_size, ped_num, gru_hidden_size)
            gru_output_history = [ 
                x[:,:,obs_len-1-i] for i,x in enumerate(gru_output) ] # (batch_size, ped_num, gru_hidden_size)
            gru_output_gt = [ 
                self.latent_encoders[i](gru_output_gt[i]) for i in range(3) ] # (batch_size, ped_num, latent_dim*2)
            gru_output_history = [ 
                self.latent_encoders[i](gru_output_history[i]) for i in range(3) ] # (batch_size, ped_num, latent_dim*2)
            mu_gt, logvar_gt = [ 
                x[:,:,:self.latent_dim] for x in gru_output_gt ], [ 
                x[:,:,self.latent_dim:] for x in gru_output_gt ] # (batch_size, ped_num, latent_dim)
            mu_history, logvar_history = [ 
                x[:,:,:self.latent_dim] for x in gru_output_history ], [ 
                x[:,:,self.latent_dim:] for x in gru_output_history ] # (batch_size, ped_num, latent_dim)
            mu_gt, logvar_gt, mu_history, logvar_history = [torch.cat(x, dim=2) for x in [mu_gt, logvar_gt, mu_history, logvar_history]] # (batch_size, ped_num, latent_dim*3)
            z_gt = self.sample(mu_gt, logvar_gt) # (batch_size, ped_num, latent_dim*3)
            random_noise = torch.randn(batch_size,ped_num,self.latent_dim).to(z_gt) # (batch_size, ped_num, latent_dim)
            z = torch.cat([z_gt, random_noise], dim=2) # (batch_size, ped_num, latent_dim*4)
            return z, (mu_gt, logvar_gt, mu_history, logvar_history)
        else:
            batch_size, ped_num, obs_len, _ = trajectory.shape
            velocity = trajectory[:,:,1:,:] - trajectory[:,:,:-1,:] # (batch_size, ped_num, obs_len-1, 2)
            acceleration = velocity[:,:,1:,:] - velocity[:,:,:-1,:] # (batch_size, ped_num, obs_len-2, 2)
            embeddings = [ 
                self.embedding_layers[i](x) for i,x in enumerate([trajectory, velocity, acceleration]) ] # (batch_size, ped_num, obs_len(-0)(-1)(-2), gru_input_size)
            gru_output = [ 
                self.grus[i](embeddings[i].reshape(batch_size*ped_num, obs_len-i, -1))[0].reshape(batch_size, ped_num, obs_len-i, -1) for i in range(3) ] # (batch_size, ped_num, obs_len(-0)(-1)(-2), gru_hidden_size)
            gru_output = [ 
                x[:,:,-1] for x in gru_output ] # (batch_size, ped_num, gru_hidden_size)
            gru_output = [ 
                self.latent_encoders[i](gru_output[i]) for i in range(3) ] # (batch_size, ped_num, latent_dim*2)
            mu, logvar = [ 
                x[:,:,:self.latent_dim] for x in gru_output ], [ 
                x[:,:,self.latent_dim:] for x in gru_output ] # (batch_size, ped_num, latent_dim)
            mu, logvar = [torch.cat(x, dim=2) for x in [mu, logvar]] # (batch_size, ped_num, latent_dim*3)

            # sample z for sample_num times
            mu, logvar = [x.repeat(sample_num,1,1) for x in [mu, logvar]] # (sample_num*batch_size, ped_num, latent_dim*3)
            z = self.sample(mu, logvar) # (sample_num*batch_size, ped_num, latent_dim*3)
            random_noise = torch.randn(sample_num*batch_size, ped_num, self.latent_dim).to(z) # (sample_num*batch_size, sample_num, ped_num, latent_dim)
            z = torch.cat([z, random_noise], dim=-1) # (sample_num*batch_size, ped_num, latent_dim*4)
            return z, None
    
    def cal_loss(self, mu_gt, logvar_gt, mu_history, logvar_history, mask):
        '''
        Args:
            mu_gt: (batch_size, ped_num, latent_dim*3)
            logvar_gt: (batch_size, ped_num, latent_dim*3)
            mu_history: (batch_size, ped_num, latent_dim*3)
            logvar_history: (batch_size, ped_num, latent_dim*3)
            mask: (batch_size, ped_num)
        Returns:
            loss: (1)
        '''
        # calculate KL divergence between history distribution and ground truth distribution
        kl_divergence = 0.5*(logvar_gt-logvar_history - 1 + (torch.exp(logvar_history)+torch.pow(mu_history-mu_gt,2))/torch.exp(logvar_gt)).sum(dim=2) # (batch_size, ped_num)
        kl_divergence = (kl_divergence*mask).sum()/mask.sum()
        return kl_divergence

class intention_classifier(nn.Module):
    def __init__(self,params):
        super(intention_classifier, self).__init__()
        self.params = params
        self.mlp = make_mlp([params['gru_hidden_size'],params['gru_hidden_size']//2,params['intention_cluster_num']])
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, decoder_hidden):
        '''
        Args:
            decoder_hidden: (1, sample_num*batch_size*ped_num, gru_hidden_size)
        Returns:
            intention: (sample_num*batch_size*ped_num, intention_cluster_num)
        '''
        intention = self.mlp(decoder_hidden.squeeze(0)) # (sample_num*batch_size*ped_num, intention_cluster_num)
        return intention
    
    def cal_loss(self, intention, intention_gt, mask):
        '''bce loss with mask
        Args:
            intention: (batch_size*ped_num, intention_cluster_num)
            intention_gt: (batch_size, ped_num)
            mask: (batch_size, ped_num)
        Returns:
            loss: (1)
        '''
        loss = self.cross_entropy(intention, intention_gt.reshape(-1)) # (batch_size*ped_num)
        loss = (loss*mask.reshape(-1)).sum()/mask.sum()
        return loss
    
class GTPPO_model(nn.Module):
    def __init__(self, params):
        super(GTPPO_model, self).__init__()
        self.params = params
        self.encoder_decoder = TAEncoderDecoder(params)
        self.graph_module = GraphAttention(params)
        self.latent_predictor = LatentPredictor(params)
        #TODO: should add dropout in each layer?
        if params['use_intention_cluster']:
            self.intention_classifier = intention_classifier(params)

    def encode(self, trajectory, groud_truth, mask, adj, sample_num=1):
        '''
        Args:
            trajectory: (batch_size, ped_num, obs_len, 2)
            groud_truth: (batch_size, ped_num, pred_len, 2)
            mask: (batch_size, ped_num)
            adj: (batch_size, ped_num, ped_num, obs_len-1)
            sample_num: (int) number of samples for each agent, only used in test
        Returns:
            decoder_hidden: (1, sample_num*batch_size*ped_num, hidden_size)
            latent_distrib:
                mu_gt: (batch_size, ped_num, latent_dim*3)
                logvar_gt: (batch_size, ped_num, latent_dim*3)
                mu_history: (batch_size, ped_num, latent_dim*3)
                logvar_history: (batch_size, ped_num, latent_dim*3)
        '''
        agents_hidden_states, encoder_output = self.encoder_decoder.encode(trajectory) # (batch_size, ped_num, obs_len-1, hidden_size), (batch_size, ped_num, hidden_size)
        ga_output = self.graph_module(agents_hidden_states, adj, mask) # (batch_size, ped_num, hidden_size)
        z, latent_distrib = self.latent_predictor(trajectory, groud_truth, sample_num) # (sample_num*batch_size, ped_num, latent_dim*4)
        decoder_hidden = self.encoder_decoder.deocder_hidden_init(encoder_output, ga_output, z, sample_num) # (1, sample_num*batch_size*ped_num, hidden_size)
        return decoder_hidden, latent_distrib

    def decode(self, decoder_hidden, relative_pos):
        '''
        Args:
            decoder_hidden: (1, sample_num*batch_size*ped_num, gru_hidden_size)
            relative_pos: (sample_num*batch_size, ped_num, 2)
            sample_num: (int) number of samples for each agent, only used in test
        Returns:
            decoder_hidden: (1, sample_num*batch_size*ped_num, gru_hidden_size)
            output: relative pos (sample_num*batch_size, ped_num, 2)
        '''
        decoder_hidden,output = self.encoder_decoder.decode(decoder_hidden, relative_pos)
        return decoder_hidden, output
    
    def cal_loss(self, pred_traj, ground_truth, mask, latent_distrib=None):
        '''
        Args:
            pred_traj: (batch_size, ped_num, pred_len, 2) 
            ground_truth: (batch_size, ped_num, pred_len, 2)
            mask: (batch_size, ped_num)
            latent_distrib: tuple,just for train mode
                mu_gt: (batch_size, ped_num, latent_dim*3)
                logvar_gt: (batch_size, ped_num, latent_dim*3)
                mu_history: (batch_size, ped_num, latent_dim*3)
                logvar_history: (batch_size, ped_num, latent_dim*3)
        Returns:
            losses: dict
                loss['trajectory_loss']: (1)
                loss['latent_loss']: (1)
        '''
        losses = {}
        # calculate l2 loss
        loss = torch.pow(pred_traj-ground_truth,2).sum(dim=3).sum(dim=2) * mask # (batch_size, ped_num)
        loss = loss.sum()/mask.sum()
        losses['trajectory_loss'] = loss
        if latent_distrib is not None:
            mu_gt, logvar_gt, mu_history, logvar_history = latent_distrib
            losses['KL_loss'] = self.latent_predictor.cal_loss(mu_gt, logvar_gt, mu_history, logvar_history, mask)
        return losses

class GTPPO:
    def __init__(self, params):
        """
        GTPPO class, following a sklearn similar class structure
        :param params: dictionary with hyperparameters
        """
        self.params = params
        self.obs_len = params['obs_len']
        self.pred_len = params['pred_len']

        self.model = GTPPO_model(params)

    def train(self, train_data_path, val_data_path, params, experiment_name, logger: Logger, writer:SummaryWriter, output_dir, device=None, dataset_name=""):
        """
        Train function
        :param train_data_path: path to train data
        :param val_data_path: path to validation data
        :param params: dictionary with training hyperparameters
        :param experiment_name: str, arbitrary name to name weights file, and logger
        :param logger: logger object
        :param output_dir: str, path to output directory
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        :return:
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info('Preprocess data...')
        dataset_name = dataset_name.lower()


        # Load train images and augment train data and images
        # TODO: implement augmentation
        # df_train, train_images = augment_data(train_data, image_path=train_image_path, image_file=image_file_name,
        #                                       seg_mask=seg_mask)

        # Initialize dataloaders
        train_dataset = TrajectoryDataset(train_data_path, params['obs_len'], params['pred_len'],
                                            params['skip'], params['non_linear_threshold'],
                                            params['min_ped'], delim='\t', 
                                            use_prepared_data=True, dump_prepared_data=True,
                                            use_intention_cluster=params['use_intention_cluster'])
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], collate_fn=gtppo_collate, shuffle=True, num_workers=0)

        val_dataset = TrajectoryDataset(val_data_path, params['obs_len'], params['pred_len'],
                                         params['skip'], params['non_linear_threshold'],
                                        params['min_ped'], delim='\t',
                                        use_prepared_data=True, dump_prepared_data=True,
                                        use_intention_cluster=params['use_intention_cluster'])
        val_loader = DataLoader(val_dataset, params['batch_size'], collate_fn=gtppo_collate, num_workers=0)

        model = self.model.to(device)

        # Freeze segmentation model
        # for param in model.semantic_segmentation.parameters():
        #     param.requires_grad = False
        

        # set different lr for different layers
        optimizer = torch.optim.Adam([
            {'params': self.model.encoder_decoder.parameters()},
            {'params': self.model.latent_predictor.parameters(), 'lr': params["lr"]['latent_predictor']},
            {'params': self.model.graph_module.parameters()}],
            lr=params["lr"]['base_lr'])
        # criterion = nn.BCEWithLogitsLoss()


        best_test_ADE = 99999999999999

        self.train_ADE = []
        self.train_FDE = []
        self.val_ADE = []
        self.val_FDE = []
        self.train_loss = []
        logger.info('Start training')
        for e in tqdm(range(params['num_epochs']), desc='Training'):
            t1 = time.time()
            logger.info(f'Start Epoch {e}')
            train_ADE, train_FDE, train_loss = train(model, train_loader, params, optimizer, e, device)
            self.train_ADE.append(train_ADE)
            self.train_FDE.append(train_FDE)
            self.train_loss.append(train_loss)
            val_ADE, val_FDE = evaluate(model, val_loader,
                                        params, e, device)
            logger.info(f'Epoch {e}: Val ADE: {val_ADE} Val FDE: {val_FDE}')
            self.val_ADE.append(val_ADE)
            self.val_FDE.append(val_FDE)

            t2 = time.time()
            logger.info('Epoch {0:} took {1:.4f} seconds'.format(e, t2-t1))

            # write to tensorboard
            writer.add_scalars('ADE', {'train': train_ADE, 'val': val_ADE}, e)
            writer.add_scalars('FDE', {'train': train_FDE, 'val': val_FDE}, e)
            writer.add_scalars('Loss', train_loss, e)

            if val_ADE < best_test_ADE:
                logger.info(f'Best Epoch {e}: Val ADE: {val_ADE} val FDE: {val_FDE}')
                checkpoint = {'epoch': e,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'val_ADE': val_ADE,
                                'params': params
                            }
                torch.save(checkpoint, os.path.join(output_dir, experiment_name + '_weights.pt'))
                best_test_ADE = val_ADE
        # save model at the end of training
        checkpoint = {'epoch': e,
                    'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'val_ADE': val_ADE,
                    'params': params
                    }
        torch.save(checkpoint, os.path.join(output_dir, experiment_name + '_final_weights.pt'))
        writer.flush()

    def evaluate(self, data_path, params, logger: Logger, writer:SummaryWriter, device=None, dataset_name=""):
        """
        Val function
        :param data_path: path to val data
        :param params: dictionary with training hyperparameters
        :param logger: logger object
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        :param dataset_name: str, name of dataset
        :return:
        """

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info('Preprocess data')

        test_dataset = TrajectoryDataset(
            data_path, params['obs_len'], params['pred_len'],
            params['skip'], params['non_linear_threshold'], params['min_ped'],
            delim='\t',use_prepared_data=True, dump_prepared_data=True,
            use_intention_cluster=params['use_intention_cluster'])
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], collate_fn=gtppo_collate, num_workers=0)

        model = self.model.to(device)

        self.eval_ADE = []
        self.eval_FDE = []

        rounds = params['test_rounds']
        logger.info('Start testing')
        for e in tqdm(range(rounds), desc='Testing'):
            logger.info(f'Round {e}')
            t1 = time.time()
            test_ADE, test_FDE = evaluate(model, test_loader, params, e, device)
            t2 = time.time()
            logger.info('Round {0:} took {1:.4f} seconds'.format(e, t2-t1))
            logger.info(f'Round {e}: \nTest ADE: {test_ADE} \nTest FDE: {test_FDE}')

            self.eval_ADE.append(test_ADE)
            self.eval_FDE.append(test_FDE)

        logger.info(f'\n\nAverage performance over {rounds} rounds: \nTest ADE: {sum(self.eval_ADE) / len(self.eval_ADE)} \nTest FDE: {sum(self.eval_FDE) / len(self.eval_FDE)}')
        writer.flush()


    def load(self, path, logger):
        checkpoint = torch.load(path)
        logger.info(f'Loaded model from epoch {checkpoint["epoch"]}')
        logger.info(self.model.load_state_dict(checkpoint['model']))

    def save(self, path):
        torch.save(self.model.state_dict(), path)



























