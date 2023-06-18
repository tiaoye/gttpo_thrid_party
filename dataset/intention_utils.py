import torch 
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def rotate_to_xaxis(obs_trajs,pred_trajs):
    '''Rotate the trajectory to x-axis, i.e. the last obs step velocity direction is x-axis.

    Args:
        obs_trajs: (N, 2, obs_len)
        pred_trajs: (N, 2, pred_len)
    Returns:
        obs_trajs_rot: (N, 2, obs_len)
        pred_trajs_rot: (N, 2, pred_len)
    '''
    # get velocity direction
    last_step_vel = obs_trajs[:, :, -1] - obs_trajs[:, :, -2] # (N, 2)
    theta = torch.atan2(last_step_vel[:, 1], last_step_vel[:, 0]) # angle between x-axis and velocity direction (N,)

    # get rotation matrix 
    theta = theta.unsqueeze(1).unsqueeze(2) # (N, 1, 1) 
    rotation_matrix = torch.cat([torch.cos(theta), torch.sin(theta), -torch.sin(theta), torch.cos(theta)], dim=1)  
    rotation_matrix = rotation_matrix.reshape(-1, 2, 2) # (N, 2, 2)

    # rotate the obs trajectory 
    center = obs_trajs[:, :, -1:].clone() # (N, 2, 1) 
    obs_trajs_rot = (obs_trajs - center) # (N, 2, obs_len) 
    # print("obs_trajs_rot: ", obs_trajs_rot.shape) 
    # print("rotation_matrix: ", rotation_matrix.shape) 
    obs_trajs_rot = torch.matmul(rotation_matrix, obs_trajs_rot) # (N, 2, obs_len)

    # rotate the pred trajectory
    pred_trajs_rot = (pred_trajs - center) # (N, 2, pred_len)
    pred_trajs_rot = torch.matmul(rotation_matrix, pred_trajs_rot) # (N, 2, pred_len)

    # evaluate the rotation 
    last_step_vel_rot = obs_trajs_rot[:, :, -1] - obs_trajs_rot[:, :, -2] # (N, 2) 
    last_step_vel_rot = last_step_vel_rot # (N, 2) 
    theta_rot = torch.atan2(last_step_vel_rot[:, 1], last_step_vel_rot[:, 0]) # angle between x-axis and velocity direction (N,) 
    # print("assert theta_rot: ", torch.allclose(theta_rot, torch.zeros_like(theta_rot), atol=1e-5)) # theta_rot should be zero, i.e. the last step velocity direction should be x-axis 
    # print("assert last_step_vel_rot: ", torch.allclose(last_step_vel_rot[:,1], torch.zeros_like(last_step_vel_rot[:,1]), atol=1e-5)) # vel_y should be zero

    return obs_trajs_rot, pred_trajs_rot


color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'olive', 'indigo', 'teal', 'skyblue',
        'coral', 'darkorange', 'lime', 'darkorchid', 'saddlebrown', 'khaki', 'dimgray'])
def plot_cluster(traj_lst, cluster_lst, subplot=None, plot_curves=True):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    if subplot is not None, draw each cluster in a subplot
    if plot_curves is False, draw each trajectory as a endpoint
    '''
    cluster_lst = np.array(cluster_lst)
    cluster_type = np.unique(cluster_lst)
    cluster_count = cluster_type.shape[0]
    assert cluster_count <= len(color_lst), "Too many clusters to plot"
    for i,cluster in enumerate(cluster_type):
        if subplot is not None:
            plt.subplot(subplot[0], subplot[1], i+1)
            plt.title("Cluster {}".format(cluster))
            plt.xlim([-6, 13])
            plt.ylim([-8, 8]) 
        traj = traj_lst[cluster_lst == cluster]
        if cluster == -1:
            # Means it it a noisy trajectory, paint it black
            if plot_curves:
                plt.plot(traj[:, 0].T, traj[:, 1].T, c='k', linestyle='dashed')
            else:
                plt.scatter(traj[:, 0,-1], traj[:, 1,-1], c='k', marker='x', s=5, alpha=0.5)
            #continue                  
        else:
            if plot_curves:
                plt.plot(traj[:, 0].T, traj[:, 1].T, c=color_lst[cluster % len(color_lst)])
            else:
                plt.scatter(traj[:, 0,-1], traj[:, 1,-1], c=color_lst[cluster % len(color_lst)], s=5, alpha=0.5)

    plt.xlim([-6, 13])
    plt.ylim([-8, 8]) 
    plt.show()


def kmeans_cluster(trajs, cluster_num=4):
    '''
    cluster trajectories using kmeans, the last step of each trajectory is used as the feature
    Args:
        trajs: (N, 2, seq_len)
        cluster_num: number of clusters
    Returns:
        cluster_lst: (N, )
        centers: (cluster_num, 2)
        kmeans: KMeans object
    '''
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(trajs[:, :, -1])
    cluster_lst = kmeans.labels_
    centers = kmeans.cluster_centers_
    return cluster_lst, centers, kmeans

def save_cluster(file_path, kmeans_cluster: KMeans):
    with open(file_path, 'wb') as f:
        pickle.dump(kmeans_cluster, f)

def load_cluster(file_path)->KMeans:
    with open(file_path, 'rb') as f:
        kmeans_cluster = pickle.load(f)
    return kmeans_cluster


if __name__ == "__main__":
    from trajectories import TrajectoryDataset
    '''
    do kmeans clustering on the given train and validation dataset, dont use test dataset.the cluster is saved at the root folder of the dataset
    '''
    dataset_name = 'eth'
    train_path = os.path.join("data/ethucy_sgan", dataset_name, 'train')
    val_path = os.path.join("data/ethucy_sgan", dataset_name, 'val')
    train_set = TrajectoryDataset(train_path, use_prepared_data=True, dump_prepared_data=True)
    val_set = TrajectoryDataset(val_path, use_prepared_data=True, dump_prepared_data=True)
    obs_trajs = torch.cat([train_set.obs_traj, val_set.obs_traj], dim=0)
    pred_trajs = torch.cat([train_set.pred_traj, val_set.pred_traj], dim=0)

    # rotate the trajectory
    obs_trajs_rot, pred_trajs_rot = rotate_to_xaxis(obs_trajs, pred_trajs)

    # cluster the rotated trajectory
    kmeans = load_cluster("data/ethucy_sgan/raw/all_data/cluster_result_4.pkl")
    print("cluster_lst: ", kmeans.labels_.shape)
    print("centers: ",  kmeans.cluster_centers_.shape)
    print("kmeans: ", kmeans)

    # plot the cluster
    plot_cluster(pred_trajs_rot, kmeans.predict(pred_trajs_rot[:,:,-1]), subplot=(2,2), plot_curves=False)

    # save the cluster
    # save_cluster(os.path.join("data/ethucy_sgan", dataset_name, 'kmeans_cluster.pkl'), kmeans)

