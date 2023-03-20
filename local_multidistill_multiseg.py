import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import scipy.integrate as integrate
from collections import deque
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools
import time

from multiutils_multiseg import *
from multi_modules import MultiNetwork, MultiNetworkFourierEmbedding, MultiNetworkLinear

# TODO: move this to utils.py
class Node:
    def __init__(self):
        pass

def create_multi_network_fourier_embedding(num_networks, num_frequencies):
    fourier_embedding = None
    num_input_channels = 3
    if num_frequencies > 0:
        fourier_embedding = MultiNetworkFourierEmbedding(num_networks, 3, num_frequencies)
        num_input_channels = fourier_embedding.num_output_channels
    return num_input_channels, fourier_embedding

def has_flag(cfg, name):
    return name in cfg and cfg[name]

def convert_index_to_res_3d(res,index):
    system_zero = res[1]*res[2]
    system_one = res[2]
    res_x = math.floor(index/system_zero)
    res_y = math.floor((index-res_x*system_zero)/system_one)
    res_z = index-res_x*system_zero-res_y*system_one
    res_3d = [res_x,res_y,res_z]
    return res_3d

def convert_res_3d_to_preres_3d(res,pre_res,res_3d):
    system_x = res[0] / pre_res[0]
    system_y = res[1] / pre_res[1]
    system_z = res[2] / pre_res[2]
    pre_res_x = math.floor(res_3d[0]/system_x)
    pre_res_y = math.floor(res_3d[1]/system_y)
    pre_res_z = math.floor(res_3d[2]/system_z)
    pre_res_3d = [pre_res_x,pre_res_y,pre_res_z]
    return pre_res_3d

def convert_res_3d_to_index(res,res_3d):
    system_zero = res[1] * res[2]
    system_one = res[2]
    res_index = res_3d[0]*system_zero + res_3d[1]*system_one + res_3d[2]
    return res_index

def convert_res_to_pre_res(res,pre_res):
    num_networks = res[0]*res[1]*res[2]
    num_pre_networks = pre_res[0]*pre_res[1]*pre_res[2]
    indices_index= np.zeros(num_networks)
    for index in range(num_networks):
        res_3d = convert_index_to_res_3d(res,index)
        pre_res_3d = convert_res_3d_to_preres_3d(res,pre_res,res_3d)
        pre_res_index = convert_res_3d_to_index(pre_res,pre_res_3d)
        indices_index[index] = pre_res_index
    return indices_index

def create_multi_network(num_networks, num_position_channels, num_direction_channels, num_output_channels, linear_implementation, cfg):
    refeed_position_index = None

    if 'refeed_position_index' in cfg:
        refeed_position_index = cfg['refeed_position_index']
    late_feed_direction = False
    if 'late_feed_direction' in cfg:
        late_feed_direction = cfg['late_feed_direction']
    direction_layer_size = cfg['hidden_layer_size']
    if 'direction_layer_size' in cfg:
        direction_layer_size = cfg['direction_layer_size']
    nonlinearity = 'relu'
    if 'nonlinearity' in cfg:
        nonlinearity = cfg['nonlinearity']
    nonlinearity_initalization = 'pass_leaky_relu'
    if 'nonlinearity_initalization' in cfg:
        nonlinearity_initalization = cfg['nonlinearity_initalization']
    use_single_net =  False
    if has_flag(cfg, 'use_single_net'):
        use_single_net = True
    use_same_initialization_for_all_networks = False
    if 'use_same_initialization_for_all_networks' in cfg:
        use_same_initialization_for_all_networks = cfg['use_same_initialization_for_all_networks']
    network_rng_seed = None
    if 'network_rng_seed' in cfg:
        network_rng_seed = cfg['network_rng_seed']
    weight_initialization_method = 'kaiming_uniform'
    if 'weight_initialization_method' in cfg:
        weight_initialization_method = cfg['weight_initialization_method']
    bias_initialization_method = 'standard'
    if 'bias_initialization_method' in cfg:
        bias_initialization_method = cfg['bias_initialization_method']
    alpha_rgb_initalization = 'updated_yenchenlin'
    if 'alpha_rgb_initalization' in cfg:
        alpha_rgb_initalization = cfg['alpha_rgb_initalization'] 
    use_hard_parameter_sharing_for_color = has_flag(cfg, 'use_hard_parameter_sharing_for_color')
    view_dependent_dropout_probability = -1
    if 'view_dependent_dropout_probability' in cfg:
        view_dependent_dropout_probability = cfg['view_dependent_dropout_probability']
    use_view_independent_color = False
    if 'use_view_independent_color' in cfg:
        use_view_independent_color = cfg['use_view_independent_color']
    return MultiNetwork(num_networks, num_position_channels, num_direction_channels, num_output_channels,
        cfg['hidden_layer_size'], cfg['num_hidden_layers'], refeed_position_index, late_feed_direction,
        direction_layer_size, nonlinearity, nonlinearity_initalization, use_single_net, linear_implementation,
        use_same_initialization_for_all_networks, network_rng_seed, weight_initialization_method, bias_initialization_method, alpha_rgb_initalization,
        use_hard_parameter_sharing_for_color, view_dependent_dropout_probability, use_view_independent_color)
    
def convert_to_local_coords_multi(points, domain_mins, domain_maxs):
    converted_points = torch.empty_like(points)
    for i in [0, 1, 2]:
        # values between -1 and 1
        converted_points[:,:,i] = 2 * (points[:,:,i] - domain_mins[:,i].unsqueeze(1)) / (domain_maxs[:,i].unsqueeze(1) - domain_mins[:,i].unsqueeze(1)) - 1
    return converted_points
    
def preprocess_examples(batch_examples, domain_mins, domain_maxs, cfg, position_fourier_embedding, direction_fourier_embedding):
    if cfg['outputs'] == 'density':
        batch_inputs = batch_examples[:, :, 0:3]
        if not has_flag(cfg, 'use_global_coordinates'):
            batch_inputs = convert_to_local_coords_multi(batch_inputs, domain_mins, domain_maxs)
        if cfg['num_frequencies'] > 0:
            batch_inputs = position_fourier_embedding(batch_inputs)
        batch_targets = batch_examples[:, :, 3].unsqueeze(2)
    elif cfg['outputs'] == 'color_and_density':
        batch_positions = batch_examples[:, :, 0:3]
        if not has_flag(cfg, 'use_global_coordinates'):
            batch_positions = convert_to_local_coords_multi(batch_positions, domain_mins, domain_maxs)
        if cfg['num_frequencies'] > 0:
            batch_positions = position_fourier_embedding(batch_positions)
        batch_directions = batch_examples[:, :, 3:6]
        if cfg['num_frequencies_direction'] > 0:
            batch_directions = direction_fourier_embedding(batch_directions)
        batch_inputs = torch.cat((batch_positions, batch_directions), dim=2)
        batch_targets = batch_examples[:, :, 6:10]
    return batch_inputs, batch_targets
    

def postprocess_output(raw_output, cfg):
    def process_density(raw_output):
        if has_flag(cfg, 'convert_density_to_alpha'):
            return (1. - torch.exp(-F.leaky_relu(raw_output[:, :, 3]) * cfg['alpha_distance'])).unsqueeze(2) # Convert to alpha with typical distance encountered during training
        else:
            return F.leaky_relu(raw_output[:, :, 3]).unsqueeze(2)  # Only apply ReLU to density output
    if cfg['outputs'] == 'density':
        out = process_density(raw_output)
    elif cfg['outputs'] == 'color_and_density':
        if has_flag(cfg, 'no_color_sigmoid'):
            rgb = raw_output[:, :, 0:3]
        else:
            rgb = F.sigmoid(raw_output[:, :, 0:3])
        density = process_density(raw_output)
        out = torch.cat((rgb, density), dim=2)
    return out
    
def list_metrics():
    return ['mse', 'mae', 'mape', 'quantile_se']
 
def train_and_test_multi_network(multi_network, all_examples, domain_mins, domain_maxs, position_fourier_embedding, direction_fourier_embedding, processing_saturated_nodes, cfg):
    if processing_saturated_nodes == False:
        initial_lr = cfg['initial_lr'] if 'initial_lr' in cfg else 0.001
    else:
        initial_lr = cfg['saturated_initial_lr'] if 'saturated_initial_lr' in cfg else 0.0001
    initial_lr = float(initial_lr)
    optimizer = optim.Adam(multi_network.parameters(), lr=initial_lr)
    if 'lr_decay_iterations' in cfg:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg['lr_decay_iterations'], cfg['lr_decay_factor'])
    num_networks = all_examples.size(0)
    train_examples = all_examples[:, :cfg['num_train_examples_per_network']]
    test_examples = all_examples[:, cfg['num_train_examples_per_network']:]
    
    best_errors_per_network, best_errors_per_network_color, best_errors_per_network_density = {}, {}, {}
    for metric in list_metrics():
        best_errors_per_network[metric] = float('inf') * torch.ones(num_networks)
        best_errors_per_network_color[metric] = float('inf') * torch.ones(num_networks)
        best_errors_per_network_density[metric] = float('inf') * torch.ones(num_networks)
    
    error_log = ['{} {}\n'.format(domain_mins[network_index].cpu().tolist(),
        domain_maxs[network_index].cpu().tolist()) for network_index in range(num_networks)]
    start_time = time.time()
    for iteration in range(1, cfg['iterations'] + 1):
        if has_flag(cfg, 'lr_exp_decay_steps'):
            new_lr = initial_lr * 0.1 ** (iteration / cfg['lr_exp_decay_steps'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        optimizer.zero_grad()
        indices = np.random.choice(cfg['num_train_examples_per_network'], size=(cfg['train_batch_size'],))
        train_batch_inputs, train_batch_targets = preprocess_examples(train_examples[:, indices].to(domain_mins), domain_mins, domain_maxs, cfg, position_fourier_embedding, direction_fourier_embedding)
        raw_output = multi_network(train_batch_inputs)
        out = postprocess_output(raw_output, cfg)
        loss = nn.functional.mse_loss(out, train_batch_targets, reduction='none')
        loss = loss.mean(dim=2).mean(dim=1).sum()
        loss.backward()
        optimizer.step()
        if iteration % 1000 == 0:
            print('{}: sum: {:.5f} avg: {:.5f}'.format(iteration, loss.item(), loss.item() / num_networks))
        if iteration % cfg['test_every'] == 0 or iteration == cfg['iterations']:
            errors_per_point, errors_per_network, errors_per_network_color, errors_per_network_density, saturation =\
                test_multi_network(multi_network, test_examples, domain_mins, domain_maxs, cfg, position_fourier_embedding, direction_fourier_embedding)
                
            for metric in list_metrics():
                best_errors_per_network[metric] = torch.min(errors_per_network[metric], best_errors_per_network[metric])
                if cfg['outputs'] == 'color_and_density':
                    best_errors_per_network_color[metric] = torch.min(errors_per_network_color[metric], best_errors_per_network_color[metric])
                    best_errors_per_network_density[metric] = torch.min(errors_per_network_density[metric], best_errors_per_network_density[metric])
            
            for network_index in range(num_networks):
                error_log[network_index] += 'it: {} | '.format(iteration)
                if 'performance_measurement' in cfg:
                    error_log[network_index] += '(%.5f) ' % (time.time() - start_time)
                for metric in list_metrics():
                    error_log[network_index] += metric + ': {:.5f} '.format(errors_per_network[metric][network_index].item())
                    if cfg['outputs'] == 'color_and_density':
                        error_log[network_index] += '(d: {:.5f}, c: {:.5f}) '.format(errors_per_network_density[metric][network_index].item(),
                            errors_per_network_color[metric][network_index].item())
                if saturation[network_index]:
                    error_log[network_index] += ' [saturation detected]'
                error_log[network_index] += '\n'      
            if has_flag(cfg, 'show_intermediate_error_log'):
                Logger.write('\n'.join(error_log))
        if 'lr_decay_iterations' in cfg:
            scheduler.step()
    Logger.write('\n'.join(error_log))
    test_points = test_examples[:, :, :3]
    return test_points, errors_per_point, best_errors_per_network, best_errors_per_network_color, best_errors_per_network_density, saturation
    
def test_multi_network(multi_network, test_examples, domain_mins, domain_maxs, cfg, position_fourier_embedding, direction_fourier_embedding):
    num_networks = test_examples.size(0)
    num_test_examples = test_examples.size(1)
    if 'test_batch_size' in cfg:
        test_batch_size = cfg['test_batch_size']
    else:
        test_batch_size = num_test_examples
    with torch.no_grad():
        if cfg['outputs'] == 'density':
            num_output_channels = 1
        if cfg['outputs'] == 'color_and_density':
            num_output_channels = 4
        out = torch.empty(num_networks, num_test_examples, num_output_channels).to(test_examples)
        test_targets = torch.empty(num_networks, num_test_examples, num_output_channels).to(test_examples)
        start = 0
        while start < num_test_examples:
            end = min(start + test_batch_size, num_test_examples)
            test_batch_inputs, test_batch_targets = preprocess_examples(test_examples[:, start:end].to(domain_mins), domain_mins, domain_maxs, cfg, position_fourier_embedding, direction_fourier_embedding)
            test_targets[:, start:end] = test_batch_targets
            raw_output = multi_network(test_batch_inputs)
            out[:, start:end] = postprocess_output(raw_output, cfg)
            start = end

        # For a small fraction of networks/regions the RGB sigmoids get trapped in an all 0 or 1 state
        # We detect when this happens in order to retrain these networks with a smaller learning rate
        tolerance = 0.001
        close_to_zero = (torch.abs(out[:, :, :3] - torch.zeros_like(out[:, :, :3])) < tolerance).all(dim=1)
        gt_close_to_zero = (torch.abs(test_targets[:, :, :3] - torch.zeros_like(test_targets[:, :, :3])) < tolerance).all(dim=1)
        saturation_zero = torch.logical_and(close_to_zero,  torch.logical_not(gt_close_to_zero)).any(dim=1)
        
        close_to_one = (torch.abs(out[:, :, :3] - torch.ones_like(out[:, :, :3])) < tolerance).all(dim=1)
        gt_close_to_one = (torch.abs(test_targets[:, :, :3] - torch.ones_like(test_targets[:, :, :3])) < tolerance).all(dim=1)
        saturation_one = torch.logical_and(close_to_one,  torch.logical_not(gt_close_to_one)).any(dim=1)
        
        saturation = torch.logical_or(saturation_zero, saturation_one)

        errors, errors_per_point, errors_per_network, errors_per_network_color, errors_per_network_density = {}, {}, {}, {}, {}
        errors['mse'] = nn.functional.mse_loss(out, test_targets, reduction='none')
        errors['mae'] = torch.abs(out - test_targets)
        mape_epsilon = 0.1
        errors['mape'] = errors['mae'] / (torch.abs(test_targets) + mape_epsilon)

        for metric in ['mse', 'mape', 'mae']:
            errors_per_point[metric] = errors[metric].mean(dim=2)
            errors_per_network[metric] = errors_per_point[metric].mean(dim=1).cpu()
            if cfg['outputs'] == 'density':
                errors_per_network_density[metric] = errors_per_network[metric]
            if cfg['outputs'] == 'color_and_density':
                errors_per_network_color[metric] = errors[metric][:, :, :3].mean(dim=2).mean(dim=1).cpu()
                errors_per_network_density[metric] = errors[metric][:, :, 3].mean(dim=1).cpu()
            errors_per_point[metric] = errors_per_point[metric].cpu()
        
        def calcululate_quantile(se_per_point):
            num_test_samples = errors['mse'].size(1)
            quantile_index = int(num_test_samples * cfg['quantile_se'])
            sorted_se_per_point = torch.sort(se_per_point, dim=1)[0]
            return sorted_se_per_point[:, quantile_index].cpu()
        
        errors_per_point['quantile_se'] = None # not really defined and this value should never be used
        errors_per_network['quantile_se'] = calcululate_quantile(errors['mse'].mean(dim=2))
        errors_per_network_color['quantile_se'] = calcululate_quantile(errors['mse'][:, :, :3].mean(dim=2))
        errors_per_network_density['quantile_se'] = calcululate_quantile(errors['mse'][:, :, 3])
        
        #visualize_errors(test_examples, out, test_targets)
    
    return errors_per_point, errors_per_network, errors_per_network_color, errors_per_network_density, saturation
    
def visualize_errors(test_examples, out, test_targets):
    num_samples = 2500
    
    xs = test_examples[0, :num_samples, 0].cpu().numpy()
    ys = test_examples[0, :num_samples, 1].cpu().numpy()
    zs = test_examples[0, :num_samples, 2].cpu().numpy()
    density_errors = nn.functional.mse_loss(out[0, :num_samples, 3], test_targets[0, :num_samples, 3], reduction='none').cpu().numpy()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    l = ax.scatter(xs, ys, zs, c=density_errors)
    fig.colorbar(l)
    plt.show()

def calculate_volume(domain_min, domain_max):
    return (domain_max[0] - domain_min[0]) * (domain_max[1] - domain_min[1]) * (domain_max[2] - domain_min[2])
    
def get_equal_error_split_threshold(test_points, errors, split_axis):
    test_points = test_points.numpy()
    errors = errors.numpy()
    half_error_sum = np.sum(errors) / np.array(2.)
    points_sort = np.argsort(test_points[:, split_axis])
    split_threshold = test_points[points_sort][np.nonzero(np.cumsum(np.cumsum(errors[points_sort]) > half_error_sum) == 1)][0, split_axis]
    return split_threshold

def train_and_test_nodes(node_batch,cycle_index, pretrained_nerf,embed_fn,embeddirs_fn, processing_saturated_nodes,pretrain_cfg_res,cfg_res, pretrain_cfg,cfg,global_domain_min, global_domain_max, dev):
    num_networks = len(node_batch)
    Logger.write('training {} networks in parallel'.format(num_networks))
    position_num_input_channels, position_fourier_embedding = create_multi_network_fourier_embedding(num_networks, cfg['num_frequencies'])
    if cfg['outputs'] == 'density': 
        direction_num_input_channels, direction_fourier_embedding = 0, None
        num_output_channels = 1
    if cfg['outputs'] == 'color_and_density':
        direction_num_input_channels, direction_fourier_embedding = create_multi_network_fourier_embedding(num_networks, cfg['num_frequencies_direction'])
        num_output_channels = 4
    MultiNetworkLinear.rng_state = None
    multi_network =  create_multi_network(num_networks, position_num_input_channels, direction_num_input_channels, 4, 'bmm', cfg).to(dev)
    all_examples = torch.empty(num_networks * cfg['num_examples_per_network'], 6 + num_output_channels) # x,y,z,dir_x,dir_y,dir_z,r,g,b,a
    start = 0
    for network_index in range(num_networks):
        start = network_index * cfg['num_examples_per_network']
        end = (network_index + 1) * cfg['num_examples_per_network']
        if 'enlarge_domain_factor' in cfg:
            enlarged_domain_min = np.array(node_batch[network_index].domain_min)
            enlarged_domain_max = np.array(node_batch[network_index].domain_max)
            lengths = (enlarged_domain_max - enlarged_domain_min) / 2
            enlarged_domain_max += cfg['enlarge_domain_factor'] * lengths
            enlarged_domain_min -= cfg['enlarge_domain_factor'] * lengths
            train_end = network_index * cfg['num_examples_per_network'] + cfg['num_train_examples_per_network']
            all_examples[start:train_end, 0:3] = torch.tensor(get_random_points_inside_domain(cfg['num_train_examples_per_network'], enlarged_domain_min, enlarged_domain_max), dtype=torch.float)
            all_examples[train_end:end, 0:3] = torch.tensor(get_random_points_inside_domain(cfg['num_examples_per_network'] - cfg['num_train_examples_per_network'],
                node_batch[network_index].domain_min, node_batch[network_index].domain_max), dtype=torch.float)
        else:
            all_examples[start:end, 0:3] = torch.tensor(get_random_points_inside_domain(cfg['num_examples_per_network'], node_batch[network_index].domain_min, node_batch[network_index].domain_max), dtype=torch.float)
        all_examples[start:end, 3:6] = torch.tensor(get_random_directions(cfg['num_examples_per_network']), dtype=torch.float)

    points_and_dirs = all_examples[:, 0:6]

    all_examples = all_examples.view(num_networks, cfg['num_examples_per_network'], -1)
    points_and_dirs = points_and_dirs.view(num_networks, cfg['num_examples_per_network'], -1)

##############################################################################################################################
    pre_res = pretrain_cfg_res['fixed_resolution']
    res = cfg_res['fixed_resolution']
    res_preres_scale = cfg_res['res_preres_scale']
    num_pre_networks = pre_res[1]*pre_res[2]
    pre_cycle_index = math.floor(cycle_index*pre_res[0]*res_preres_scale/res[0])

    num_pre_examples_per_network = int(cfg['num_examples_per_network']*num_networks/num_pre_networks)
    pre_points_and_dirs = torch.empty(num_pre_networks ,num_pre_examples_per_network , 6)
    pre_all_examples = torch.empty(num_pre_networks ,num_pre_examples_per_network , 10)

    indice_index = convert_res_to_pre_res(res, pre_res)

    indice_index = indice_index[cycle_index*num_networks:(cycle_index+1)*num_networks]



    res_to_pre_res_index = np.zeros(shape=(num_pre_networks, int(num_networks/num_pre_networks)))    # relationship between two networks

    #compute res_to_pre_res_index & pre_points_and_dirs
    for i in range(num_pre_networks):
        a=np.where( indice_index== (i+pre_cycle_index*num_pre_networks) )
        b=a[0]
        res_to_pre_res_index[i,:]=b
        points_and_dirs_process=points_and_dirs[b,:,:]
        points_and_dirs_process=points_and_dirs_process.view(num_pre_examples_per_network,6)
        pre_points_and_dirs[i,:,:] = points_and_dirs_process
        all_examples_process=all_examples[b,:,:]
        all_examples_process=all_examples_process.view(num_pre_examples_per_network,10)
        pre_all_examples[i,:,:] = all_examples_process

    del points_and_dirs_process
    del all_examples_process
##################################################################################################################################
    num_points_and_dirs = pre_points_and_dirs.shape[1]

    fixed_resolution=pre_res
    voxel_size = (global_domain_max - global_domain_min) / fixed_resolution
    start=0
    pre_domain_mins_whole = torch.empty(pre_res[0]*pre_res[1]*pre_res[2],3)
    pre_domain_maxs_whole = torch.empty(pre_res[0]*pre_res[1]*pre_res[2],3)
    for voxel_indices in itertools.product(*[range(axis_resolution) for axis_resolution in fixed_resolution]):
        domain_min = (global_domain_min + voxel_indices * voxel_size)
        pre_domain_mins_whole[start, 0] = domain_min[0]
        pre_domain_mins_whole[start, 1] = domain_min[1]
        pre_domain_mins_whole[start, 2] = domain_min[2]
        domain_max = (global_domain_min + (voxel_indices + np.array(1)) * voxel_size)
        pre_domain_maxs_whole[start, 0] = domain_max[0]
        pre_domain_maxs_whole[start, 1] = domain_max[1]
        pre_domain_maxs_whole[start, 2] = domain_max[2]
        start=start+1



    #pre_points_and_dirs = pre_points_and_dirs.to(dev)
    pre_domain_mins = pre_domain_mins_whole[pre_cycle_index*num_pre_networks:(pre_cycle_index+1)*num_pre_networks,:]
    pre_domain_maxs = pre_domain_maxs_whole[pre_cycle_index*num_pre_networks:(pre_cycle_index+1)*num_pre_networks,:]
    pre_domain_mins = pre_domain_mins.to(dev)
    pre_domain_maxs = pre_domain_maxs.to(dev)




    if 'query_batch_size' in cfg:
        query_batch_size = cfg['query_batch_size']
    else:
        query_batch_size = num_points_and_dirs
    with torch.no_grad():
        start = 0
        while start < num_points_and_dirs:
            end = min(start + query_batch_size, num_points_and_dirs)
            train_batch_inputs, _ = preprocess_examples(pre_points_and_dirs[:,start:end].to(pre_domain_mins), pre_domain_mins,
                                                                          pre_domain_maxs, pretrain_cfg, embed_fn,
                                                                          embeddirs_fn)

            raw_multioutput = pretrained_nerf(train_batch_inputs) # Get complete RGBA output from NeRF    important
            output = postprocess_output(raw_multioutput, cfg)
            if cfg['outputs'] == 'density':
                pre_all_examples[:,start:end, 3] = output
            if cfg['outputs'] == 'color_and_density':
                pre_all_examples[:,start:end, 6:10] = output

            if has_flag(cfg, 'use_premultiplied_colors'):
                pre_all_examples[:,start:end, 6:9] *= output

            del raw_multioutput
            del output
            start = end
    if cfg['outputs'] == 'density':
        pre_all_examples = pre_all_examples[:, :4]



    num_points_and_dirs = int(all_examples.shape[1])

    for i in range(res_to_pre_res_index.shape[0]):
        start = 0
        for j in range(res_to_pre_res_index.shape[1]):
            num_networks_index = int(res_to_pre_res_index[i,j])
            pre_num_networks_index = i
            end = start + num_points_and_dirs
            all_examples[num_networks_index] = pre_all_examples[pre_num_networks_index,start:end,:]
            start = end


##########################################################################




    domain_mins = torch.tensor([node_batch[network_index].domain_min for network_index in range(num_networks)],
                               dtype=torch.float).to(dev)
    domain_maxs = torch.tensor([node_batch[network_index].domain_max for network_index in range(num_networks)],
                               dtype=torch.float).to(dev)
    test_points, errors_per_point, best_errors_per_network, best_errors_per_network_color, best_errors_per_network_density, saturation = \
        train_and_test_multi_network(multi_network, all_examples, domain_mins, domain_maxs, position_fourier_embedding,
                                     direction_fourier_embedding, processing_saturated_nodes, cfg)

    return multi_network, test_points, errors_per_point, best_errors_per_network, best_errors_per_network_color, best_errors_per_network_density, saturation



def log_error_stats(initial_nodes, phase, cfg):
    domain_mins = []
    domain_maxs = []
    volumes = []
    best_errors = {}
    if cfg[phase]['outputs'] == 'color_and_density':
        best_errors_color = {}
        best_errors_density = {}
    for metric in list_metrics():
        best_errors[metric] = []
        if cfg[phase]['outputs'] == 'color_and_density':
            best_errors_color[metric] = []
            best_errors_density[metric] = []

    nodes_to_visit = deque(initial_nodes)
    while nodes_to_visit:
        node = nodes_to_visit.popleft()
        if hasattr(node, 'leq_child'):
            nodes_to_visit.append(node.leq_child)
            nodes_to_visit.append(node.gt_child)
        if (phase == 'discovery' and hasattr(node, 'discovery_best_error')) or (phase == 'final' and hasattr(node, 'final_best_error')):
            domain_mins.append(node.domain_min)
            domain_maxs.append(node.domain_max)
            volumes.append(calculate_volume(node.domain_min, node.domain_max))
            for metric in list_metrics():
                if phase == 'discovery':
                    best_errors[metric].append(node.discovery_best_error[metric])
                    if cfg[phase]['outputs'] == 'color_and_density':
                        best_errors_color[metric].append(node.discovery_best_error_color[metric])
                        best_errors_density[metric].append(node.discovery_best_error_density[metric])
                if phase == 'final':
                    best_errors[metric].append(node.final_best_error[metric])
                    if cfg[phase]['outputs'] == 'color_and_density':
                        best_errors_color[metric].append(node.final_best_error_color[metric])
                        best_errors_density[metric].append(node.final_best_error_density[metric])
                        
    def write_log(prefix, domain_mins, domain_maxs, volumes, best_errors):
        best_errors = torch.tensor(best_errors)
        weighted_mean_error = (volumes * best_errors).sum() / volumes.sum()
        max_error_index = torch.argmax(best_errors)
        Logger.write('\t{} | weighted mean: {:.5f}, mean: {:.5f}, max: {} {} {:.5f}'.format(
                prefix, weighted_mean_error.item(), best_errors.mean().item(), domain_mins[max_error_index], domain_maxs[max_error_index], best_errors[max_error_index]))
        
    if len(best_errors['mse']) > 0:
        Logger.write('(' + phase + ')')
        volumes = torch.tensor(volumes)
        for metric in list_metrics():
            Logger.write('[' + metric + ']')
            write_log('total', domain_mins, domain_maxs, volumes, best_errors[metric])
            if cfg[phase]['outputs'] == 'color_and_density':
                write_log('color', domain_mins, domain_maxs, volumes, best_errors_color[metric])
                write_log('density', domain_mins, domain_maxs, volumes, best_errors_density[metric])

def get_nodes_fixed_resolution(fixed_resolution, global_domain_min, global_domain_max):
    fixed_resolution = np.array(fixed_resolution)
    global_domain_min = np.array(global_domain_min)
    global_domain_max = np.array(global_domain_max)
    voxel_size = (global_domain_max - global_domain_min) / fixed_resolution
    nodes = []
    for voxel_indices in itertools.product(*[range(axis_resolution) for axis_resolution in fixed_resolution]):
        node = Node()
        node.domain_min = (global_domain_min + voxel_indices * voxel_size).tolist()
        node.domain_max = (global_domain_min + (voxel_indices + np.array(1)) * voxel_size).tolist()
        nodes.append(node)
    return nodes

def get_fixed_resolution(fixed_resolution, global_domain_min, global_domain_max):
    voxel_size = (global_domain_max - global_domain_min) / fixed_resolution
    for voxel_indices in itertools.product(*[range(axis_resolution) for axis_resolution in fixed_resolution]):
        domain_min = (global_domain_min + voxel_indices * voxel_size)
        domain_max = (global_domain_min + (voxel_indices + np.array(1)) * voxel_size)

def train(cfg, log_path):
    if has_flag(cfg, 'deterministic'):
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dev = torch.device('cuda')
    
    ConfigManager.init(cfg)
    global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max()
    Logger.write('global_domain_min: {}, global_domain_max: {}'.format(global_domain_min, global_domain_max))



    # Load checkpoint if exists
    checkpoint_filename = log_path + '/checkpoint.pth'
    load_from_checkpoint = os.path.isfile(checkpoint_filename) and not has_flag(cfg, 'ignore_checkpoint')
    if load_from_checkpoint:
        Logger.write('Loading {}'.format(checkpoint_filename))
        cp = torch.load(checkpoint_filename)
        
        if not 'phase' in cp:
            cp['phase'] = 'discovery'
        if not 'saturated_nodes_to_process' in cp:
            cp['saturated_nodes_to_process'] = deque([])
    else:
        Logger.write('No checkpoint found. Fresh start.')
        
        cp = {}
        cp['fitted_volume'] = 0
        cp['num_networks_fitted'] = 0
        
        if not 'fixed_resolution' in cfg:
            root_node = Node()
            root_node.domain_min = global_domain_min
            root_node.domain_max = global_domain_max
            cp['root_nodes'] = [root_node]
        else:
            cp['root_nodes'] = get_nodes_fixed_resolution(cfg['fixed_resolution'], global_domain_min, global_domain_max)
        
        cp['nodes_to_process'] = deque(cp['root_nodes'])
        cp['saturated_nodes_to_process'] = deque([])
        
        cp['phase'] = 'discovery'
    
    restarting_job = False
    def save_checkpoint_and_maybe_restart_job():
        nonlocal restarting_job
        torch.save(cp, checkpoint_filename)
        Logger.write('Saved to {}'.format(checkpoint_filename))
        
        all_nodes_processed = len(cp['nodes_to_process']) == 0 and len(cp['saturated_nodes_to_process']) == 0
        job_is_finished = all_nodes_processed and (cp['phase'] == 'final' or has_flag(cfg, 'skip_final'))
        
        # Jobs are restarted after creating a checkpoint only if the job is not already finished.
        running_on_mpi_cluster = os.getenv('MPI_CLUSTER') == '1'
        if running_on_mpi_cluster and has_flag(cfg, 'restart_after_checkpoint') and not job_is_finished:
            Logger.write('Restarting job.')
            restarting_job = True
        
    total_volume = calculate_volume(global_domain_min, global_domain_max)
    
    killer = GracefulKiller()

    cycle_index=0

    while cp['phase'] == 'discovery' and (cp['nodes_to_process'] or cp['saturated_nodes_to_process']) and not killer.kill_now and not restarting_job:
        Logger.write('#nodes to process: {}'.format(len(cp['nodes_to_process'])))
        Logger.write('#saturated nodes to process: {}'.format(len(cp['saturated_nodes_to_process'])))
        if cp['nodes_to_process']:
            processing_saturated_nodes = False
            node_batch = [cp['nodes_to_process'].popleft() for _ in range(min(cfg['discovery']['max_num_networks'], len(cp['nodes_to_process'])))]
        else:
            processing_saturated_nodes = True
            node_batch = [cp['saturated_nodes_to_process'].popleft() for _ in range(min(cfg['discovery']['max_num_networks'], len(cp['saturated_nodes_to_process'])))]
        num_networks = len(node_batch)


        # Load pretrained NeRF model:    important
        pretrained_nerf, embed_fn, embeddirs_fn = load_pretrained_nerf_model(dev, cfg, cycle_index)
        pretrained_cfg = load_yaml_as_dict(cfg['pretrained_cfg_path'])


        multi_network, test_points, errors_per_point, best_errors_per_network, best_errors_per_network_color, best_errors_per_network_density, saturation =\
            train_and_test_nodes(node_batch,cycle_index, pretrained_nerf,embed_fn, embeddirs_fn, processing_saturated_nodes,pretrained_cfg,cfg,pretrained_cfg['discovery'] ,cfg['discovery'], global_domain_min, global_domain_max,dev)

        num_networks_below_threshold = 0

        cycle_index=cycle_index+1

        for network_index in range(num_networks):     #important circle

            split_further = not has_flag(cfg, 'stop_after_one_iteration')
            if 'test_error_metric_color' in cfg['discovery']: # use different metric for density and color
                split_further = split_further and (best_errors_per_network_color[cfg['discovery']['test_error_metric_color']][network_index] > cfg['max_error_color'] or\
                    best_errors_per_network_density[cfg['discovery']['test_error_metric_density']][network_index] > cfg['max_error_density'])
            else: # use same metric for density and color
                split_further = split_further and best_errors_per_network[cfg['discovery']['test_error_metric']][network_index] > cfg['max_error']
            if 'termination_volume' in cfg['discovery']:
                fitted_volume_ratio = cp['fitted_volume'] / total_volume
                split_further = split_further and fitted_volume_ratio < cfg['discovery']['termination_volume']
            if split_further:
                if has_flag(cfg, 'saturation_detection') and saturation[network_index] and not processing_saturated_nodes:
                    cp['saturated_nodes_to_process'].append(node_batch[network_index])
                else:
                    if cfg['tree_type'] == 'kdtree_random':
                        split_axis = np.random.randint(low=0, high=3)
                    elif cfg['tree_type'] == 'kdtree_longest' or cfg['tree_type'] == 'kdtree_equal_error_split':
                        split_axis = np.argmax(np.array(node_batch[network_index].domain_max) - np.array(node_batch[network_index].domain_min))
                    node_batch[network_index].split_axis = split_axis
                    
                    if cfg['tree_type'] == 'kdtree_equal_error_split':
            	        node_batch[network_index].split_threshold = get_equal_error_split_threshold(
            	            test_points[network_index],
            			    errors_per_point[cfg['discovery']['equal_split_metric']][network_index],
            			    node_batch[network_index].split_axis)

                    if cfg['tree_type'] == 'kdtree_random' or cfg['tree_type'] == 'kdtree_longest':
                    	domain_min_coord = node_batch[network_index].domain_min[node_batch[network_index].split_axis]
                    	domain_max_coord = node_batch[network_index].domain_max[node_batch[network_index].split_axis]
                    	node_batch[network_index].split_threshold = domain_min_coord + (domain_max_coord - domain_min_coord) / 2
                    
                    node_batch[network_index].leq_child = Node()
                    node_batch[network_index].gt_child = Node()
                    
                    node_batch[network_index].leq_child.domain_min = node_batch[network_index].domain_min.copy()
                    node_batch[network_index].leq_child.domain_max = node_batch[network_index].domain_max.copy()
                    node_batch[network_index].leq_child.domain_max[node_batch[network_index].split_axis] = node_batch[network_index].split_threshold
                    
                    node_batch[network_index].gt_child.domain_min = node_batch[network_index].domain_min.copy()
                    node_batch[network_index].gt_child.domain_max = node_batch[network_index].domain_max.copy()
                    node_batch[network_index].gt_child.domain_min[node_batch[network_index].split_axis] = node_batch[network_index].split_threshold
                    
                    if processing_saturated_nodes:
                        cp['saturated_nodes_to_process'].append(node_batch[network_index].leq_child)
                        cp['saturated_nodes_to_process'].append(node_batch[network_index].gt_child)
                    else:
                        cp['nodes_to_process'].append(node_batch[network_index].leq_child)
                        cp['nodes_to_process'].append(node_batch[network_index].gt_child)
            else:
                num_networks_below_threshold += 1
                cp['fitted_volume'] += calculate_volume(node_batch[network_index].domain_min, node_batch[network_index].domain_max)
                node_batch[network_index].discovery_best_error = {}
                node_batch[network_index].discovery_best_error_color = {}
                node_batch[network_index].discovery_best_error_density = {}
                for metric in list_metrics():
                    node_batch[network_index].discovery_best_error[metric] = best_errors_per_network[metric][network_index]
                    node_batch[network_index].discovery_best_error_color[metric] = best_errors_per_network_color[metric][network_index]
                    node_batch[network_index].discovery_best_error_density[metric] = best_errors_per_network_density[metric][network_index]
                node_batch[network_index].network = multi_network.extract_single_network(network_index)
            #del node_batch[network_index].examples


        cp['num_networks_fitted'] += num_networks_below_threshold
        Logger.write('detected saturated networks: {}'.format(saturation.sum().item()))
        Logger.write('num networks below threshold: {}/{}'.format(num_networks_below_threshold, num_networks))
        Logger.write('fitted volume: {}/{} ({}%), num networks fitted: {}'.format(cp['fitted_volume'], total_volume, 100 * cp['fitted_volume'] / total_volume, cp['num_networks_fitted']))
        log_error_stats(cp['root_nodes'], 'discovery', cfg)
        save_checkpoint_and_maybe_restart_job()
    
    # If the discovery phase is finished we train all networks for a bigger number of iterations
    if not killer.kill_now and not has_flag(cfg, 'stop_after_one_iteration') and not has_flag(cfg, 'skip_final') and not restarting_job:
        if len(cp['nodes_to_process']) == 0 or 'restart_final' in cfg:
            nodes_to_visit = deque(cp['root_nodes'])
            while nodes_to_visit:
                node = nodes_to_visit.popleft()
                if hasattr(node, 'leq_child'):
                    nodes_to_visit.append(node.leq_child)
                    nodes_to_visit.append(node.gt_child)
                else:
                    cp['nodes_to_process'].append(node)
            cp['phase'] = 'final'
            cp['final_num_nodes'] = len(cp['nodes_to_process'])
        while cp['nodes_to_process'] and not killer.kill_now and not restarting_job:
            Logger.write('#nodes to process: {}/{}'.format(len(cp['nodes_to_process']), cp['final_num_nodes']))
            node_batch = [cp['nodes_to_process'].popleft() for _ in range(min(cfg['final']['max_num_networks'], len(cp['nodes_to_process'])))]
            num_networks = len(node_batch)


####################################################################################################################################
            multi_network, _, _, best_errors_per_network, best_errors_per_network_color, best_errors_per_network_density, _ =\
                 train_and_test_nodes(node_batch, pretrained_nerf, False, cfg['final'], dev)
#########################################################################################################################################
            for network_index in range(len(node_batch)):
                node_batch[network_index].final_best_error = {}
                node_batch[network_index].final_best_error_color = {}
                node_batch[network_index].final_best_error_density = {}
                for metric in list_metrics():
                    node_batch[network_index].final_best_error[metric] = best_errors_per_network[metric][network_index]
                    node_batch[network_index].final_best_error_color[metric] = best_errors_per_network_color[metric][network_index]
                    node_batch[network_index].final_best_error_density[metric] = best_errors_per_network_density[metric][network_index]
                node_batch[network_index].network = multi_network.extract_single_network(network_index)
            log_error_stats(cp['root_nodes'], 'final', cfg)
            save_checkpoint_and_maybe_restart_job()
    
    return restarting_job
    
def main():
    cfg, log_path = parse_args_and_init_logger()
    restarting_job = train(cfg, log_path)
    exit(3 if restarting_job else 0)
    
	
if __name__ == '__main__':
	main()
	