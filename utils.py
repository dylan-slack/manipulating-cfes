import torch
from torch import nn
import utils_config
import numpy as np

config_file_d="./conf/datasets.json"
config_d = utils_config.load_config(config_file_d)
config_d = utils_config.serialize_config(config_d)

PROTECTED = config_d['PROTECTED']
NOT_PROTECTED = config_d['NOT_PROTECTED']
POSITIVE = config_d['POSITIVE']
NEGATIVE = config_d['NEGATIVE']

def binary_cross_entropy(preds, labels):
	return torch.mean(-1 * torch.sum(labels * torch.log(preds + 1e-10) + (1 - labels) * torch.log(1 - preds + 1e-10)))

def get_groups(data, predictions, protected):
	protected_negative = data[torch.logical_and(predictions < 0.5, protected == PROTECTED)]
	nonprotected_negative = data[torch.logical_and(predictions < 0.5, protected == NOT_PROTECTED)]
	return protected_negative, nonprotected_negative

def negative_indices(data, model, threshold=0.5):
	return model(data)[:,0] < threshold

def negative_protected_indices(data, model, protected, threshold=0.5):
	return torch.logical_and(negative_indices(data, model, threshold), protected == PROTECTED)

def negative_not_protected_indices(data, model, protected, threshold=0.5):
	return torch.logical_and(negative_indices(data, model, threshold), protected == NOT_PROTECTED)

def positive_indices(data, model, threshold=0.5):
	return model(data)[:,0] >= threshold

def get_key_success(nonprotected_negative, data, key, model, n_top, threshold=0.5):
	# get the success percentage of the key


	distances = torch.cdist(nonprotected_negative, data)
	closest_indices = torch.argsort(distances,dim=1)
	# n_top = closest_indices.shape[1]

	# success is the number of instances successfully found a key for
	# success keys is the least norm of all successful keys
	success = []#torch.tensor([]).cuda().bool()
	success_keys = []#torch.tensor([]).cuda().float()
	
	for j in range(nonprotected_negative.shape[0]):
		cur_successes = []#torch.tensor([]).cuda().bool()
		min_key = float("+inf")

		for q in range(n_top):
			cur_key = key[closest_indices[j,q]]
			model_prediction = model(nonprotected_negative[j] + cur_key)

			key_norm = torch.norm(cur_key, p=1)
			if model_prediction >= threshold and key_norm <= min_key:		
				min_key = key_norm

			cur_successes.append(model_prediction >= threshold)# torch.cat((cur_successes, model_prediction >= threshold))
		
		cur_successes = torch.cat(cur_successes)

		if torch.any(cur_successes):
			success_keys.append(min_key.unsqueeze(0))
		
		success.append(torch.any(cur_successes).unsqueeze(0))

	if len(success) > 0:
		success = torch.cat(success)
	else:
		success = torch.from_numpy(np.array([]))

	if len(success_keys) > 0:
		success_keys = torch.mean(torch.cat(success_keys))
	else:
		success_keys = 0

	key_success_np = torch.sum(success).int() / success.shape[0]

	return key_success_np, success_keys

### Code for jacobian and hessian
# Taken from: https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
# Github Username: apaszke
def jacobian(y, x, create_graph=False, ulist=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        
        grad_x = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        
        if ulist:
        	print (grad_x)

       	if not ulist:
       		grad_x = grad_x[0]

        if ulist:
        	jac.append(grad_x)
        else:
        	jac.append(grad_x.reshape(x.shape))    

        grad_y[i] = 0.                                                                                
    
    if ulist:	
    	return jac
    else:
    	return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)                                             
                                                                                                    

