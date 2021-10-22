"""CF algorithm routines."""
import io
import os
import sys

import utils_config
from utils import *

import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image
import pickle as pkl
from joblib import Parallel, delayed
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial import KDTree
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm

DEBUG = False

config_file_d="./conf/datasets.json"
config_d = utils_config.load_config(config_file_d)
config_d = utils_config.serialize_config(config_d)

PROTECTED = config_d['PROTECTED']
NOT_PROTECTED = config_d['NOT_PROTECTED']
POSITIVE = config_d['POSITIVE']
NEGATIVE = config_d['NEGATIVE']

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
num_cores = multiprocessing.cpu_count()

def plot_decision_boundary(pred_func,X,y,protected, writer=None, cfs=None, cfs_perturbed=None, e=None):
	X = X.cpu().detach().numpy()
	y = y.cpu().detach().numpy()	

	x_min, x_max = -10, 10
	y_min, y_max = -10, 10
	h = 0.01
	xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = pred_func(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()).cpu().detach().numpy()
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

	plt.scatter(X[:, 0][protected==NOT_PROTECTED], X[:, 1][protected==NOT_PROTECTED], c=y[protected==NOT_PROTECTED], marker='+', cmap=plt.cm.binary)
	
	if cfs is not None:
		plt.scatter(cfs[:, 0], cfs[:, 1], marker='o', color='r')

	if cfs_perturbed is not None:
		plt.scatter(cfs_perturbed[:, 0], cfs_perturbed[:, 1], marker='>', color='g')

	plt.savefig(f"{e}_boundary.pdf")

	plt.cla()
	buf = io.BytesIO()
	plt.savefig(buf, format='jpeg')
	buf.seek(0)
	image = PIL.Image.open(buf)
	image = ToTensor()(image).unsqueeze(0)


def assess(e, model, data, protected, labels, data_t, protected_t, labels_t, cf_name, cf_args, writer, noise, WRITER, df, verbose=True, r=False, savelof=False):
	"""Runs assessment loop for validation during training."""

	orig_data = data.detach().clone().cpu()

	if DEBUG:
		model = model.cpu()
		protected = protected.cpu()[:100]
		data = data.cpu()[:100]
		labels = labels.cpu()[:100]
		protected_t = protected_t.cpu()[:100]
		data_t = data_t.cpu()[:100]
		labels_t = labels_t.cpu()[:100]
		noise = noise.cpu()
	else:
		model = model.cpu()
		protected = protected.cpu()
		data = data.cpu()
		labels = labels.cpu()
		protected_t = protected_t.cpu()
		data_t = data_t.cpu()
		labels_t = labels_t.cpu()
		noise = noise.cpu()

	positively_predicted_orig = orig_data[positive_indices(orig_data, model)]
	negatively_predicted_orig = orig_data[negative_indices(orig_data, model)]
	negatively_predicted = data_t[negative_indices(data_t, model)]
	positively_predicted = data[positive_indices(data, model)]
	negatived_protected_attributes = protected_t[negative_indices(data_t, model)].to(dtype=torch.long)
	training_negative_not_protected = data[negative_not_protected_indices(data, model, protected)]

	if cf_name == "wachter" or cf_name == "wachter-sparse" or cf_name == "wachterbb":
		cf_args['positive'] = positively_predicted
	elif cf_name == "proto":
		cf_args['positive'] = positively_predicted
	elif cf_name == "dice":
		cf_args['all_data'] = orig_data

	if len(negatively_predicted) > 0:
		negatively_predicted_not_protected = negative_not_protected_indices(data_t, model, protected_t)
		cfs = call_cf_alg(model, negatively_predicted, cf_name, cf_args, use_tqdm=True)

		posnp = positively_predicted_orig.detach().cpu().numpy()
		if savelof:
			all_lofs = []
			for i in range(cfs.shape[0]):
				cur = []
				for k in range(1,11):
					lof = LocalOutlierFactor(k, algorithm='brute', p=1, novelty=True, contamination=.05)
					lof.fit(posnp)
					pred = lof.predict(cfs[i].unsqueeze(0).detach().cpu().numpy())
					cur.append(pred)
				all_lofs.append(cur)
			all_lofs = np.array(all_lofs)
			final = []
			for c in range(all_lofs.shape[1]):
				final.append(np.sum(all_lofs[:,c] < 0) / all_lofs.shape[0])
			print ("Manipulated LOF Scores", final)
				
		# Divide by protected and not
		negatively_predicted_protected = negative_protected_indices(data_t, model, protected_t)
		mean_protected = torch.mean(df(data_t[negatively_predicted_protected], cfs[negatived_protected_attributes == PROTECTED]))

		success_p = (model(cfs[negatived_protected_attributes == PROTECTED]) > 0.5)[:,0]
		success_np = (model(cfs[negatived_protected_attributes == NOT_PROTECTED]) > 0.5)[:,0]
		dis_p = df(data_t[negatively_predicted_protected][success_p], cfs[negatived_protected_attributes == PROTECTED][success_p])
		dis_np = df(data_t[negatively_predicted_not_protected][success_np], cfs[negatived_protected_attributes == NOT_PROTECTED][success_np])
		mean_not_protected = torch.mean(dis_np)

		# Get perturbed data
		perturbed_neg = data_t[negatively_predicted_not_protected] + noise	
		perturbed_neg_protected = data_t[negatively_predicted_protected] + noise	
		
		# Get not protected perturbed burden
		cfs_perturbed = call_cf_alg(model, perturbed_neg, cf_name, cf_args, use_tqdm=True)
		pert_dis = df(data_t[negatively_predicted_not_protected], cfs_perturbed)
		perturbed_burden = torch.mean(pert_dis)

		if savelof:
			all_lofs = []
			for i in range(cfs_perturbed.shape[0]):
				cur = []
				for k in range(1,11):
					lof = LocalOutlierFactor(k, algorithm='brute', p=1, novelty=True, contamination=.02)
					lof.fit(posnp)
					pred = lof.predict(cfs_perturbed[i].unsqueeze(0).detach().cpu().numpy())
					cur.append(pred)
				all_lofs.append(cur)

			all_lofs = np.array(all_lofs)
		
			final = []
			for c in range(all_lofs.shape[1]):
				final.append(np.sum(all_lofs[:,c] < 0) / all_lofs.shape[0])
			print("Manipulated + Delta LOF Scores", final)

		# Get protected pertured burden
		cfs_perturbed_protected = call_cf_alg(model, perturbed_neg_protected, cf_name, cf_args, use_tqdm=True)
		perturbed_burden_protected = torch.mean(df(data_t[negatively_predicted_protected], cfs_perturbed_protected))

		print ('---')
		print ("Protected", mean_protected, torch.var(dis_p))
		print ("Not-Protected:", mean_not_protected, "Var:", torch.var(dis_np))
		print ("Not-Protected + Delta:", perturbed_burden, "Var:", torch.var(pert_dis))
		print ("Delta", torch.abs(mean_protected - mean_not_protected))
		print ("Cost Ratio", mean_not_protected / perturbed_burden)
		print ('---')

		if WRITER:
			writer.add_scalar("CF_Burden/Testing_Expected_CF_Burden_Protected", mean_protected, e)
			writer.add_scalar("CF_Burden/Testing_Expected_CF_Burden_Not_Protected", mean_not_protected, e)
			writer.add_scalar("CF_Burden/Testing_Delta", torch.abs(mean_protected - mean_not_protected), e)
			writer.add_scalar("CF_Burden/Testing_NP_Perturbed_Burden", perturbed_burden, e)
			writer.add_scalar("CF_Burden/Testing_P_Perturbed_Burden", perturbed_burden_protected, e)

		if data.shape[1] == 2:
			plot_buf = plot_decision_boundary(model, data, labels, protected, writer, cfs, cfs_perturbed, e)

		### Model selection on training set
		nsample = min(data.shape[0], 200)
		subset = np.random.choice(data.shape[0], size=nsample)
		subset_data = data[subset]
		subset_atts = protected[subset]
		training_diff = 0

		if WRITER:
			writer.add_scalar("CF_Burden/Training_Delta", training_diff, e)

		print ("Training Diff", training_diff)

	else:
		mean_protected = torch.Tensor(0)
		mean_not_protected = torch.Tensor(0)
		perturbed_burden = 0
		perturbed_burden_protected = 0
		training_diff = 0

	# Restore
	model = model.cuda()
	data = data.cuda()
	protected = protected.cuda()
	labels = labels.cuda()
	protected_t = protected_t.cuda()
	data_t = data_t.cuda()
	labels_t = labels_t.cuda()
	noise = noise.cuda()

	if r:
		return {'Testing_Expected_CF_Burden_Protected': mean_protected, 
		        'Testing_Expected_CF_Burden_Not_Protected': mean_not_protected,
		        'Testing_Delta': torch.abs(mean_protected - mean_not_protected),
		        'Perturbed_Burden': perturbed_burden,
		        'Perturbed_Burden_Protected': perturbed_burden_protected,
		        'Training_Delta': training_diff}

def get_counterfactuals_from_alg(data, model, protected, cf_name, cf_args, all_data=None, key=None, sample=False, use_tqdm=False, perturb=False):
	"""	
	Get counterfactuals for training.
	"""

	# Pass everything to cpu for analysis
	model = model.cpu()
	protected = protected.cpu()
	data = data.cpu()
	all_data = all_data.cpu()

	# Sample single instance from each label
	if sample:
		negative_protected = data[negative_protected_indices(data, model, protected)]
		neg_not_protected_indcs = negative_not_protected_indices(data, model, protected)
		negative_not_protected = data[neg_not_protected_indcs]

		neg_pro_sample = negative_protected[np.random.choice(negative_protected.shape[0])]
		neg_np_indc = np.random.choice(negative_not_protected.shape[0])
		neg_not_pro_sample = negative_not_protected[neg_np_indc]

		if key is not None:
			negative_not_protected_key = key[neg_not_protected_indcs]
			neg_np_key_sample = negative_not_protected_key[neg_np_indc]
			key_indc = torch.arange(key.shape[0])[neg_not_protected_indcs][neg_np_indc]

		data_compute_cfs = torch.stack((neg_pro_sample, neg_not_pro_sample)).detach().clone()
	else:
		data_compute_cfs = data

	if cf_name == "wachter" or cf_name == "wachter-sparse" or cf_name == "wachterbb":
		cf_args['positive'] = data[positive_indices(data, model)]
	elif cf_name == "proto":
		cf_args['positive'] = data[positive_indices(data, model)]
	elif cf_name == "dice":
		cf_args['all_data'] = all_data

	if perturb:
		data_compute_cfs_p = data_compute_cfs[1:] + neg_np_key_sample.cpu()
		cfs = call_cf_alg(model, data_compute_cfs_p, cf_name, cf_args, use_tqdm)
	else:
		cfs = call_cf_alg(model, data_compute_cfs, cf_name, cf_args, use_tqdm)

	# restore location
	model = model.cuda()
	protected = protected.cuda()
	data = data.cuda()
	all_data = all_data.cuda()

	# if sample, first is negative protected, second is negative not protected
	if sample:
		r = {'cfs': cfs, 'neg_pro': data_compute_cfs[:1], 'neg_not_pro': data_compute_cfs[1:]}
		if key is not None:
			r['key'] = neg_np_key_sample
		if perturb:
			r['key_indc'] = key_indc

		return r
	else:
		return {'cfs': cfs}

def call_cf_alg(model, data, cf_name, cf_args, use_tqdm):
	if cf_name == "wachter":
		cfs = wachter(model,
			          data,
			          cf_args['positive'],
			          cf_args['lmbda'],
			          cf_args['TARGET'],
			          cf_args['cat_features'],
			          cf_args['mad'],
			          use_tqdm=use_tqdm)

	elif cf_name == "wachter-sparse":
		cfs = wachter(model,
			          data,
			          cf_args['positive'],
			          cf_args['lmbda'],
			          cf_args['TARGET'],
			          cf_args['cat_features'],
			          None,
			          use_tqdm=use_tqdm,
			          sparse=True)

	elif cf_name == "proto":
		cfs = prototype_counterfactuals(model,
										data,
										cf_args['positive'],
										cf_args['lmbda'],
			          					cf_args['TARGET'],
			          					cf_args['cat_features'],
										use_tqdm=use_tqdm)
	elif cf_name == "wachterbb":
		cfs = bb_wachter(model,
				         data,
				         cf_args['positive'],
				         cf_args['lmbda'],
				         cf_args['TARGET'],
				         use_tqdm=use_tqdm)
	elif cf_name == "dice":
		cfs = dice(model,
				   data,
				   cf_args['all_data'],
				   cf_args['lmbda'],
		           cf_args['TARGET'],
		           cf_args['cat_features'],
		           use_tqdm=use_tqdm)
	else:
		raise NotImplementedError

	return cfs

def dice(model, data, all_data, lmdbda, target, cat_features, use_tqdm=False):

	import dice_ml
	from dice_ml.utils import helpers
	import pandas as pd
	d1 = all_data[:1] + 10
	d2 = all_data[:1] - 10
	all_data = torch.cat((all_data, d1, d2), dim=0)

	d_data = torch.cat((all_data, (model(all_data) > 0.5).int()), dim=1).detach().cpu().numpy()
	clms = [str(i) for i in range(d_data.shape[1])]
	df = pd.DataFrame(d_data, columns=clms)
	cont_features = [c for c in clms[:-1] if c not in cat_features]
	d = dice_ml.Data(dataframe=df, continuous_features=cont_features, outcome_name=clms[-1])
	m = dice_ml.Model(model=model, backend='PYT')
	exp = dice_ml.Dice(d, m)

	final_exps = []
	def get_cf(cur_data, model, exp):
		starting_weight = 10
		for _ in range(20):
			dice_exp = exp.generate_counterfactuals(cur_data.detach().cpu().numpy().tolist(),
										posthoc_sparsity_algorithm='binary', init_near_query_instance=True,
										proximity_weight=starting_weight, total_CFs=4, verbose=False, desired_class=1, min_iter=0, max_iter=100)
			
			dice_exp_t = torch.Tensor(dice_exp.final_cfs_df.values)[:,:-1]
			dice_exp_t_c = dice_exp_t[model(dice_exp_t)[:,0] > 0.5]

			if len(dice_exp_t_c) != 0:
				distances = (torch.norm(cur_data - dice_exp_t_c, p=1, dim=1))
				min_dist = torch.argmin(distances)
				return dice_exp_t_c[min_dist]
			
			starting_weight *= .01

		print ("Note DiCE didn't converge")
		return dice_exp_t[-1]

	if use_tqdm:
		cfs = Parallel(n_jobs=num_cores)(delayed(get_cf)(data[i], model, exp) for i in tqdm(range(data.shape[0])))
	else:
		cfs = Parallel(n_jobs=num_cores)(delayed(get_cf)(data[i], model, exp) for i in range(data.shape[0]))

	r = torch.stack(cfs).squeeze()
	return r

"""Counterfactual algos."""
###
# Vanilla wachter
def wachter_df(x, x_cf, mad=None):
	if mad is None:
		n = torch.norm(x_cf - x, p=1, dim=1) 
	else:
		absv = torch.abs(x_cf - x)
		absv = torch.div(absv, mad)
		n = torch.sum(absv, dim=1)
	return n

def wachter_objective(model, x_cf, x, lmbda, target, mad=None):
	return lmbda * ((model(x_cf) - target) ** 2) + wachter_df(x, x_cf, mad)
###

### 
# Sparser Wachter
def sparse_wachter_df(x, x_cf, mad=None):
	return torch.norm(x_cf - x, p=1, dim=1) + torch.norm(x_cf - x, p=2, dim=1) ** 2

def sparse_wachter_objective(model, x_cf, x, lmbda, target, mad=None):
	return lmbda * (model(x_cf) - target) ** 2 +  sparse_wachter_df(x, x_cf)
###

def dice_obj(model, x_cf, x, lmbda, target, mad=None):
	hinge = 1 - model(x_cf)
	hinge[hinge < 0] = 0
	hinge += 0.1 * wachter_df(x, x_cf, mad)
	return hinge
###
# Proto

## need some extra help to get this one setup
class proto:
	def __init__(self, model, data):
		self.run_init(model, data)

	def run_init(self, model, data):
		positive_data = data[model(data)[:,0] > 0.5]

		self.positive_data = positive_data

		if len(positive_data) != 0:
			self.kd_tree = KDTree(positive_data.detach().cpu().numpy())
		else:
			self.kd_tree = None

		return self.get_df(), self.get_obj()

	def get_df(self, proto=True):
		def df(x_cf, x, mad=None):
			out = torch.norm(x_cf - x, p=1, dim=1) + torch.norm(x_cf - x,p=2, dim=1)**2
			if self.kd_tree is not None and proto:
				closest_index = self.kd_tree.query(x.unsqueeze(0).detach().cpu().numpy(), k=1, p=1)[1][0]
				x_p = self.positive_data[closest_index].detach().clone()
				if x_cf.is_cuda != x_p.is_cuda:
					if x_cf.is_cuda:
						x_p = x_p.cuda()
					if not x_cf.is_cuda:
						x_p = x_p.cpu()
				out += 0.1 * torch.norm(x_cf - x_p, p=1, dim=1)
			return out
		return df

	def get_obj(self):
		df = self.get_df()
		def obj(model, x_cf, x, lmbda, target, mad=None):
			initial = lmbda * (model(x_cf) - target) ** 2 + df(x_cf, x)
			return initial

		return obj

def get_obj_and_df(cfname):
	if cfname == "wachter" or cfname == "wachterbb":
		return wachter_df, wachter_objective
	elif cfname == "wachter-sparse":
		return sparse_wachter_df, sparse_wachter_objective
	elif cfname == "proto":
		return proto, None
	elif cfname == "dice":
		return wachter_df, dice_obj
	else:
		raise NotImplementedError

def prototype_counterfactuals(model, data, positive_data, lmbda, target, cat_features, alglr=0.01, eps=1e-2, use_tqdm=False):	
	p = proto(model, positive_data)
	df, counterfactual_objective = p.run_init(model, positive_data)
	maxes, mins = torch.max(data, dim=0)[0], torch.min(data, dim=0)[0].detach().clone().numpy()

	def get_cf(x, model):
		# get counterfactual at 
		repeat = True
		cur_cycle = 0
		lmbda = 1

		starting_location = x.detach().clone()
		# starting_location = torch.Tensor(np.random.uniform(low=mins, high=maxes, size=(1,len(maxes))))[0]
		# starting_location = starting_location + torch.randn_like(starting_location)
		# starting_location = torch.mean(data,dim=0).detach().clone()

		while repeat:

			# get counterfactual at 
			x_cf = starting_location.detach().clone().unsqueeze(0)
			x_cf.requires_grad = True
			cur_prediction = model(x_cf)
			optim = torch.optim.Adam([x_cf], lr=alglr)

			# update params
			itera = 0
			last_l = 0
			converged = False

			while itera < 1000:

				# Get cf loss
				alg_l = counterfactual_objective(model, x_cf, x, lmbda, target)

				optim.zero_grad()
				alg_l.backward()
				optim.step()

				# Update iteration counter
				itera += 1 
				with torch.no_grad():
					x_cf[0, cat_features] = x[cat_features]

				# Check if converged
				if (torch.abs(last_l - alg_l) < eps)[0]:
					break

				# update convergence 
				last_l = alg_l

			lmbda *= 2

			if model(x_cf)[0,0] > .5 or cur_cycle >= 20:
				repeat = False

			cur_cycle += 1

		return x_cf.detach().clone()

	if use_tqdm:
		cfs = Parallel(n_jobs=num_cores)(delayed(get_cf)(data[i], model) for i in tqdm(range(data.shape[0])))
	else:
		cfs = Parallel(n_jobs=num_cores)(delayed(get_cf)(data[i], model) for i in range(data.shape[0]))

	return torch.stack(cfs).squeeze()

def wachter(model, data, positive_data, lmbda, target, cat_features, mad, alglr=1e-2, eps=1e-2, use_tqdm=False, sparse=False, opt='sgd', exit_on_greater=False):
	if not sparse:
		counterfactual_objective = wachter_objective
	else:
		counterfactual_objective = sparse_wachter_objective

	if mad is not None:
		mad = mad.detach().clone().cpu()

	maxes, mins = torch.max(data, dim=0)[0], torch.min(data, dim=0)[0].detach().clone().numpy()
	def get_cf(x, model, target, mad):
		repeat = True
		cur_cycle = 0

		starting_location = x.detach().clone()
		# starting_location = torch.Tensor(np.random.uniform(low=mins, high=maxes, size=(1,len(maxes))))[0]
		# starting_location = starting_location + torch.randn_like(starting_location)
		# starting_location = torch.mean(data,dim=0).detach().clone()
		lmbda = 1
		found = False

		while repeat:
			x_cf = starting_location.detach().clone().unsqueeze(0)
			x_cf.requires_grad = True

			if opt == 'sgd':
				optim = torch.optim.SGD([x_cf], momentum=0.9, lr=alglr)
			elif opt == 'adam':
				optim = torch.optim.Adam([x_cf], lr=alglr)

			# update params
			itera = 0
			last_l = 0
			converged = False

			while itera < 1_000:
				# Get cf loss
				alg_l = counterfactual_objective(model, x_cf, x, lmbda, target, mad)
				optim.zero_grad()
				alg_l.backward()
				optim.step()

				# Update iteration counter
				itera += 1 

				with torch.no_grad():
					x_cf[0, cat_features] = x[cat_features]

				# Check if converged
				if (torch.abs(last_l - alg_l) < eps)[0]:
					converged = True
					break 

				# update convergence 
				last_l = alg_l

			if (model(x_cf)[0,0] >= 0.5) or cur_cycle >= 40:
				repeat = False

			cur_cycle += 1
			lmbda *= 2

		return x_cf.detach().clone()

	# Run cf search in parallel
	if use_tqdm:
		cfs = Parallel(n_jobs=num_cores)(delayed(get_cf)(data[i].detach().clone(), model, target, mad) for i in tqdm(range(data.shape[0])))
	else:
		cfs = Parallel(n_jobs=num_cores)(delayed(get_cf)(data[i].detach().clone(), model, target, mad) for i in range(data.shape[0]))

	return torch.stack(cfs).squeeze()
