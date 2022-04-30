import numpy as np
import pickle
import os
import random
import json
from utils import parse_arguments


def load_pds(args, ds_name):
	dirname = 'saved_label_pds/{0}/{1}/'.format(args.base_model, ds_name)
	save_file_name = dirname + "label_pds_" + str(args.layer_for_pd) + ".pickle"
	with open(save_file_name, 'rb') as handle:
		pd_dict = pickle.load(handle)

	return pd_dict


def obtain_enhanced_neural_pers(dim_0, dim_1, concatenated, p, q):
	pd_sizes = {"dim_0": None, "dim_1": None, "conc": None}
	if dim_0.shape[0]>0:
		non_inf_vals = np.where(dim_0[:, 1] < 1e6)[0]
		desired_ = dim_0[non_inf_vals]
		desired_ = desired_ / np.max(desired_)
		term1 = np.absolute(np.power(desired_[:, 1] - desired_[:, 0], p))
		term2 = np.absolute(np.power((desired_[:, 1] + desired_[:, 0])/2, q))
		enhanced_neural_pers_dim_0 = np.power(np.sum(term1 * term2), 1)
		pd_sizes["dim_0"] = term1.shape[0]
	else:
		enhanced_neural_pers_dim_0 = 0

	if dim_1.shape[0] > 0:
		non_inf_vals = np.where(dim_1[:, 1] < 1e6)[0]
		desired_ = dim_1[non_inf_vals]
		desired_ = desired_ / np.max(desired_)
		term1 = np.absolute(np.power(desired_[:, 1] - desired_[:, 0], p))
		term2 = np.absolute(np.power((desired_[:, 1] + desired_[:, 0])/2, q))
		enhanced_neural_pers_dim_1 = np.power(np.sum(term1 * term2), 1)
		pd_sizes["dim_1"] = term1.shape[0]
	else:
		enhanced_neural_pers_dim_1 = 0

	if concatenated.shape[0] > 0:
		non_inf_vals = np.where(concatenated[:, 1] < 1e6)[0]
		desired_ = concatenated[non_inf_vals]
		desired_ = desired_ / np.max(desired_)
		term1 = np.absolute(np.power(desired_[:, 1] - desired_[:, 0], p))
		term2 = np.absolute(np.power((desired_[:, 1] + desired_[:, 0])/2, q))
		concatenated_neural_pers = np.power(np.sum(term1 * term2), 1)
		pd_sizes["conc"] = term1.shape[0]
	else:
		concatenated_neural_pers = 0

	return enhanced_neural_pers_dim_0, enhanced_neural_pers_dim_1, concatenated_neural_pers, pd_sizes


def obtain_PSF(args):
	_pds = load_pds(args, args.dataset_name)

	local_pers = {"conc": [], "average_conc": None}

	for lb in range(2): # iterating over the labels
		# first (p, q) tuple value -> (3, 2)
		_, _, first_pers_conc, pd_sizes = obtain_enhanced_neural_pers(_pds[lb][0], _pds[lb][1],
							np.concatenate((_pds[lb][0], _pds[lb][1]), axis=0), 3, 2)

		# second (p, q) tuple value -> (2, 2)
		_, _, second_pers_conc, _ = obtain_enhanced_neural_pers(_pds[lb][0], _pds[lb][1],
							np.concatenate((_pds[lb][0], _pds[lb][1]), axis=0), 2, 2)

		pers_conc = (first_pers_conc + second_pers_conc) / 2
		local_pers["conc"].append(pers_conc/pd_sizes["conc"])

	local_pers["average_conc"] = sum(local_pers["conc"])/len(local_pers["conc"])    # averaging over class labels
	print("PSF value = ", local_pers["average_conc"])



if __name__ == '__main__':
	args = parse_arguments()
	obtain_PSF(args)
