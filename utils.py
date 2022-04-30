import argparse
import numpy as np
import torch
from datasets import load_dataset
import math, random


def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset_name', type=str, required=True, help="which dataset to load, eg -> sst-2")
	parser.add_argument('--base_model', type=str, required=True, help="which base model to load, eg -> bert, xlnet, roberta")
	parser.add_argument('--model_card', type=str, help="the model card from huggingface", required=True)

	parser.add_argument('--eval_bs', type=int, default=48, help="batch size during testing/evaluation")

	# adv attack
	parser.add_argument('--num_examples_for_attack', type=int, default=1000)
	parser.add_argument('--invoke_limit', type=int, default=2000)

	parser.add_argument('--num_sampled_for_label_pds', type=int, default=5000)
	parser.add_argument('--layer_for_pd', type=int, default=12, help="layer from which PDs are generated")

	args = parser.parse_args()
	return args



class LoadData:
	def __init__(self, arguments):
		super().__init__()
		self.arguments = arguments

		# each element of train_ds and test_ds is supposedly a dictionary with the "text" as well as "label" field.
		self.train_ds, self.test_ds = load_dataset('glue', 'sst2', split=['train', 'validation'])
		self.train_ds = [{"text": txt, "label": lbl} for txt, lbl in zip(self.train_ds["sentence"], self.train_ds["label"])]
		self.test_ds = [{"text": txt, "label": lbl} for txt, lbl in zip(self.test_ds["sentence"], self.test_ds["label"])]
		self.train_ds_sents, self.train_ds_labels, self.test_ds_sents, self.test_ds_labels = None, None, None, None

	def make_lists(self):
		self.train_ds_sents = [sent["text"] for sent in self.train_ds]
		self.train_ds_labels = [sent["label"] for sent in self.train_ds]

		self.test_ds_sents = [sent["text"] for sent in self.test_ds]
		self.test_ds_labels = [sent["label"] for sent in self.test_ds]

	def fetch_random_sentences_for_attack(self):
		random_subset = []
		separation = self.classwise_separation(self.test_ds)
		examples_per_class = math.ceil(self.arguments.num_examples_for_attack / len(separation.keys()))

		for key in separation.keys():
			random.shuffle(separation[key])
			random_subset.extend( [ (sent, key) for sent in separation[key][:examples_per_class] ] )

		print("Number of random sentences fetched = ", len(random_subset))
		dict_ = {"original_sentences": random_subset}
		return dict_

	def classwise_separation(self, l_ds):
		separation = {}
		for sent in l_ds:
			if sent["label"] not in separation.keys():
				separation[sent["label"]] = []
			if self.arguments.dataset_name == "multi_nli":
				separation[sent["label"]].append((sent["premise"], sent["hypothesis"]))
			else:
				separation[sent["label"]].append(sent["text"])
		return separation
