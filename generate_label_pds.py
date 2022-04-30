import numpy as np
import pickle
import os
import torch
import time
import random
import math
import torch.nn as nn
from ripser import ripser
from utils import parse_arguments, LoadData
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModel


class LabelPDGenerator:
	def __init__(self, args):
		self.args = args

		self.config = AutoConfig.from_pretrained(args.model_card)
		full_model = AutoModelForSequenceClassification.from_pretrained(args.model_card, output_attentions=True,
					output_hidden_states=True)
		print("\nModel successfully loaded\n")
		self.tokenizer = AutoTokenizer.from_pretrained(args.model_card)
		if args.base_model == "bert":
			self.embedding_model = full_model.bert
		else:
			raise Exception("Adjust embedding_model")


	def load_samples(self):
		complete_dataset = LoadData(self.args)
		classwise = complete_dataset.classwise_separation(complete_dataset.train_ds)

		for key, val in classwise.items():
			random.shuffle(classwise[key])
			print("Num sentences in label", key, " = ", len(classwise[key]))

		return classwise

	def obtain_embeddings(self, sents):
		with torch.no_grad():
			self.embedding_model.eval()
			self.embedding_model.cuda()

			all_embeds = []
			iters = math.ceil(len(sents) / self.args.eval_bs)
			for i in range(iters):
				batch_sentences = sents[i*self.args.eval_bs : (i+1)*self.args.eval_bs]
				batch = self.tokenizer(batch_sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
				for key, val in batch.items():
					batch[key] = batch[key].cuda()

				outs = self.embedding_model(**batch)
				all_embeds.append(outs[2][self.args.layer_for_pd][:, 0, :].cpu())

			all_embeds = torch.cat(all_embeds, dim=0).numpy()
		return all_embeds

	def fetch_sample_representatives(self, embeds):
		if embeds.shape[0] < self.args.num_sampled_for_label_pds:
			return embeds
		from sklearn.neighbors import KernelDensity
		kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(embeds)
		b = kde.sample(n_samples=self.args.num_sampled_for_label_pds, random_state=None)
		return b

	def create_pds(self):
		start_time = time.time()
		classwise = self.load_samples()

		all_samples_reps = {}
		all_pds = {}
		for label, _ in classwise.items():
			embeds = self.obtain_embeddings(classwise[label])
			print("label = ", label, " , embeds shape = ", embeds.shape)
			representatives = self.fetch_sample_representatives(embeds)	# downsampling via KDE
			print("sample representatives shape = ", representatives.shape)
			pds = ripser(representatives)['dgms']
			all_pds[label] = pds
			all_samples_reps[label] = representatives

		self.save_all_label_pds(all_pds)
		self.save_samples_reps(all_samples_reps)
		print("Time taken = ", time.time() - start_time)

	def save_all_label_pds(self, all_pds):
		dirname = 'saved_label_pds/{0}/{1}/'.format(self.args.base_model, self.args.dataset_name)
		if not os.path.exists(dirname):
			os.makedirs(dirname)

		out_file = dirname + "label_pds_" + str(self.args.layer_for_pd) + ".pickle"
		with open(out_file, 'wb') as handle:
			pickle.dump(all_pds, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def save_samples_reps(self, all_samples_reps):
		dirname = 'saved_sample_reps/{0}/{1}/'.format(self.args.base_model, self.args.dataset_name)
		if not os.path.exists(dirname):
			os.makedirs(dirname)

		out_file = dirname + "sample_reps_" + str(self.args.layer_for_pd) + ".pickle"
		with open(out_file, 'wb') as handle:
			pickle.dump(all_samples_reps, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	args = parse_arguments()

	pd_generator = LabelPDGenerator(args)
	pd_generator.create_pds()
