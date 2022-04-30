import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import math
from utils import parse_arguments, LoadData
import json, os
import OpenAttack as oa
import tensorflow_text
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from openattack_utils import OurInvokeLimitedAttackEval



def load_dataset_for_text_attack(args):
	load_data = LoadData(args)
	load_data.make_lists()
	dict_ = load_data.fetch_random_sentences_for_attack()
	sentences = [ s[0].lower() for s in dict_["original_sentences"] ]
	return sentences


class AttackedModel(oa.Classifier):
	def __init__(self, args, loaded_model, tokenizer):
		self.args = args
		self.model = loaded_model
		self.tokenizer = tokenizer

	def get_both_pred_and_prob(self, *input_):
		# input_ is generally a list of strings, each string being a sentence
		input_ = input_[0]
		with torch.no_grad():
			all_probs = []
			all_preds = []

			num_iters = math.ceil(len(input_) / self.args.eval_bs)
			for itr in range(num_iters):
				current_inp = input_[itr*self.args.eval_bs : (itr+1)*self.args.eval_bs]
				if len(current_inp) == 0:
					break

				current_inp = self.tokenizer(current_inp, padding=True, truncation=True, max_length=128, return_tensors="pt")
				for key, val in current_inp.items():
					current_inp[key] = current_inp[key].cuda()

				outs = self.model(**current_inp)['logits']
				probs = F.softmax(outs, dim=1)
				all_probs.append(probs)
				label = torch.argmax(probs, dim=1)
				all_preds.append(label)

			all_probs = torch.cat(all_probs, dim=0)
			all_preds = torch.cat(all_preds, dim=0)
			return all_probs.cpu().numpy(), all_preds.cpu().numpy()

	def get_prob(self, *input_):
		# input_ is generally a list of strings, each string being a sentence
		input_ = input_[0]
		with torch.no_grad():
			all_probs = []
			num_iters = math.ceil(len(input_) / self.args.eval_bs)
			for itr in range(num_iters):
				current_inp = input_[itr*self.args.eval_bs : (itr+1)*self.args.eval_bs]
				if len(current_inp) == 0:
					break

				current_inp = self.tokenizer(current_inp, padding=True, truncation=True, max_length=128, return_tensors="pt")
				for key, val in current_inp.items():
					current_inp[key] = current_inp[key].cuda()

				outs = self.model(**current_inp)['logits']
				probs = F.softmax(outs, dim=1)
				all_probs.append(probs)
			all_probs = torch.cat(all_probs, dim=0)
			return all_probs.cpu().numpy()

	def get_pred(self, *input_):
		input_ = input_[0]
		with torch.no_grad():
			all_preds = []
			num_iters = math.ceil(len(input_) / self.args.eval_bs)

			for itr in range(num_iters):
				current_inp = input_[itr*self.args.eval_bs : (itr+1)*self.args.eval_bs]
				if len(current_inp) == 0:
					break

				current_inp = self.tokenizer(current_inp, padding=True, truncation=True, max_length=128, return_tensors="pt")
				for key, val in current_inp.items():
					current_inp[key] = current_inp[key].cuda()

				outs = self.model(**current_inp)['logits']
				probs = F.softmax(outs, dim=1)
				label = torch.argmax(probs, dim=1)
				all_preds.append(label)
			all_preds = torch.cat(all_preds, dim=0)
			return all_preds.cpu().numpy()



def run_attacker(args):
	loaded_sentences = load_dataset_for_text_attack(args)
	sentences = []

	for i, ss in enumerate(loaded_sentences):
		if ss == "":
			continue
		else:
			sentences.append(ss)

	max_sent_len = max([len(sent.split(' ')) for sent in sentences])

	data_ = oa.utils.Dataset(data_list=sentences, copy=True)

	loaded_model = AutoModelForSequenceClassification.from_pretrained(args.model_card, output_attentions=True,
				output_hidden_states=True)
	print("\nModel successfully loaded\n")
	tokenizer = AutoTokenizer.from_pretrained(args.model_card)

	loaded_model.cuda()
	loaded_model.eval()
	victim = AttackedModel(args, loaded_model, tokenizer)

	attacker = oa.attackers.TextBuggerAttacker(blackbox=True)

	current_invoke_limit = args.invoke_limit
	print("Invoke Limit = ", current_invoke_limit)

	options = {
		"success_rate": True,
		"fluency": True,
		"mistake": True,
		"semantic": True,
		"levenstein": True,
		"word_distance": True,
		"modification_rate": True,
		"running_time": True,
		"invoke_limit": current_invoke_limit,
		"average_invoke": True
	}

	attack_eval = OurInvokeLimitedAttackEval(attacker, victim, **options)
	result, success_list = attack_eval.generate_adv(data_)
	print("\nAttack success rate = ", sum(success_list)/len(success_list), "\n")


if __name__ == '__main__':
	args = parse_arguments()
	run_attacker(args)
