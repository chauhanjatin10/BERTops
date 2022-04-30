import torch
from torch.utils.data import Dataset
from utils import parse_arguments, LoadData
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from datasets import load_metric
import math



class CustomDataset(Dataset):
	def __init__(self, sents, labels, tokenizer):
		self.sentences = sents
		self.labels = labels
		self.tokenizer = tokenizer
		self.pad_token_id = self.tokenizer.convert_tokens_to_ids("<pad>")

	def __len__(self):
		return len(self.sentences)

	def __getitem__(self, idx):
		sample = self.tokenizer(self.sentences[idx], truncation=True, max_length=128)
		sample["labels"] = self.labels[idx]
		sample["pad_token"] = self.pad_token_id
		return sample



class EvalRun:
	def __init__(self, arguments):
		super().__init__()
		self.arguments = arguments
		self.metric = load_metric("accuracy")

	def evaluator(self, model, sents, labels, tokenizer):
		with torch.no_grad():
			model.eval()
			model.cuda()

			y_true, y_pred = [], []

			iters = math.ceil(len(sents) / self.arguments.eval_bs)
			for i in range(iters):
				batch_sentences = sents[i*self.arguments.eval_bs : (i+1)*self.arguments.eval_bs]

				# batch = model.tokenize_text(batch_sentences)
				batch = tokenizer(batch_sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
				for key, val in batch.items():
					batch[key] = batch[key].cuda()

				# _, model_predictions, _ = model(**batch)
				output = model(**batch)['logits']
				model_predictions = torch.argmax(output, dim=-1).cpu()

				references = labels[i*self.arguments.eval_bs : (i+1)*self.arguments.eval_bs]
				self.metric.add_batch(predictions=model_predictions, references=references)

				y_true.extend(references)
				y_pred.extend(list(model_predictions.cpu().numpy()))

			final_score = self.metric.compute()
			print("\naccuracy = ", final_score, "\n")



if __name__ == '__main__':
	args = parse_arguments()

	complete_dataset = LoadData(args)
	complete_dataset.make_lists()

	tokenizer = AutoTokenizer.from_pretrained(args.model_card)

	eval_dataset = CustomDataset(complete_dataset.test_ds_sents, complete_dataset.test_ds_labels,
						tokenizer)

	model = AutoModelForSequenceClassification.from_pretrained(args.model_card, output_attentions=True,
					output_hidden_states=True)

	EvalRun(args).evaluator(model, eval_dataset.sentences, eval_dataset.labels, tokenizer)
