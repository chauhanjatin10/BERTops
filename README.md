# BERTops
Source code for the paper [BERTops: Studying BERT Representations under a
Topological Lens]() accepted at [IJCNN 2022]() as on **Oral** Presentation.

## Usage

### Requirements
* [pytorch](https://pytorch.org/)
* [ripser](https://ripser.scikit-tda.org/en/latest/)
* [transformers](https://huggingface.co/docs/transformers/installation)
* [datasets](https://huggingface.co/docs/datasets/installation)
</br>

### Sample usage for SST-2 dataset

To generate the Persistence Diagrams (PDs), run the following sample code
```
python generate_label_pds.py --dataset_name sst-2 --base_model bert --model_card echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid
```
Here, the dataset_name, base_model and model_card (from Huggingface community models) can be adjusted for the usage.
</br></br>

To compute the PSF value for generated PDs, run the following sample code
```
python compute_PSF.py --dataset_name sst-2 --base_model bert --model_card echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid
```
</br>

To evaluate (compute test accuracy) the model, run the following sample code
```
python evaluate_test.py --dataset_name sst-2 --base_model bert --model_card echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid
```
</br>

## Citation

If you find this repository helpful, please consider citing our work:

```BibTeX

```
