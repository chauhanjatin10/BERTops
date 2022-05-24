# BERTops
Source code for the paper [BERTops: Studying BERT Representations under a
Topological Lens]() accepted at [IJCNN 2022](https://wcci2022.org/call-for-papers/) as an **Oral** Presentation. [Arxiv Link](https://arxiv.org/abs/2205.00953)

## Usage

### Requirements
Provided in [requirements.txt](requirements.txt) file
</br></br>

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

To perform adversaril attack over the model (example run with Textbugger Black-box), run the following sample code
```
python perform_attack.py --dataset_name sst-2 --base_model bert --model_card echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid
```

## Citation

If you find this repository helpful, please consider citing our work:

```BibTeX
@article{https://doi.org/10.48550/arxiv.2205.00953,
  doi = {10.48550/ARXIV.2205.00953},
  url = {https://arxiv.org/abs/2205.00953},
  author = {Chauhan, Jatin and Kaul, Manohar},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {BERTops: Studying BERT Representations under a Topological Lens},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
