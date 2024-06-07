# ara_seminar_research

In this repo we will document the reproducibility issues of the paper [Aligning Actions Across Recipe Graphs](https://aclanthology.org/2021.emnlp-main.554/).

![ara_seminar_research recipe.png](asset/Recipes-Banner.jpg)

## Issues Part

### Issues with Aligment_Model/requirements.txt 

### The problem:
They provide package versions that are not compatible with eachother. Pandas 1.2.3 is not compatible with Python 3.7  

### The fix: 
Just download Pandas 1.1

### Issues with CUDA compilation:
CUDA would complain since the original paper is using a super old version of `torch==1.7.1` and the new generation of GPUs can not be compiled with old version torch, this will cause `torch.device("cuda")` unusable. \
To solve this error, please run the following commands:   \
```shell
pip uninstall torch
pip3 install torch==1.7.1  --force-reinstall  --extra-index-url https://download.pytorch.org/whl/cu110
```

## Data
Download the data from [here](https://github.com/interactive-cookbook/alignment-models/tree/main/data) and put it into `data/` in the root of your project to reproduce the results.


## Training
To train the model, run the following command from root directory:

`python train.py [model_name] --embedding_name [embedding_name]`

where `[model_name]` could be one of the following:
- `Sequence` : Sequential Ordering of Alignments
- `Cosine_similarity` : Cosine model (Baseline)
- `Naive` :  Common Action Pair Heuristics mode (Naive Model)
- `Alignment-no-feature` : Base Alignment model (w/o parent+child nodes)
- `Alignment-with-feature` : Extended Alignment model (with parent+child nodes)

and `[embedding_name]` could be one of the following:
- `bert` : BERT embeddings (default)
- `elmo` : ELMO embeddings


## Testing
To test the model, choose the application from the following:

Run the following command from this directory:

`python test_best_alignment.py [model_name] --embedding_name [embedding_name]`

As output, a prediction file named after the test dish(es) will be created. Here the best alignment computed for each action of the test recipes is saved.

## Google Documentation for our experiments.
our [Google Doc](https://docs.google.com/document/d/1NwZypYE2r4qWeEWYPY2tegbrlSTPtCY4laGUlINOYp0/edit)

