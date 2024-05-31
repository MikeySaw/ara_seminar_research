# ara_seminar_research

In this repo we will document the reproducibility issues of the paper [Aligning Actions Across Recipe Graphs](https://aclanthology.org/2021.emnlp-main.554/).

![ara_seminar_research recipe.png](asset/Recipes-Banner.jpg)

## Issues with Aligment_Model/requirements.txt 

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

### Google Documentation for our experiments.
our [Google Doc](https://docs.google.com/document/d/1NwZypYE2r4qWeEWYPY2tegbrlSTPtCY4laGUlINOYp0/edit)
