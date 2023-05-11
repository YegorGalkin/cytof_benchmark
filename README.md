# Table of contents
1. [Requirements](#requirements)
2. [Installation guide](#install)
3. [Data download](#download)
4. [Preprocessing](#preprocessing)
5. [Simple network training](#training)
6. [HPO with population based training](#pbt)

## Requirements <a name="requirements"/>
Recommended OS: Ubuntu 22.04 LTS  
Required hardware: CUDA capable NVIDIA GPU with at least 8GB VRAM, 64GB RAM for the preprocessing of the largest dataset.
## Installation guide <a name="install"/>
1. Install CUDA following the official installation [nvidia guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (requires root acces)
2. Validate CUDA installation following the previous guide
3. Install python 3.10 
4. Install packages from requirements.txt
   1. For projections of spherical latent spaces into 2d maps, install dependencies for the cartopy package.  
   `sudo apt -y install libgeos-dev`  
   or by using official [installation guide](https://scitools.org.uk/cartopy/docs/latest/installing.html)
   2. If the CUDA version is 11.7, reinstall pytorch using installation command from the [official site](https://pytorch.org/get-started/locally/)
5. Validate pytorch install by running cuda detection command from pytorch
    ```
    import torch 
    torch.cuda.is_available()
    ```
    ```
    True
    ```
6. Some visualization and data analysis pipelines require additional R installation
   1. Install [R](https://cloud.r-project.org/)
   2. Install [RStudio](https://posit.co/download/rstudio-desktop/)
   3. Install required R packages depending on the script (tidyverse, RANN, data.table, etc)
## Data download <a name="download"/>
### Organoid dataset
Sourced from paper [Cell-type-specific signaling networks in heterocellular organoids](https://www.nature.com/articles/s41592-020-0737-8)
1. Register in [cytobank](https://community.cytobank.org/) 
2. [Download](https://community.cytobank.org/cytobank/experiments/83654/download_files) and extract all experiment files in a single folder for the processed time course experiment. They will send download link to an email when it's ready.
3. It should contain 42 files with names following the template: `Figure-4_{cell_type}_Day-{day}.fcs`
4. Run `preprocessing/prepare_organoids.R` script on the downloaded data with the following arguments:
   1. Directory with downloaded files
   2. Output directory for the project
   3. Installation could take >5 min
   4. This will create a data subdirectory with organoid dataset data inside.  
   The final structure should be:
      `{project_root}/data/organoids/full/data.csv.gz`  
   Command example:
   ```
   Rscript ./prepare_organoids.R /home/egor/Downloads/experiment_83654_20230510135553026_files /data/PycharmProjects/cytof_benchmark
   ```
5. The dataset could now be loaded in python:
   ```
    from datasets import OrganoidDataset
    data = OrganoidDataset(data_dir='./data/organoids')
    X,y = data.train
   ```
### CAF dataset
Sourced from paper [Cancer-Associated Fibroblasts Regulate Patient-Derived Organoid Drug Responses](https://www.biorxiv.org/content/10.1101/2022.10.19.512668v2)  
Data is available at [Mendeley](https://data.mendeley.com/datasets/hc8gxwks3p)
1. Download and unzip the data (4.5 GB)
2. Move the `Metadata_final_paper` file to the directory `{project_root}/data/caf/Metadata_PDO_CAF_screening`
3. It is just a python pickle file. It could now be loaded in python:
   ```
    from datasets import CafDataset
    data = CafDataset(data_dir='./data/caf')
    X,y = data.train
   ```
### Breast cancer challenge dataset
Sourced from [Single Cell Signaling in Breast Cancer Challenge](https://www.synapse.org/#!Synapse:syn20366914/wiki/594730)  
1. Visit the [files](https://www.synapse.org/#!Synapse:syn20366914/files/) web page of the breast cancer challenge
2. There, find a data/single_cell_phospo/complete_cell_lines file subdirectory
3. Manually download every file. There should be 44 unique items in total with ~6.5GB total size. It is only possible to download 10 files at once.
4. Move all csv files to the directory `{project_root}/data/breast_cancer_challenge/`
5. It could now be loaded in python:
   ```
    from datasets import ChallengeDataset
    data = ChallengeDataset(data_dir='./data/breast_cancer_challenge')
    X,y = data.train
   ```
## Preprocessing <a name="preprocessing"/>
All datasets required different preprocessing pipelines, which are implemented in the `datasets.py` file.  
FACS data required normalization using `arcsinh(x/5)`, and some NAs were filtered.
## Training a single network <a name="training"/>
There is an example notebook that shows how to run neural network training step by step in the `examples/training.ipynb`.
It is also possible to use `experiment.py` script to train networks with specific parameters. It currently only works for the Organoid dataset.
This script uses cosine decay learning rate scheduler. The following parameters could be changed:
1. Output directory
2. Model (BetaVAE, db-VAE, WAE-MMD, S-VAE)
3. Batch size
4. Number of epochs
5. Maximum learning rate for the scheduler
6. Maximum gradient norm for the gradient clipping
7. Number of latent dimensions
8. Size and number of hidden layers
9. Activation function
10. Random seed

Example running scripts are available in the file `gridsearch.py`

## Hyperparameter optimization with population based training <a name="pbt"/>
`benchmark.py` has the code to run the population based training HPO for all 4 models and 3 datasets with 16 models in population for 8 hours each for a total of 4 day runtime.
It also provides an example of how to run population based training.
Important parameters that have to be changed depending on hardware, described in `configs/pbt/base_pbt.py`:
1. `gpus` - set to a number of gpus in system
2. `cpus` - set to a number of cpus in system
3. `concurrent` - if less than 24 GB VRAM, set to 8 or 4, while also setting `synch=True`.  
   This will force models to unload after each tuning iteration, until all 16 models complete a single tuning loop. In this case, HPO could take more time to complete.
4. `soft_time_limit` will set time limit, after which models will try to finish the last tuning iteration and terminate the HPO run.

The following parameters change behavior of the population based training itself:
1. `perturbation_interval` - Number of epochs to complete a single tuning iteration.
2. `perturbation_factors` - How metaparameters are changed during PBT perturbations
3. `lr_lower`, `lr_upper`, `bs_lower`, `bs_upper` - limits for the model metaparameters. `bs_lower` and `bs_upper` are hard limits depending on GPU VRAM.
4. `group_size` - Since a single sample only has ~ 40 features, they are minibatched in a group of such size to reduce data loader overhead. Data Loader considers `group_size` number of samples as a single item
5. `max_failures` - Ray Air will retry model trainign in case of errors, and terminate if amount of errors exceeds this value

The rest of parameters are model-specific and dataset-specific.  
Example usage is available in the file `benchmark.py`