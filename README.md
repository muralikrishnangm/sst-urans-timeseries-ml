# Interpretable Deep Time-Series ML Closure of URANS for SST

This repo consists of codes to perform modeling of Unsteady Reynolds-Averaged Navier-Stokes (URANS) equations for stably stratified turbulence (SST) using deep time-series ML models and interpreting the data requirments for the models.

**NOTE:** Use the Jupyter notebooks only for debugging purposes. Use the Python scripts for the production usage.

Referece:
* M. Gopalakrishnan Meena, D. Liousas, A. D. Simin, A. Kashi, W. H. Brewer, J. J. Riley, and S. M. de Bruyn Kops, "Machine-Learned Closure of URANS for stably stratified turbulence: Connecting physical timescales & data hyperparameters of deep time-series models," [arXiv:2404.16141](https://doi.org/10.48550/arXiv.2404.16141), 2024

Contributors:
* Murali Gopalakrishnan Meena (Oak Ridge National Laboratory)
* Demetri Liousas (University of Massachusetts Amherst; Now at MIT Lincoln Laboratory)
* Andrew Simin (University of Massachusetts Amherst; Now at DoD HPCMP PET/GDIT)
* Steve de Bruyn Kops (University of Massachusetts Amherst)

# Installation

Create custom conda environment with the ML libraries.

1. [Optional] Steps for your own mini-conda installation
    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /path/to/your/local/dir/for/miniconda
    source /path/to/your/local/dir/for/miniconda/bin/activate
    ```
3. Make custom conda env
    ```
    conda create --name sst-urans-ml python=3.11
    conda activate sst-urans-ml
    ```
4. Install libraries
    ```
    pip install -r requirements.txt --no-cache-dir
    ```


# Run

## Using JupyterLab

* Resources:
  * [OLCF JupyterHub](https://docs.olcf.ornl.gov/services_and_applications/jupyter/overview.html#jupyter-at-olcf)
  * [NERSC JupyterHub](https://docs.nersc.gov/services/jupyter/)
* Always install and start JupyterLab in your base conda env.
* Use custom kernels as needed. See below for how to install custom kernels.
* To import your custom conda env to JupyterLab, follow the steps below which are modified from [OLCF Jupyter docs](https://docs.olcf.ornl.gov/services_and_applications/jupyter/overview.html#example-creating-a-conda-environment-for-rapids):
  * Install JupyterLab in your custom conda env. Do the rest of the steps in your base env.
  * Follow steps 1-2 in [OLCF Jupyter docs](https://docs.olcf.ornl.gov/services_and_applications/jupyter/overview.html#example-creating-a-conda-environment-for-rapids): Open a Terminal on JupyterLab using the Launcher.
  * Skip step 3: You don't have to create your own custom conda env as you have already done this.
  * Follow step 4 (source activate your custom env) using the custom env you created.
  * Follow step 5 (make your env visible in JupyterLab) using your desired env name: `python -m ipykernel install --user --name [env-name] --display-name [env-name]`. You may have to pip install the library `wcwidth` on the Jupyter terminal: `pip install wcwidth`
  * Finally refresh your page and the Lancher (and kernel selector for notebooks) will have your env.

## For LSTM models

For getting a general understanding of the modeling process and debugging the codes, use the following Jupyter notebooks. These will perform the modeling for the SST case.

1. [DataAcqu_LSTM.ipynb](DataAcqu_LSTM.ipynb): for loading the time series data and processing it into a format readable by the ML training script.
2. [nnTraining.ipynb](nnTraining.ipynb): for training the LSTM model.
3. [nnTesting_apriori.ipynb](nnTesting_apriori.ipynb): for *a priori* testing (offline testing) of the LSTM model.
4. [nnTesting_aposteriori.ipynb](nnTesting_aposteriori.ipynb): for *a posteriori* testing (online testing) - replacing the RHS of the ODE with the LSTM model.

**Pendulum case:** A data generation code for the pendulum case is also provided ([DataAcqu_LSTM_pendulum.py](DataAcqu_LSTM_pendulum.py)). For rest of the codes, please change the prefix of the data and model file names accordingly.

For batch submission of multiple cases on HPC machines, run the following scripts (examples for [OLCF Andes](https://docs.olcf.ornl.gov/systems/andes_user_guide.html)):
1. Get data and train LSTM models ([job_andes_LSTM_training.sh](job_andes_LSTM_training.sh)):
    ```
    bsub  job_andes_LSTM_training.sh
    ```
2. Offline testing - *a priori* testing ([job_andes_LSTM_apriori.sh](job_andes_LSTM_apriori.sh)):
    ```
    bsub  job_andes_LSTM_apriori.sh
    ```
3. Online testing - *a posteriori* testing ([job_andes_LSTM_aposteriori.sh](job_andes_LSTM_aposteriori.sh)):
    ```
    bsub  job_andes_LSTM_aposteriori.sh
    ```

## For NODE models

1. First generate data for LSTM models as shown above.
2. [DataAcqu_NODE.ipynb](DataAcqu_NODE.ipynb): for processing data generated from the LSTM data generation script. Can be used to generate data with multiple cases.
3. [NODE_nnTraining.ipynb](NODE_nnTraining.ipynb): for training the NODE model.
4. [NODE_nnTesting.ipynb](NODE_nnTesting.ipynb): for testing the NODE model

# Plotting results

Use the following Jupyter notebooks for plotting the results and comparison of the ML models. Model outputs and data needed for the plotting routines are provided here: [10.5281/zenodo.13787850](https://doi.org/10.5281/zenodo.13787850)

1. [plot_compare-LSTM-NODE.ipynb](plot_compare-LSTM-NODE.ipynb): for comparing LSTM and NODE results.
2. [plot_data-hyperparameter-study_comparison.ipynb](plot_data-hyperparameter-study_comparison.ipynb): for plotting results of data-hyperparameter (time-scale) study for a given flow case (F2R32 is analyzed in this code).