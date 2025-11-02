# CDT Workshop ML 4 Time-series
### The Oxford EPSRC CDT in Health Data Science
HDS-M05: Module - Machine Learning for Time Series <br>
November 3 - 7, 2025 <br>


## Lab Course Designers
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->


<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<img src="./img/oxford_eng_logo.png" width="500" height="150" />
CHI Lab,<br/>
The Institute of Biomedical Engineering, <br />
Department of Engineering Science,<br />
University of Oxford<br />

## Lab Overview
This repository provides a comprehensive introduction to the fundamentals of applying machine learning (ML) to medical time-series data. You’ll explore how ML techniques for time-series data differ from traditional image-based or static models and discover key pre-processing steps specific to time-series data. The module also covers strategies for framing problems and tasks within a temporal context.

Throughout this course, you will be introduced to essential time-series models, including the Autoregressive (AR) process, Recurrent Neural Networks (RNNs), State Space Models and Transformers—fundamental tools for applying ML to healthcare data. In the latter part of the course, we address privacy considerations in healthcare, examining how techniques like federated learning and dataset condensation can support the democratization of healthcare data.<br>

## Lab Schedule: ML 4 Time-series
- Lab 1: Essential Methodology - Pre-processing, basic predictive modelling & autogressive modelling ([lab materials](https://github.com/AnshThakur/CDT-TimeSeries/blob/main/labs/lab_1/CDT_ML4timeseries_Lab_1.ipynb))
---
- Lab 2: Recurrent Neural Networks and State Space Models
---

- Lab 3.1: Graph Neural Networks 
- Lab 3.2: Wearable 

---
- Lab 4: Federated Learning and Dataset condensation for time-series 
---

## Data Access
The accompanying pre-processed data for this module can be downloaded via 
[canvas.ox.ac.uk](https://canvas.ox.ac.uk/courses/268831/files/7599044?wrap=1) <br/>
It can also be downloaded from [my Google Drive](https://drive.google.com/file/d/1wXPC1brdP07Ln8rfeKrLCm-7Fg8fsBza/view?usp=sharing)

## Setup instructions on the Virtual Machines
1. Load and initialize Anaconda. This needs to be done only once (you may not need to run this if you already see `(bash)` written in front of your prompt).

   ```bash
   module load Anaconda3
   conda init bash
   ```
   Exit and re-login so that the above takes effect.
3. Create an anaconda environment from the provided requirements YAML file: 
   ```bash
   conda env create -f lab1Env2024.yml
   ```
4. You are now ready to use the environment: 
   ```bash
   conda activate lab1Env2024
   ```
   In future logins, you only need to run this last command.

## How to run Jupyter notebooks remotely

1. In your remote machine, launch a Jupyter notebook with a specified port, e.g. 9000:
   ```bash
   jupyter-notebook --no-browser --port=9000
   ```
   This will output something like:
   ```bash
   To access the notebook, open this URL:
   http://localhost:9000/?token=
   b3ee74d492a6348430f3b74b52309060dcb754e7bf3d6ce4
   ```

1. On your local machine, perform port-forwarding, e.g. the following forwards the remote port 9000 to the local port 8888:
   ```bash
   ssh -N -f -L localhost:8888:localhost:9000 username@remote_address
   ```
   Note: You can use the same port numbers for both local and remote.

1. Finally, copy the URL from step 1. Then in your local machine, open
Chrome and paste the URL, but change the port to the local port (or do nothing else if you used the same port).
You should be able see the notebooks now.



<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
   


