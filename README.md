# CDT Workshop ML 4 Time-series
### The Oxford EPSRC CDT in Health Data Science
HDS-M05: Module - Machine Learning for Time Series <br>
November 6 - 10, 2023 <br>


## Lab Course Designers
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
   <td align="center"><a href="https://eng.ox.ac.uk/people/anshul-thakur/"><img src="https://eng.ox.ac.uk/media/4496/photo_ath.jpg?center=0.33112582781456956,0.31168831168831168&mode=crop&width=250&height=250&rnd=132651141320000000" width="120px;" alt=""/><br /><sub><b>Dr. Anshul Thakur</b></sub> </a> </td>
      <td align="center"><a href="https://eng.ox.ac.uk/people/tingting-zhu/"><img src="https://eng.ox.ac.uk/media/9549/photo_ttz2.jpg?center=0.30357142857142855,0.46745562130177515&mode=crop&width=100&height=100&rnd=132690066410000000" width="120px;" alt=""/><br /><sub><b>Dr. Tingting Zhu</b></sub> </a> </td>
   <td align="center"><a href="https://eng.ox.ac.uk/people/david-clifton/"><img src="https://www.turing.ac.uk/sites/default/files/styles/people/public/2021-12/david_clifton.jpg?itok=PYtYVF5_" width="100px;" alt=""/><br /><sub><b>Prof. David Clifton</b></sub> </a> </td>
    </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<img src="./img/oxford_eng_logo.png" width="500" height="150" />

The Institute of Biomedical Engineering, <br />
Department of Engineering Science,<br />
University of Oxford<br />

## Lab Overview
This repository aims to provide an introduction to the fundamentals of applying machine learning (ML) to medical time-series data. Within this module, you will discover that ML for time-series differs from traditional image-based or static modeling. You will gain insights into the essential pre-processing steps tailored to time-series data and learn how to frame the problem and task in a temporal context. This course will introduce fundamental time-series models, such as Autogresssive (AR) process, Markov Chains, and Hidden Markov Models (HMM), right through to Recurrent Neural Networks (RNNs) - staples of time-series data applied to healthcare problems.  In the latter part of this course, we delve into the privacy considerations surrounding healthcare data, exploring how federated learning and dataset condensation have the potential to facilitate the democratization of healthcare data.<br>

## Lecture Materials: AI/ML 4 time-series
- Day 1: Introduction to time-series analysis (lecture slides)
- Day 1: State-of-the art of AI/ML 4 time-series (lecture slides)
- Lab 1: Essential Methodology ([lab materials](https://github.com/apcreagh/CDTworkshop_ML4timeseries/blob/main/labs/lab_1))([solutions](https://github.com/apcreagh/CDTworkshop_ML4timeseries/blob/main/solutions/lab_1/))
---
- Day 2: Getting started with Gaussian processes (lecture slides)
- Day 2: Advanced Gaussian processes ([lecture slides](https://canvas.ox.ac.uk/courses/151592/files/4934754?wrap=1))
- Lab 2: Gaussian processes ([lab materials](https://github.com/apcreagh/CDTworkshop_ML4timeseries/blob/main/labs/lab_2))([solutions](https://github.com/apcreagh/CDTworkshop_ML4timeseries/tree/main/solutions/lab_2))
---
- Day 3: Recurrent Neural Networks ([lecture slides](https://github.com/apcreagh/CDTworkshop_ML4timeseries/blob/main/doc/CDT_HDS_ML4Timeseries-RNN_2022.pdf))
- Day 3: Advanced Recurrent Neural Networks and Multi-task learning ([lecture slides](https://github.com/apcreagh/CDTworkshop_ML4timeseries/blob/main/doc/CDT_HDS_ML4Timeseries-MTL-MTA_2022.pdf))
---
- Day 4: Transformations ([lecture slides](https://github.com/apcreagh/CDTworkshop_ML4timeseries/blob/main/doc/CDT_HDS_ML4Timeseries-FFT_2022.pdf))
- Day 4: Learning time-series features ([lecture slides](https://github.com/apcreagh/CDTworkshop_ML4timeseries/blob/main/doc/CDT_HDS_ML4Timeseries-CNN_2022.pdf))
- Lab 3: Recurrent Neural Networks ([lab materials](https://github.com/apcreagh/CDTworkshop_ML4timeseries/blob/main/labs/lab_3))
---
- Day 5: Deep Survival Analysis (lecture slides)
- Day 5: AI/ML 4 clincial time-series applications (lecture slides)
- Lab 4: Multi-task learning and Meta-learning for time-series ([lab materials](https://canvas.ox.ac.uk/courses/151592/files/4943018?))

Further lecture materials can be found on
[canvas.ox.ac.uk](https://canvas.ox.ac.uk/courses/151592/pages/hds-m05-module-info-machine-learning-for-time-series)
## Data Access
The accompanying pre-processed data for this module can be downloaded via 
[canvas.ox.ac.uk](https://canvas.ox.ac.uk/courses/151592/files/4929999?wrap=1)

## Setup instructions on the Virtual Machines
1. Load and initialize Anaconda. This needs to be done only once (you may not need to run this if you already see `(bash)` written in front of your prompt).

   ```bash
   module load Anaconda3
   conda init bash
   ```
   Exit and re-login so that the above takes effect.
3. Create an anaconda environment from the provided requirements YAML file: 
   ```bash
   conda env create -f ml4timeseries.yml
   ```
4. You are now ready to use the environment: 
   ```bash
   conda activate ml4timeseries
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
