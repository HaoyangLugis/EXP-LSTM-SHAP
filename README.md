# EXP-LSTM-SHAP
Quick Start
Install
Package                      Version
---------------------------- -----------
absl-py                      2.1.0
astunparse                   1.6.3
cachetools                   5.5.2
certifi                      2025.1.31
charset-normalizer           3.4.1
clang                        5.0
cloudpickle                  3.1.1
colorama                     0.4.6
contourpy                    1.3.0
cycler                       0.12.1
et-xmlfile                   2.0.0
flatbuffers                  25.2.10
fonttools                    4.56.0
gast                         0.4.0
google-auth                  2.38.0
google-auth-oauthlib         0.4.6
google-pasta                 0.2.0
grpcio                       1.70.0
h5py                         3.1.0
idna                         3.10
importlib-metadata           8.6.1
joblib                       1.4.2
keras                        2.10.0
Keras-Preprocessing          1.1.2
kiwisolver                   1.4.7
libclang                     18.1.1
llvmlite                     0.43.0
Markdown                     3.7
MarkupSafe                   3.0.2
matplotlib                   3.5.2
numba                        0.60.0
numpy                        1.25.2
oauthlib                     3.2.2
openpyxl                     3.0.9
opt-einsum                   3.3.0
packaging                    24.2
pandas                       2.0.3
pillow                       11.1.0
pip                          20.2.3
protobuf                     3.19.6
pyasn1                       0.6.1
pyasn1-modules               0.4.1
pyparsing                    3.2.1
python-dateutil              2.9.0.post0
pytz                         2025.1
requests                     2.32.3
requests-oauthlib            2.0.0
rsa                          4.9
scikit-learn                 1.6.1
scipy                        1.11.0
setuptools                   49.2.1
shap                         0.42.0
six                          1.15.0
slicer                       0.0.7
tensorboard                  2.10.1
tensorboard-data-server      0.6.1
tensorboard-plugin-wit       1.8.1
tensorflow                   2.10.0
tensorflow-estimator         2.10.0
tensorflow-gpu               2.10.0
tensorflow-io-gcs-filesystem 0.31.0
termcolor                    1.1.0
threadpoolctl                3.5.0
tqdm                         4.67.1
typing-extensions            3.7.4.3
tzdata                       2025.1
urllib3                      2.3.0
werkzeug                     3.1.3
wheel                        0.45.1
wrapt                        1.12.1
xlsxwriter                   3.2.5
zipp                         3.21.0

Data Setup
Download CAMELS-US dataset to data/folder. Expected structure:
data/
├── basin_attributes/
├── meteorology/ 
├── streamflow/
└── basin_list.csv
Basic Usage
from src import ExpHydroLSTM

# Initialize model
model = ExpHydroLSTM(n_regions=9)

# Train and predict
results = model.run_simulation(
    train_period=('1980','2000'),
    test_period=('2000','2010')
)

# SHAP analysis
shap_results = model.explain_errors()
Outputs
Corrected runoff simulations (NSE improved from 0.14 to 0.38)
SHAP feature importance rankings
Regional error analysis plots
Cite
If using this code, please reference our paper [Paper Title] and the CAMELS-US dataset.
# Data availability
CAMELS data can be downloaded at https://doi.org/10.5065/D6MW2F4D (Addor et al., 2017).
