grasp-lift-eeg-detection

** Setup instructions

Install python virtualenv.
Create a virtualenv named .venv in the root directory of the project
Run source .venv/bin/activate to activate the virtualenv
Run pip install -r requirements.txt to install all the necessary requirements


The input files should be placed in data/train and data/test folders


Run python preprocessing/generate_raw_arrays.py  to generate the processed npy files that can be found in data/processed directory
