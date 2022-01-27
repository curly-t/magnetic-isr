# magnetic-isr
To set up this package it is best to use a fresh conda environment for now.
To do this, one must first install conda, and then download the file conda_environment.yml
which can be found at https://github.com/curly-t/magnetic-isr/blob/master/conda_environment.yml

You then execute commands

conda env create -n NAME_OF_NEW_ENV --file /path/to/conda_environment.yml && conda activate NAME_OF_NEW_ENV && cd /path/to/project_working_dir && pip install -e git+https://github.com/curly-t/magnetic-isr.git@master#egg=misr

Then you are all set.

If you want to get the latest version of the master branch you just run the command (while in the working directory and the correct conda environment)

pip install -e git+https://github.com/curly-t/magnetic-isr.git@master#egg=misr

Minimum python version is 3.9

Installed python modules:
- pyserial
- tkfilebrowser
- gvar
- numpy
- matplotlib
- scipy

For windows you also need:
- pywin32

WINDOWS ONLY: After install of pywin32, you need to run /path/to/conda/environment/Scripts/pywin32_postinstall.py 

