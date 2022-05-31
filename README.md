mlops_final_project
==============================

Final project in the DTU MLOPS course

Prepare data
------------
Download datset from https://www.kaggle.com/datasets/alessiocorrado99/animals10

Create a folder named data and put the downloaded data into the datafolder

run a root:

usage: python src/data/make_dataset.py --input_filepath --output_filepath

args:
--input_filepath    The path of tha raw data
--output_filepath   The path to where the processed data will be saved

Script for creating train/test splits and saving the processed data.
--------

Training
------------
Script for training image classifier.

run a root:
usage: python src/models/train_model.py --accelerator --devices --model_save_path --model_name

args:
--accelerator       write cpu or gpu
--devices           the number of devices to use
--model_save_path   the path were the model will be saved to
--model_name        save the model as this name

To use different augmentation edit/add config file to src/models/config/ and specify the settings for augmentation.
--------

Prediction
------------
predicting using a pre-trained image classifier.

run a root:
usage: python src/models/predict_model.py --model_path --image_path

--model_path    the path to the model
--image_path    path to the image to preform predictions on 
--------


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
