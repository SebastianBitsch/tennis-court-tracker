# court-tracker

Simple deep learning based court tracking from a single camera broadcast. The main model is a CNN model based on the [TrackNet](https://arxiv.org/abs/1907.03698) architecture. The model predicts a heatmap over the corner points of the court which are then postprocessed to find the point coordinates from which the homography matrix can be constructed. 

## Results
For a more comprehensive demo see [this video](https://youtu.be/0OSVb9aRtk8?si=XEevofPiD7iMibOt) where the general outline of the method is explained and some more video demos are included.

#### 16 examples of different frames
<img width="884" alt="grid" src="https://github.com/SebastianBitsch/tennis-court-tracker/assets/72623007/9188e7f2-84d0-40da-b327-d1baf7756168">

#### Video sequence
![sequence](https://github.com/SebastianBitsch/tennis-court-tracker/assets/72623007/6493ca08-2d89-4f84-ba6f-72d4559fb94d)




## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── tennis_court_tracker  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
