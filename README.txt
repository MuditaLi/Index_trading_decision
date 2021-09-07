├── LICENSE
├── README.md          <- The top-level README for developers using this project.
│
├── main.py            <- Main script to run the model.
│
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── references         <- Data dictionaries, assginment requirements.
│
├── reports            <- Generated evaluation metrics, feature importance, predicting performance, report according to requirement.
│
├── setup.py           <- Make this project pip installable with `pip install -e`
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Functions to load data and model.
    │   └── data_loader.py
    │
    ├── configs        <- Select configures & functions.
    │	├── config.py
    │	└── config_function_selection.yaml
    │
    ├── features       <- Select features to feed the model.
    │	└── feature_selection.yaml
    │
    ├── models         <- Scripts to train models and then use trained models to make predictions.
    │   │                 
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── utils          <- Function scripts to load & save datasets, as well as generating descriptions.
        │                 
        ├── inout_output.py
        ├── logger.py
	└── path.py
   
