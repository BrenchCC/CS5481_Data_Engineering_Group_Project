# CS5481-Data-Engineering-Group-Project
> - KKBOX-Music Recommendation Group Project, including: Data preparation; Data pre-processing and Exploratory data analysis; Multiple model algorithm implementation; Hyper Parameter Optimization with multi metrics evaluation;
> - See the [Report Overview Feishu Document](https://nwtbnnqpuup.feishu.cn/wiki/Qxouw3EPjiNZhakcgA2cl63rnkf)
> - See the [Project tutorial](docs/Group_project_tutorial.pdf); [Project tutorial Chinese](docs/group_project_tutorial_cn.md)
# Project Stucture
```ascii
CS5481_Group_Project
├── data
│   ├── aggregated_data
│   ├── processed_data
│   ├── raw_data
│   └── README.md
├── docs
│   ├── Group_project_tutorial.pdf
│   └── group_project_tutorial_cn.md
├── images
│   ├── EDA
│   └── models
├── model_ckpts
├── models
│   ├── __init__.py
│   ├── collaborative_filter.py
│   ├── lgbm.py
│   ├── lgfm.py
│   └── xgb.py
├── utils
│   ├── __init__.py
│   └── data_transform.py
├── data_pre-process_with_EDA.ipynb
├── model_using.ipynb
├── requirements.txt
└── README.md
```

## 1. Data Preparation
- See the data related [instruction](data/README.md)
  - Download raw data
  - Aggerate Data in one files
  - Store the processed data

## 2. Code Environment Preparation
```bash
pip install -r requirements.txt
```
- **Tips one: If you find errors when you try `pip install lightfm`, try to using `conda install -c conda-forge lightfm`**
- **Tips two: If you have GPU with CUDA backend inference system, you can install `torch` complied with CUDA in [Pytorch official website](https://pytorch.org/) to speed up model training**

## 3. Code Related Instruction
### 3.1 Data Transform Code
> [Data Transform Code](utils/data_transform.py)
- Aggerate Data in one files for data pre-processing

### 3.2 Model Algorithms Code
> [Model Algorithms Code](models)
```ascii
├── models
│   ├── __init__.py
│   ├── collaborative_filter.py
│   ├── lgbm.py
│   ├── lgfm.py
│   └── xgb.py
```
Four sets of model algorithms have been completed and encapsulated into classes, featuring the following functionalities: 
- Model training 
- Model saving for using
- Evaluation for testing metrics
- Visualization and figure saving
- Model loading for using
- Logging for code running information

### 3.3 Data Pre-Processing
- Using the [Jupyter Notebook](data_pre-process_with_EDA.ipynb)

### 3.4 Model Using after training
- Using the [Jupyter Notebook](model_using.ipynb)

