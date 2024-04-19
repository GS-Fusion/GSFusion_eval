# GSFusion_eval
Automatic evaluation system for GSFusion


## Installation

Create virtual environment
```
conda create -n gs_eval python=3.8
conda activate gs_eval
pip install -r requirements.txt
```


## Evaluation

First render training/novel views using the trained model
```
python render.py -m <path to trained model> --iteration <#iters> --data_device cuda
```

Then calculate error metrics of rendered images
```
python eval.py --data <path to rendered image folder> --no-eval-depth
```