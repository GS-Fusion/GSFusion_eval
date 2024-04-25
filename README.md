# GSFusion_eval
Automatic evaluation system for GSFusion


## Installation

Clone the repository
```
git clone https://github.com/goldoak/GSFusion_eval.git --recursive
```

Create virtual environment
```
cd GSFusion_eval
conda create -n gs_eval python=3.8
conda activate gs_eval
pip install -r requirements.txt
```

Build differentiable Gaussian rasterizer
```
cd diff-gaussian-rasterization
python setup.py install
```


## Evaluation

First render training/novel views using the trained model
```
python render.py -m <path to trained model> --iteration <#iters> --dataset_type <supported dataset type> --data_device cuda
```

Then calculate error metrics of rendered images
```
python eval.py --data <path to rendered image folder> --no-eval-depth
```