# Data Organization
We recommend follow [THuman4.0](https://github.com/ZhengZerong/THUman4.0-Dataset) dataset to organize your own data as shown below:
```
data_dir
├── images
|   └── cam00
│   └── cam01
├── masks
│   └── cam00
│   └── cam01
├── calibration.json
├── smpl_params.npz
```


# Preprocessing

1. (Optional) Reconstruct a template if the character is wearing loose clothes.
* Install additional libs.
```
cd ./utils/posevocab_custom_ops
python setup.py install
cd ../..

cd ./utils/root_finding
python setup.py install
cd ../..
```
* Generate a canonical LBS weight volume. Download [AdaptiveSolvers.x64.zip](https://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version16.01/AdaptiveSolvers.x64.zip) and extract ```PointInterpolant.exe``` to ```./bins```.
This step is required to be conducted on Windows.
```
python -m gen_data.gen_weight_volume -c configs/***/template.yaml
```
* Run the following script to reconstruct a template.
```
python main_template.py -c configs/***/template.yaml
```

2. Generate position maps.
```
python -m gen_data.gen_pos_maps -c configs/***/avatar.yaml
```
