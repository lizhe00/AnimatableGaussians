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
* Generate a canonical LBS weight volume. 
    * **For Windows**: Download [AdaptiveSolvers.x64.zip](https://www.cs.jhu.edu/~misha/Code/PoissonRecon/Version16.01/AdaptiveSolvers.x64.zip) and extract ```PointInterpolant.exe``` to ```./bins```
    * **For Linux**:
        * Clone [Adaptive Multigrid Solvers (Version 16.04)](https://github.com/mkazhdan/PoissonRecon.git) to directory of your choice 
        * ```cd path/to/cloned/repo```
        * ```make pointinterpolant```
        * The resulting executable file is at ```path/to/cloned/repo/Bin/Linux/PointInterpolant```. Copy it to ```./bins``` (you may need to do ```mkdir ./bins``` beforehand)
        * Go to ```./gen_data/gen_weight_volume.py line 115```, change ```solve(smpl_model.lbs_weights.shape[-1], ".\\bins\\PointInterpolant.exe")``` to ```solve(smpl_model.lbs_weights.shape[-1], "./bins/PointInterpolant")```. (This is the resulting executable file we previously made.)
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
