News
- `09/15/2024` Release the [templates](https://github.com/user-attachments/files/17004283/ActorsHQ_templates.zip) of ActorsHQ (Actor01 & Actor04) to facilitate training.
- `05/22/2024` :loudspeaker: <font color='magenta'><b> An extension work of Animatable Gaussians for human avatar relighting is available [here](https://animatable-gaussians.github.io/relight). Welcome to check it!</b></font>
- `03/11/2024` The code has been released. Welcome to have a try!
- `03/11/2024` [AvatarReX](AVATARREX_DATASET.md) dataset, a high-resolution multi-view video dataset for avatar modeling, has been released.
- `02/27/2024` Animatable Gaussians is accepted by CVPR 2024!

Todo
- [x] Release the code.
- [x] Release AvatarReX dataset.
- [ ] <del>Release all the checkpoints and preprocessed dataset.</del> Cancelled due to graduation. Please run on other cases yourself with the provided [configs](configs).

<div align="center">

# <b>Animatable Gaussians</b>: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling

<h2>CVPR 2024</h2>

[Zhe Li](https://lizhe00.github.io/) <sup>1</sup>, [Zerong Zheng](https://zhengzerong.github.io/) <sup>2</sup>, [Lizhen Wang](https://lizhenwangt.github.io/) <sup>1</sup>, [Yebin Liu](https://www.liuyebin.com) <sup>1</sup>

<sup>1</sup>Tsinghua Univserity <sup>2</sup>NNKosmos Technology

### [Projectpage](https://animatable-gaussians.github.io/) · [Paper](https://arxiv.org/pdf/2311.16096.pdf) · [Video](https://www.youtube.com/watch?v=kOmZxD0HxZI)

</div>

https://github.com/lizhe00/AnimatableGaussians/assets/61936670/484e1263-06ed-409b-b9a1-790f5b514832

***Abstract**: Modeling animatable human avatars from RGB videos is a long-standing and challenging problem. Recent works usually adopt MLP-based neural radiance fields (NeRF) to represent 3D humans, but it remains difficult for pure MLPs to regress pose-dependent garment details. To this end, we introduce Animatable Gaussians, a new avatar representation that leverages powerful 2D CNNs and 3D Gaussian splatting to create high-fidelity avatars. To associate 3D Gaussians with the animatable avatar, we learn a parametric template from the input videos, and then parameterize the template on two front & back canonical Gaussian maps where each pixel represents a 3D Gaussian. The learned template is adaptive to the wearing garments for modeling looser clothes like dresses. Such template-guided 2D parameterization enables us to employ a powerful StyleGAN-based CNN to learn the pose-dependent Gaussian maps for modeling detailed dynamic appearances. Furthermore, we introduce a pose projection strategy for better generalization given novel poses. Overall, our method can create lifelike avatars with dynamic, realistic and generalized appearances. Experiments show that our method outperforms other state-of-the-art approaches.*

## Demo Results
We show avatars animated by challenging motions from [AMASS](https://amass.is.tue.mpg.de/) dataset.

https://github.com/lizhe00/AnimatableGaussians/assets/61936670/123b026a-3fac-473c-a263-c3dcdd2ecc4c
<details><summary>More results (click to expand)</summary>

https://github.com/lizhe00/AnimatableGaussians/assets/61936670/9abfa02f-65ec-46b3-9690-ac26191a5a7e

https://github.com/lizhe00/AnimatableGaussians/assets/61936670/c4f1e499-9bea-419c-916b-8d9ec4169ac3

https://github.com/lizhe00/AnimatableGaussians/assets/61936670/47b08e6f-a1f2-4597-bb75-d85e784cd97c
</details>

# Installation
0. Clone this repo.
```
git clone https://github.com/lizhe00/AnimatableGaussians.git
# or
git clone git@github.com:lizhe00/AnimatableGaussians.git
```
1. Install environments.
```
# install requirements
pip install -r requirements.txt

# install diff-gaussian-rasterization-depth-alpha
cd gaussians/diff_gaussian_rasterization_depth_alpha
python setup.py install
cd ../..

# install styleunet
cd network/styleunet
python setup.py install
cd ../..
```
2. Download [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php) model, and place pkl files to ```./smpl_files/smplx```.

# Data Preparation
## AvatarReX, ActorsHQ or THuman4.0 Dataset
1. Download [AvatarReX](./AVATARREX_DATASET.md), [ActorsHQ](https://www.actors-hq.com/dataset), or [THuman4.0](https://github.com/ZhengZerong/THUman4.0-Dataset) datasets.
2. Data preprocessing. We provide two manners below. The first way is recommended if you plan to employ our pretrained models, because the renderer utilized in preprocessing may cause slight differences.
    1. (Recommended) Download our preprocessed files from [PREPROCESSED_DATASET.md](PREPROCESSED_DATASET.md), and unzip them to the root path of each character. 
    2. Follow the instructions in [gen_data/GEN_DATA.md](gen_data/GEN_DATA.md#Preprocessing) to preprocess the dataset.
    
*Note for ActorsHQ dataset: 1) **DATA PATH.** The subject from ActorsHQ dataset may include more than one sequences, but we only utilize the first sequence, i.e., ```Sequence1```. The root path is ```ActorsHQ/Actor0*/Sequence1```. 2) **SMPL-X Registration.** We provide SMPL-X fitting for ActorsHQ dataset. You can download it from [here](https://drive.google.com/file/d/1DVk3k-eNbVqVCkLhGJhD_e9ILLCwhspR/view?usp=sharing), and place `smpl_params.npz` at the corresponding root path of each subject.*

## Customized Dataset
Please refer to [gen_data/GEN_DATA.md](gen_data/GEN_DATA.md) to run on your own data.

# Avatar Training
Take `avatarrex_zzr` from AvatarReX dataset as an example, run:
```
python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml --mode=train
```
After training, the checkpoint will be saved in `./results/avatarrex_zzr/avatar`. 

# Avatar Animation
1. Download pretrained checkpoint from [PRETRAINED_MODEL.md](./PRETRAINED_MODEL.md), unzip it to `./results/avatarrex_zzr/avatar`, or train the network from scratch.
2. Download [THuman4.0_POSE](https://drive.google.com/file/d/1pbToBV6klq6-dXCorwjjsmnINXZCG8n9/view?usp=sharing) or [AMASS](https://amass.is.tue.mpg.de/) dataset for acquiring driving pose sequences.
We list some awesome pose sequences from AMASS dataset in [configs/awesome_amass_poses.yaml](configs/awesome_amass_poses.yaml).
Specify the testing pose path in [configs/avatarrex_zzr/avatar.yaml#L57](configs/avatarrex_zzr/avatar.yaml#L57).
3. Run:
```
python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml --mode=test
```
You will see the animation results like below in `./test_results/avatarrex_zzr/avatar`.

https://github.com/lizhe00/AnimatableGaussians/assets/61936670/5aad39d2-2adb-4b7b-ab90-dea46240344a

# Evaluation
We provide evaluation metrics and example codes of comparison with body-only avatars in [eval/comparison_body_only_avatars.py](eval/comparison_body_only_avatars.py).

# Acknowledgement
Our code is based on these wonderful repos:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization) and its [adapted version](https://github.com/ashawkey/diff-gaussian-rasterization)
- [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar)

# Citation
If you find our code or data is helpful to your research, please consider citing our paper.
```bibtex
@inproceedings{li2024animatablegaussians,
  title={Animatable Gaussians: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling},
  author={Li, Zhe and Zheng, Zerong and Wang, Lizhen and Liu, Yebin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

