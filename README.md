# CVPR-2025-reading-papers-with-code 

## 收集CVPR 2025论文&源码
## 收集全网对CVPR 2025论文的优质讲解

---

> 注1：欢迎各位作者大佬提交issue，分享CVPR 2025论文和开源项目！
>
> 注2：关于CV领域顶级期刊（TPAMI、IJCV等）论文解读大盘点，详见： [https://github.com/Paper2Chinese/Paper2Chinese](https://github.com/Paper2Chinese/Paper2Chinese)
>
> 注3：关于人工智能领域**NeurIPS顶会**论文解读大盘点，详见： [https://github.com/Paper2Chinese/NeurIPS2024-Reading-Paper-With-Code](https://github.com/Paper2Chinese/NeurIPS2024-Reading-Paper-With-Code)



# 【CVPR 2025 论文开源目录】

- [3DGS(Gaussian Splatting)](#3DGS)
- [Mamba / SSM)](#Mamba)
- [Avatars](#Avatars)
- [Backbone](#Backbone)
- [CLIP](#CLIP)
- [MAE](#MAE)
- [Embodied AI](#Embodied-AI)
- [GNN](#GNN)
- [具身智能](EmAI)
- [多模态大语言模型(MLLM)](#MLLM)
- [大语言模型(LLM)](#LLM)
- [视觉语言模型(VLM)](#VLM)
- [多模态(Multi-modal)](#multimodal)
- [NAS](#NAS)
- [OCR](#OCR)
- [NeRF](#NeRF)
- [DETR](#DETR)
- [Prompt](#Prompt)
- [视觉问答(Visual Question Answering)](#VQA)
- [强化学习(Reinforcement Learning)](#RL)
- [扩散模型(Diffusion Models)](#Diffusion)
- [ReID(重识别)](#ReID)
- [长尾分布(Long-Tail)](#Long-Tail)
- [Vision Transformer](#Vision-Transformer)
- [自监督学习(Self-supervised Learning)](#SSL)
- [联邦学习(Federated Learning)](#FL)
- [增量学习(Incremental Learning)](#IL)
- [数据增强(Data Augmentation)](#DA)
- [目标检测(Object Detection)](#Object-Detection)
- [异常检测(Anomaly Detection)](#Anomaly-Detection)
- [目标跟踪(Visual Tracking)](#VT)
- [语义分割(Semantic Segmentation)](#Semantic-Segmentation)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)
- [生物工程(bioengineering)](#bio)
- [医学图像(Medical Image)](#MI)
- [医学图像分割(Medical Image Segmentation)](#MIS)
- [视频目标分割(Video Object Segmentation)](#VOS)
- [视频实例分割(Video Instance Segmentation)](#VIS)
- [参考图像分割(Referring Image Segmentation)](#RIS)
- [图像抠图(Image Matting)](#Matting)
- [图像编辑(Image Editing)](#Image-Editing)
- [Low-level Vision](#LLV)
- [超分辨率(Super-Resolution)](#SR)
- [去噪(Denoising)](#Denoising)
- [去模糊(Deblur)](#Deblur)
- [自动驾驶(Autonomous Driving)](#Autonomous-Driving)
- [生成对抗网络(GAN)](#GAN)
- [3D点云(3D Point Cloud)](#3D-Point-Cloud)
- [3D目标检测(3D Object Detection)](#3DOD)
- [3D语义分割(3D Semantic Segmentation)](#3DSS)
- [3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking)
- [3D语义场景补全(3D Semantic Scene Completion)](#3DSSC)
- [3D配准(3D Registration)](#3D-Registration)
- [3D人体姿态估计(3D Human Pose Estimation)](#3D-Human-Pose-Estimation)
- [3D人体Mesh估计(3D Human Mesh Estimation)](#3D-Human-Pose-Estimation)
- [少样本学习(Few-Shot Learning)](#FewShot)
- [医学图像(Medical Image)](#Medical-Image)
- [图像生成(Image Generation)](#Image-Generation)
- [视频生成(Video Generation)](#Video-Generation)
- [3D生成(3D Generation)](#3D-Generation)
- [视频理解(Video Understanding)](#Video-Understanding)
- [持续学习(Continual Learning)](#CL)
- [行为识别(Action Recognition)](#Action-Recognition)
- [行为检测(Action Detection)](#Action-Detection)
- [人脸识别(Face Recognition)](#face-recognition)
- [文本检测(Text Detection)](#Text-Detection)
- [知识蒸馏(Knowledge Distillation)](#KD)
- [模型剪枝(Model Pruning)](#Pruning)
- [图像压缩(Image Compression)](#IC)
- [三维重建(3D Reconstruction)](#3D-Reconstruction)
- [深度估计(Depth Estimation)](#Depth-Estimation)
- [轨迹预测(Trajectory Prediction)](#TP)
- [车道线检测(Lane Detection)](#Lane-Detection)
- [图像描述(Image Captioning)](#Image-Captioning)
- [手语识别(Sign Language Recognition)](#SLR)
- [视频预测(Video Prediction)](#Video-Prediction)
- [新视点合成(Novel View Synthesis)](#NVS)
- [Zero-Shot Learning(零样本学习)](#ZSL)
- [立体匹配(Stereo Matching)](#Stereo-Matching)
- [特征匹配(Feature Matching)](#Feature-Matching)
- [场景图生成(Scene Graph Generation)](#SGG)
- [计数(Counting)](#Counting)
- [隐式神经表示(Implicit Neural Representations)](#INR)
- [图像质量评价(Image Quality Assessment)](#IQA)
- [视频质量评价(Video Quality Assessment)](#Video-Quality-Assessment)
- [数据集(Datasets)](#Datasets)
- [反学习(Machine Unlearning)](#Unlearning)
- [新任务(New Tasks)](#New-Tasks)
- [模型加速(Improving Reasoning)](#Improving-Reasoning)
- [时间序列(Time Series)](#Time-Series)
- [其他(Others)](#Others)

<a name="EmAI"></a>
# 具身智能（Embodied AI）
#### InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions
#### Link：[https://arxiv.org/pdf/2502.20390](https://arxiv.org/pdf/2502.20390)
#### Code：[https://sirui-xu.github.io/InterMimic/](https://sirui-xu.github.io/InterMimic/)


#### G3Flow: Generative 3D Semantic Flow for Pose-aware and Generalizable Object Manipulation
#### Link：[https://arxiv.org/pdf/2411.18369](https://arxiv.org/pdf/2411.18369)
#### Code：https://tianxingchen.github.io/G3Flow/




<a name="3DGS"></a>

# 3DGS(Gaussian Splatting)

#### Generative Gaussian Splatting for Unbounded 3D City Generation
- Link：[https://arxiv.org/pdf/2406.06526](https://arxiv.org/pdf/2406.06526)
- Code：[https://haozhexie.com/project/gaussian-city](https://haozhexie.com/project/gaussian-city)

#### HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting
- Link：[https://arxiv.org/pdf/2412.03844](https://arxiv.org/pdf/2412.03844)
- Code：[https://gujiaqivadin.github.io/hybridgs/](https://gujiaqivadin.github.io/hybridgs/)



<a name="3D-Reconstruction"></a>
# 三维重建(3D Reconstruction)

#### MESC-3D:Mining Effective Semantic Cues for 3D Reconstruction from a Single Image
- Link：[https://arxiv.org/abs/2502.20861](https://arxiv.org/abs/2502.20861)
- Code：[https://github.com/QINGQINGLE/MESC-3D](https://github.com/QINGQINGLE/MESC-3D)



<a name="Mamba"></a>

# Mamba / SSM

#### MAMBA4D: Efficient Long-Sequence Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models
#### Link：[https://arxiv.org/pdf/2405.14338](https://arxiv.org/pdf/2405.14338)
#### Code：[https://github.com/IRMVLab/Mamba4D](https://github.com/IRMVLab/Mamba4D)

<a name="Avatars"></a>

# Avatars

<a name="Autonomous-Driving"></a>

# 自动驾驶(Autonomous Driving)

#### CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving
#### Link：[https://arxiv.org/pdf/2502.19908](https://arxiv.org/pdf/2502.19908)


#### EVDiffuser: Plug-and-Play Diffusion Model for BEV Denoising with Ground-Truth Guidance
#### Link：[https://arxiv.org/pdf/2502.19694](https://arxiv.org/pdf/2502.19694)

<a name="Backbone"></a>

# Backbone

#### OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels
#### Link：[https://arxiv.org/pdf/2502.20087](https://arxiv.org/pdf/2502.20087)
#### Code：[https://bit.ly/4bdbmdl](https://bit.ly/4bdbmdl)


<a name="CLIP"></a>

# CLIP

#### CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation
#### Link：[https://arxiv.org/pdf/2502.19842](https://arxiv.org/pdf/2502.19842)
#### Code：[https://clip-analysis.github.io/](https://clip-analysis.github.io/)

<a name="MAE"></a>
# MAE



<a name="OCR"></a>

# OCR



<a name="Occupancy"></a>

# Occupancy




<a name="NeRF"></a>

# NeRF



<a name="DETR"></a>

# DETR


<a name="GNN"></a>

# GNN


<a name="Prompt"></a>

# Prompt



<a name="LLM"></a>
# 大语言模型(LLM)



<a name="VLM"></a>
# 视觉语言模型(LLM)

#### PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction
#### Link：[https://arxiv.org/pdf/2410.17247](https://arxiv.org/pdf/2410.17247)
#### Code：[https://github.com/Cooperx521/PyramidDrop](https://github.com/Cooperx521/PyramidDrop)
<a name="MLLM"></a>

# 多模态大语言模型(MLLM)

#### Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy
#### Link：[https://arxiv.org/pdf/2502.19902](https://arxiv.org/pdf/2502.19902)

<a name="multimodal"></a>

# 多模态

#### Do computer vision foundation models learn the low-level characteristics of the human visual system?
#### Link：[https://arxiv.org/pdf/2502.20256](https://arxiv.org/pdf/2502.20256)
####


#### Knowledge Bridger: Towards Training-free Missing Multi-modality Completion
#### Link：[https://arxiv.org/pdf/2502.19834](https://arxiv.org/pdf/2502.19834)


#### SSHNet: Unsupervised Cross-modal Homography Estimation via Problem Redefinition and Split Optimization
#### Link：[https://arxiv.org/pdf/2409.17993](https://arxiv.org/pdf/2409.17993)
#### Code：[https://github.com/Junchen-Yu/SSHNet](https://github.com/Junchen-Yu/SSHNet)



<a name="NAS"></a>

# NAS

<a name="VQA"></a>


<a name="RL"></a>

## 强化学习(Reinforcement Learning) 




<a name="ReID"></a>

# ReID(重识别)



<a name="Diffusion"></a>

# 扩散模型(Diffusion Models)

#### Attention Distillation: A Unified Approach to Visual Characteristics Transfer
- Link：[https://arxiv.org/pdf/2502.20235](https://arxiv.org/pdf/2502.20235)
- Code：https://github.com/xugao97/AttentionDistillation

#### Data-free Universal Adversarial Perturbation with Pseudo-semantic Prior
- Link: [https://arxiv.org/abs/2502.21048](https://arxiv.org/abs/2502.21048)



<a name="Vision-Transformer"></a>

# Vision Transformer



<a name="VL"></a>

# 视觉和语言(Vision-Language)



<a name="Object-Detection"></a>

# 目标检测(Object Detection)

<a name="DA"></a>

## 数据增强(Data Augmentation)




<a name="Anomaly-Detection"></a>

# 异常检测(Anomaly Detection)

#### Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection
- Link: [https://arxiv.org/abs/2502.20981](https://arxiv.org/abs/2502.20981)


<a name="VT"></a>

# 目标跟踪(Object Tracking)






<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

#### Towards Generalizable Scene Change Detection
- Link：[https://arxiv.org/abs/2409.06214](https://arxiv.org/abs/2409.06214)


<a name="FewShot"></a>

# 少样本学习(Few-Shot Learning)

#### ProAPO: Progressively Automatic Prompt Optimization for Visual Classification
#### Link：[https://arxiv.org/pdf/2502.19844](https://arxiv.org/pdf/2502.19844)
#### Code：[https://github.com/MorningStarOvO/ProAPO](https://github.com/MorningStarOvO/ProAPO)


<a name="bio"></a>



<a name="MI"></a>

# 医学图像(Medical Image)


<a name="MIS"></a>

# 医学图像分割(Medical Image Segmentation)

#### Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation
- Link：[https://arxiv.org/pdf/2502.20056](https://arxiv.org/pdf/2502.20056)
- Code：[https://github.com/mk-runner/MLRG](https://github.com/mk-runner/MLRG)

#### LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging
- Link: [https://arxiv.org/abs/2502.20985](https://arxiv.org/abs/2502.20985)
- Code: [https://github.com/MIC-DKFZ/LesionLocator](https://github.com/MIC-DKFZ/LesionLocator)


<a name="VOS"></a>

# 视频目标分割(Video Object Segmentation)



<a name="Autonomous-Driving"></a>

<a name="face-recognition"></a>

# 人脸识别(Face Recognition)


<a name="3D-Point-Cloud"></a>

# 3D点云(3D-Point-Cloud)

#### GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors
#### Link：[https://arxiv.org/pdf/2502.19896](https://arxiv.org/pdf/2502.19896)

<a name="SSL"></a>
# 自监督学习(Self-supervised Learning)

#### Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach
#### Link：[https://arxiv.org/pdf/2502.19691](https://arxiv.org/pdf/2502.19691)
#### Code：[https://github.com/chenchenzong/EAOA](https://github.com/chenchenzong/EAOA)

<a name="FL"></a>
# 联邦学习(Federated Learning)


<a name="IL"></a>
# 增量学习(Incremental Learning)



<a name="3DOD"></a>

# 3D语义分割(3D Semantic Segmentation)

<a name="Image-Editing"></a>

# 图像编辑(Image Editing)





<a name="Image-Inpainting"></a>

# 图像补全/图像修复(Image Inpainting)


<a name="GAN"></a>

# 生成对抗网络(GAN)




<a name="Video-Editing"></a>

# 视频编辑(Video Editing)



<a name="LLV"></a>

# Low-level Vision

#### One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion
#### Link：[https://arxiv.org/pdf/2502.19854](https://arxiv.org/pdf/2502.19854)
#### Code：[https://github.com/AWCXV/GIFNet](https://github.com/AWCXV/GIFNet)






# 超分辨率(Super-Resolution)



<a name="Denoising"></a>

# 去噪(Denoising)

## 图像去噪(Image Denoising)

<a name="3D-Human-Pose-Estimation"></a>



<a name="Image-Generation"></a>

# 图像生成(Image Generation)

#### Finding Local Diffusion Schrödinger Bridge using Kolmogorov-Arnold Network
#### Link：[https://arxiv.org/pdf/2502.19754](https://arxiv.org/pdf/2502.19754)
#### Code：[https://github.com/Qiu-XY/LDSB](https://github.com/Qiu-XY/LDSB)

<a name="Video-Generation"></a>

# 视频生成(Video Generation)





<a name="3D-Generation"></a>
# 3D生成

#### InsTaG: Learning Personalized 3D Talking Head from Few-Second Video
- Link：[https://arxiv.org/pdf/2502.20387](https://arxiv.org/pdf/2502.20387)
- Code：[https://fictionarry.github.io/InsTaG/](https://fictionarry.github.io/InsTaG/)

#### CADDreamer: CAD object Generation from Single-view Images
- Link：[https://arxiv.org/abs/2502.20732](https://arxiv.org/abs/2502.20732)




<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)





<a name="CL"></a>

# 持续学习(Continual Learning)




<a name="Action-Recognition"></a>

# 行为识别(Action Recognition)

#### TIMotion: Temporal and Interactive Framework for Efficient Human-Human Motion Generation
- Link：[https://arxiv.org/abs/2408.17135](https://arxiv.org/abs/2408.17135)
- Code：[https://aigc-explorer.github.io/TIMotion-page](https://aigc-explorer.github.io/TIMotion-page)

<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)

#### MonoTAKD: Teaching Assistant Knowledge Distillation for Monocular 3D Object Detection
#### Link：[https://arxiv.org/pdf/2404.04910](https://arxiv.org/pdf/2404.04910)
#### Code：[https://github.com/hoiliu-0801/MonoTAKD](https://github.com/hoiliu-0801/MonoTAKD)




<a name="IC"></a>

# 图像压缩(Image Compression)

#### Balanced Rate-Distortion Optimization in Learned Image Compression
- Link：[https://arxiv.org/pdf/2502.20161](https://arxiv.org/pdf/2502.20161)

#### Towards Practical Real-Time Neural Video Compression
- Link：[https://arxiv.org/abs/2502.20762](https://arxiv.org/abs/2502.20762)
- Code：[https://github.com/microsoft/DCVC](https://github.com/microsoft/DCVC)




<a name="Stereo-Matching"></a>

# 立体匹配(Stereo Matching)



<a name="SGG"></a>

# 场景图生成(Scene Graph Generation)



<a name="Counting"></a>

# 计数(Counting)




<a name="INR"></a>

# 隐式神经表示(Implicit Neural Representations)

#### RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings
#### Link：[https://arxiv.org/pdf/2502.19781](https://arxiv.org/pdf/2502.19781)

<a name="Video-Quality-Assessment"></a>

# 视频质量评价(Video Quality Assessment)

<a name="Datasets"></a>

# 数据集(Datasets)

#### EEE-Bench: A Comprehensive Multimodal Electrical And Electronics Engineering Benchmark
#### Link：[https://arxiv.org/pdf/2411.01492](https://arxiv.org/pdf/2411.01492)




#### Fish-Vista: A Multi-Purpose Dataset for Understanding & Identification of Traits from Images
#### Code：[https://arxiv.org/pdf/2407.08027](https://arxiv.org/pdf/2407.08027)

<a name="Unlearning"></a>
# 反学习(Machine Unlearning)



<a name="New-Tasks"></a>
# 新任务(New Tasks)


<a name="Improving-Reasoning"></a>
# 模型加速(Improving Reasoning)



<a name="Time-Series"></a>
# 时间序列(Time Series)


# 其他(Others)

#### Decoder Gradient Shield: Provable and High-Fidelity Prevention of Gradient-Based Box-Free Watermark Removal
- Link：[https://arxiv.org/abs/2502.20924](https://arxiv.org/abs/2502.20924)


#### DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations
- Link：[https://arxiv.org/abs/2502.06029](https://arxiv.org/abs/2502.06029)
- Code：[https://github.com/ipsitmantri/DiTASK](https://github.com/ipsitmantri/DiTASK)







