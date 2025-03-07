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
- [脉冲网络](#SNN)
- [图像检索](#IRetrieval)

<a name="EmAI"></a>
# 具身智能（Embodied AI）
#### InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions
- Link：[https://arxiv.org/pdf/2502.20390](https://arxiv.org/pdf/2502.20390)
- Code：[https://sirui-xu.github.io/InterMimic/](https://sirui-xu.github.io/InterMimic/)


#### G3Flow: Generative 3D Semantic Flow for Pose-aware and Generalizable Object Manipulation
- Link：[https://arxiv.org/pdf/2411.18369](https://arxiv.org/pdf/2411.18369)
- Code：https://tianxingchen.github.io/G3Flow/

#### EgoLife: Towards Egocentric Life Assistant
- Link：[https://arxiv.org/pdf/2503.03803](https://arxiv.org/pdf/2503.03803)

#### SpiritSight Agent: Advanced GUI Agent with One Look
- Link：[https://arxiv.org/pdf/2503.03196](https://arxiv.org/pdf/2503.03196)
- Code：[https://huggingface.co/SenseLLM/SpiritSight-Agent-8B](https://huggingface.co/SenseLLM/SpiritSight-Agent-8B)

#### UniGraspTransformer: Simplified Policy Distillation for Scalable Dexterous Robotic Grasping
- Link：[https://arxiv.org/pdf/2412.02699](https://arxiv.org/pdf/2412.02699)
- Code：[https://dexhand.github.io/UniGraspTransformer](https://dexhand.github.io/UniGraspTransformer)

<a name="3DGS"></a>

# 3DGS(Gaussian Splatting)

#### Generative Gaussian Splatting for Unbounded 3D City Generation
- Link：[https://arxiv.org/pdf/2406.06526](https://arxiv.org/pdf/2406.06526)
- Code：[https://haozhexie.com/project/gaussian-city](https://haozhexie.com/project/gaussian-city)

#### HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting
- Link：[https://arxiv.org/pdf/2412.03844](https://arxiv.org/pdf/2412.03844)
- Code：[https://gujiaqivadin.github.io/hybridgs/](https://gujiaqivadin.github.io/hybridgs/)

#### S2Gaussian: Sparse - View Super - Resolution 3D Gaussian Splatting
- Link：[https://arxiv.org/pdf/2503.04314](https://arxiv.org/pdf/2503.04314)
- Code：[https://jeasco.github.io/S2Gaussian/](https://jeasco.github.io/S2Gaussian/)

#### GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting
- Link：[https://arxiv.org/pdf/2411.19895](https://arxiv.org/pdf/2411.19895)
- Code：[https://github.com/NarcissusEx/GuardSplat](https://github.com/NarcissusEx/GuardSplat)

<a name="3D-Reconstruction"></a>
# 三维重建(3D Reconstruction)

#### MESC-3D:Mining Effective Semantic Cues for 3D Reconstruction from a Single Image
- Link：[https://arxiv.org/abs/2502.20861](https://arxiv.org/abs/2502.20861)
- Code：[https://github.com/QINGQINGLE/MESC-3D](https://github.com/QINGQINGLE/MESC-3D)

#### FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video
- Link：[https://arxiv.org/abs/2503.04720](https://arxiv.org/abs/2503.04720)
- Code：[https://yuegao.me/FluidNexus/](https://yuegao.me/FluidNexus/)

#### IMFine: 3D Inpainting via Geometry - guided Multi - view Refinement
- Link：[https://arxiv.org/pdf/2503.04501](https://arxiv.org/pdf/2503.04501)
- Code：[https://xinxinzuo2353.github.io/imfine/](https://xinxinzuo2353.github.io/imfine/)





<a name="Mamba"></a>

# Mamba / SSM

#### MAMBA4D: Efficient Long-Sequence Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models
- Link：[https://arxiv.org/pdf/2405.14338](https://arxiv.org/pdf/2405.14338)
- Code：[https://github.com/IRMVLab/Mamba4D](https://github.com/IRMVLab/Mamba4D)

#### JamMa: Ultra - lightweight Local Feature Matching with Joint Mamba
- Link：[https://arxiv.org/pdf/2503.03437](https://arxiv.org/pdf/2503.03437)
- Code：https://leoluxxx.github.io/JamMa-page/

<a name="Avatars"></a>

# Avatars

<a name="Autonomous-Driving"></a>

# 自动驾驶(Autonomous Driving)

#### CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving
- Link：[https://arxiv.org/pdf/2502.19908](https://arxiv.org/pdf/2502.19908)


#### EVDiffuser: Plug-and-Play Diffusion Model for BEV Denoising with Ground-Truth Guidance
- Link：[https://arxiv.org/pdf/2502.19694](https://arxiv.org/pdf/2502.19694)


#### CoSDH: Communication - Efficient Collaborative Perception via Supply - Demand Awareness and Intermediate - Late Hybridization
- Link：[https://arxiv.org/pdf/2503.03430](https://arxiv.org/pdf/2503.03430)
- Code：[https://github.com/Xu2729/CoSDH](https://github.com/Xu2729/CoSDH)






<a name="Backbone"></a>

# Backbone

#### OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels
#### Link：[https://arxiv.org/pdf/2502.20087](https://arxiv.org/pdf/2502.20087)
#### Code：[https://bit.ly/4bdbmdl](https://bit.ly/4bdbmdl)


<a name="CLIP"></a>

# CLIP

#### CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation
- Link：[https://arxiv.org/pdf/2502.19842](https://arxiv.org/pdf/2502.19842)
- Code：[https://clip-analysis.github.io/](https://clip-analysis.github.io/)

#### CLIP is Strong Enough to Fight Back: Test - time Counterattacks towards Zero - shot Adversarial Robustness of CLIP
- Link：[https://arxiv.org/pdf/2503.03613](https://arxiv.org/pdf/2503.03613)
- Code：[https://github.com/Sxing2/CLIP-Test-time-Counterattacks](https://github.com/Sxing2/CLIP-Test-time-Counterattacks)


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
- Link：[https://arxiv.org/pdf/2410.17247](https://arxiv.org/pdf/2410.17247)
- Code：[https://github.com/Cooperx521/PyramidDrop](https://github.com/Cooperx521/PyramidDrop)

#### Reasoning to Attend: Try to Understand How <SEG> Token Work
- Link：[https://arxiv.org/pdf/2412.17741](https://arxiv.org/pdf/2412.17741)
- Code：[https://github.com/rui-qian/READ](https://github.com/rui-qian/READ)

<a name="MLLM"></a>

# 多模态大语言模型(MLLM)

#### Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy
#### Link：[https://arxiv.org/pdf/2502.19902](https://arxiv.org/pdf/2502.19902)

<a name="multimodal"></a>

# 多模态

#### Do computer vision foundation models learn the low-level characteristics of the human visual system?
- Link：[https://arxiv.org/pdf/2502.20256](https://arxiv.org/pdf/2502.20256)

#### Human Motion Instruction Tuning
- Link：[https://arxiv.org/pdf/2411.16805](https://arxiv.org/pdf/2411.16805)
- Code：[https://github.com/ILGLJ/LLaMo](https://github.com/ILGLJ/LLaMo)

#### Knowledge Bridger: Towards Training-free Missing Multi-modality Completion
- Link：[https://arxiv.org/pdf/2502.19834](https://arxiv.org/pdf/2502.19834)


#### SSHNet: Unsupervised Cross-modal Homography Estimation via Problem Redefinition and Split Optimization
- Link：[https://arxiv.org/pdf/2409.17993](https://arxiv.org/pdf/2409.17993)
- Code：[https://github.com/Junchen-Yu/SSHNet](https://github.com/Junchen-Yu/SSHNet)


#### LaVin - DiT: Large Vision Diffusion Transformer
- Link：[https://arxiv.org/pdf/2411.11505](https://arxiv.org/pdf/2411.11505)
- Code：[https://derrickwang005.github.io/LaVin-DiT/](https://derrickwang005.github.io/LaVin-DiT/)

  
#### DoraCycle: Domain - Oriented Adaptation of Unified Generative Model in Multimodal Cycles
- Link：[https://arxiv.org/pdf/2503.03651](https://arxiv.org/pdf/2503.03651)
- Code：[https://github.com/showlab/DoraCycle](https://github.com/showlab/DoraCycle)




#### LION - FS: Fast & Slow Video - Language Thinker as Online Video Assistant
- Link：[https://arxiv.org/pdf/2503.03663](https://arxiv.org/pdf/2503.03663)
- Code：[https://github.com/JiuTian-VL/LION-FS](https://github.com/JiuTian-VL/LION-FS)

  

<a name="NAS"></a>

# NAS

<a name="VQA"></a>

## 视觉问答(Visual Question Answering)

#### Question - Aware Gaussian Experts for Audio - Visual Question Answering
- Link：[https://arxiv.org/abs/2503.04459](https://arxiv.org/abs/2503.04459)
- Code：[https://github.com/AIM-SKKU/QA-TIGER](https://github.com/AIM-SKKU/QA-TIGER)

#### DSPNet: Dual - vision Scene Perception for Robust 3D Question Answering
- Link：[https://arxiv.org/pdf/2503.03190](https://arxiv.org/pdf/2503.03190)
- Code: [https://github.com/LZ-CH/DSPNet](https://github.com/LZ-CH/DSPNet)


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

#### Optimizing for the Shortest Path in Denoising Diffusion Model
- Link：[https://arxiv.org/pdf/2503.03265](https://arxiv.org/pdf/2503.03265)
- Code：[https://github.com/UnicomAI/ShortDF](https://github.com/UnicomAI/ShortDF)

<a name="Vision-Transformer"></a>

# Vision Transformer



<a name="VL"></a>

# 视觉和语言(Vision-Language)



<a name="Object-Detection"></a>

# 目标检测(Object Detection)

#### ReRAW: RGB - to - RAW Image Reconstruction via Stratified Sampling for Efficient Object Detection on the Edge
- Link：[https://arxiv.org/pdf/2503.03782](https://arxiv.org/pdf/2503.03782)
- Code：[https://anonymous.4open.science/r/ReRAW-0C87/](https://anonymous.4open.science/r/ReRAW-0C87/)
  



<a name="DA"></a>

## 数据增强(Data Augmentation)




<a name="Anomaly-Detection"></a>

# 异常检测(Anomaly Detection)

#### Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection
- Link: [https://arxiv.org/abs/2502.20981](https://arxiv.org/abs/2502.20981)

#### Towards Visual Discrimination and Reasoning of Real - World Physical Dynamics: Physics - Grounded Anomaly Detection
- Link：[https://arxiv.org/pdf/2503.03562](https://arxiv.org/pdf/2503.03562)



<a name="VT"></a>

# 目标跟踪(Object Tracking)

####  Omnidirectional Multi - Object Tracking
- Link：[https://arxiv.org/abs/2503.04565](https://arxiv.org/abs/2503.04565)
- Code：[https://github.com/xifen523/OmniTrack](https://github.com/xifen523/OmniTrack)




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

#### Q - PART: Quasi - Periodic Adaptive Regression with Test - time Training for Pediatric Left Ventricular Ejection Fraction Regression
- Link：[https://arxiv.org/pdf/2503.04131](https://arxiv.org/pdf/2503.04131)
- Code：[https://github.com/ljwztc/Q-PART](https://github.com/ljwztc/Q-PART)

#### MedUnifier: Unifying Vision - and - Language Pre - training on Medical Data with Vision Generation Task using Discrete Visual Representations
- Link：[https://arxiv.org/pdf/2503.01019](https://arxiv.org/pdf/2503.01019)



<a name="MIS"></a>

# 医学图像分割(Medical Image Segmentation)

#### Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation
- Link：[https://arxiv.org/pdf/2502.20056](https://arxiv.org/pdf/2502.20056)
- Code：[https://github.com/mk-runner/MLRG](https://github.com/mk-runner/MLRG)

#### LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging
- Link: [https://arxiv.org/abs/2502.20985](https://arxiv.org/abs/2502.20985)
- Code: [https://github.com/MIC-DKFZ/LesionLocator](https://github.com/MIC-DKFZ/LesionLocator)

#### Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation
- Link：[https://arxiv.org/pdf/2503.04639](https://arxiv.org/pdf/2503.04639)


<a name="VOS"></a>

# 视频目标分割(Video Object Segmentation)



<a name="Autonomous-Driving"></a>

<a name="face-recognition"></a>

# 人脸识别(Face Recognition)


<a name="3D-Point-Cloud"></a>

# 3D点云(3D-Point-Cloud)

#### GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors
- Link：[https://arxiv.org/pdf/2502.19896](https://arxiv.org/pdf/2502.19896)

#### Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation
- Link：[https://arxiv.org/pdf/2503.04718](https://arxiv.org/pdf/2503.04718)

#### Self - Supervised Large Scale Point Cloud Completion for Archaeological Site Restoration
- Link：[https://arxiv.org/pdf/2503.04030](https://arxiv.org/pdf/2503.04030)


#### ArcPro: Architectural Programs for Structured 3D Abstraction of Sparse Points
- Link：[https://arxiv.org/pdf/2503.02745](https://arxiv.org/pdf/2503.02745)
- Code：[https://vcc.tech/research/2025/ArcPro](https://vcc.tech/research/2025/ArcPro)

#### CMMLoc: Advancing Text - to - PointCloud Localization with Cauchy - Mixture - Model Based Framework
- Link：[https://arxiv.org/pdf/2503.02593](https://arxiv.org/pdf/2503.02593)
- Code：[https://github.com/kevin301342/CMMLoc](https://github.com/kevin301342/CMMLoc)


<a name="SSL"></a>
# 自监督学习(Self-supervised Learning)

#### Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach
#### Link：[https://arxiv.org/pdf/2502.19691](https://arxiv.org/pdf/2502.19691)
#### Code：[https://github.com/chenchenzong/EAOA](https://github.com/chenchenzong/EAOA)

<a name="FL"></a>
# 联邦学习(Federated Learning)

#### Handling Spatial - Temporal Data Heterogeneity for Federated Continual Learning via Tail Anchor
- Link：[https://arxiv.org/pdf/2412.18355](https://arxiv.org/pdf/2412.18355)

<a name="IL"></a>
# 增量学习(Incremental Learning)
#### Dual Consolidation for Pre - Trained Model - Based Domain - Incremental Learning
- Link：[https://arxiv.org/pdf/2410.00911](https://arxiv.org/pdf/2410.00911)
- Code：[https://github.com/Estrella-fugaz/CVPR25-Duct](https://github.com/Estrella-fugaz/CVPR25-Duct)


<a name="3DOD"></a>

# 3D语义分割(3D Semantic Segmentation)

<a name="Image-Editing"></a>

# 图像编辑(Image Editing)


#### OmniGuard: Hybrid Manipulation Localization via Augmented Versatile Deep Image Watermarking
- Link: [https://arxiv.org/pdf/2412.01615](https://arxiv.org/pdf/2412.01615)


#### SCSA: A Plug - and - Play Semantic Continuous - Sparse Attention for Arbitrary Semantic Style Transfer
- Link：[https://arxiv.org/pdf/2503.04119](https://arxiv.org/pdf/2503.04119)
- Code：[https://github.com/scn-00/SCSA](https://github.com/scn-00/SCSA)

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




<a name="SR"></a>

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

#### GEN3C: 3D - Informed World - Consistent Video Generation with Precise Camera Control
- Link：https://arxiv.org/pdf/2503.03751
- Code：[https://research.nvidia.com/labs/toronto-ai/GEN3C/](https://research.nvidia.com/labs/toronto-ai/GEN3C/)



<a name="3D-Generation"></a>
# 3D生成

#### InsTaG: Learning Personalized 3D Talking Head from Few-Second Video
- Link：[https://arxiv.org/pdf/2502.20387](https://arxiv.org/pdf/2502.20387)
- Code：[https://fictionarry.github.io/InsTaG/](https://fictionarry.github.io/InsTaG/)

#### CADDreamer: CAD object Generation from Single-view Images
- Link：[https://arxiv.org/abs/2502.20732](https://arxiv.org/abs/2502.20732)

#### StdGEN: Semantic - Decomposed 3D Character Generation from Single Images
- Link: [https://arxiv.org/pdf/2411.05738](https://arxiv.org/pdf/2411.05738)
- Code: [https://stdgen.github.io/](https://stdgen.github.io/)


<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)





<a name="CL"></a>

# 持续学习(Continual Learning)

#### Synthetic Data is an Elegant GIFT for Continual Vision - Language Models
- Link: [https://arxiv.org/pdf/2503.04229](https://arxiv.org/pdf/2503.04229)
- Code：[https://github.com/Luo-Jiaming/GIFT_CL](https://github.com/Luo-Jiaming/GIFT_CL)
- 
#### Solving the Catastrophic Forgetting Problem in Generalized Category Discovery
- Link：[https://arxiv.org/pdf/2501.05272](https://arxiv.org/pdf/2501.05272)
- Code：[https://github.com/Cliffia123/LegoGCD](https://github.com/Cliffia123/LegoGCD)

<a name="Action-Recognition"></a>

# 行为识别(Action Recognition)

#### TIMotion: Temporal and Interactive Framework for Efficient Human-Human Motion Generation
- Link：[https://arxiv.org/abs/2408.17135](https://arxiv.org/abs/2408.17135)
- Code：[https://aigc-explorer.github.io/TIMotion-page](https://aigc-explorer.github.io/TIMotion-page)

<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)

#### MonoTAKD: Teaching Assistant Knowledge Distillation for Monocular 3D Object Detection
- Link：[https://arxiv.org/pdf/2404.04910](https://arxiv.org/pdf/2404.04910)
- Code：[https://github.com/hoiliu-0801/MonoTAKD](https://github.com/hoiliu-0801/MonoTAKD)

#### Temporal Separation with Entropy Regularization for Knowledge Distillation in Spiking Neural Networks
- Link：[https://arxiv.org/pdf/2503.03144](https://arxiv.org/pdf/2503.03144)


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
- Link：[https://arxiv.org/pdf/2502.19781](https://arxiv.org/pdf/2502.19781)

#### Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis
- Link：[https://arxiv.org/pdf/2503.03132](https://arxiv.org/pdf/2503.03132)
- Code：[https://4d-dsns.github.io/DSNS/](https://4d-dsns.github.io/DSNS/)


<a name="IQA"></a>

# 图像质量评价(Image Quality Assessment)


#### Q - Eval - 100K: Evaluating Visual Quality and Alignment Level for Text - to - Vision Content
- Link：[https://arxiv.org/pdf/2503.02357](https://arxiv.org/pdf/2503.02357)
- Code：https://github.com/zzc-1998/Q-Eval


<a name="Video-Quality-Assessment"></a>

# 视频质量评价(Video Quality Assessment)

<a name="Datasets"></a>

# 数据集(Datasets)

#### EEE-Bench: A Comprehensive Multimodal Electrical And Electronics Engineering Benchmark
- Link：[https://arxiv.org/pdf/2411.01492](https://arxiv.org/pdf/2411.01492)

#### Fish-Vista: A Multi-Purpose Dataset for Understanding & Identification of Traits from Images
- Link：[https://arxiv.org/pdf/2407.08027](https://arxiv.org/pdf/2407.08027)

#### SMTPD: A New Benchmark for Temporal Prediction of Social Media Popularity
- Link: [https://arxiv.org/abs/2503.04446](https://arxiv.org/abs/2503.04446)
- Code: [https://github.com/zhuwei321/SMTPD](https://github.com/zhuwei321/SMTPD)

#### HyperPose: Hypernetwork - Infused Camera Pose Localization and an Extended Cambridge Landmarks Dataset
- Link：[https://arxiv.org/pdf/2303.02610](https://arxiv.org/pdf/2303.02610)
- Code：[https://ronferens.github.io/hyperpose/](https://ronferens.github.io/hyperpose/)

#### HarmonySet: A Comprehensive Dataset for Understanding Video - Music Semantic Alignment and Temporal Synchronization
- Link：[https://arxiv.org/pdf/2503.01725](https://arxiv.org/pdf/2503.01725)
- Code：[https://harmonyset.github.io/](https://harmonyset.github.io/)



<a name="Unlearning"></a>
# 反学习(Machine Unlearning)



<a name="New-Tasks"></a>
# 新任务(New Tasks)


<a name="Improving-Reasoning"></a>
# 模型加速(Improving Reasoning)



<a name="Time-Series"></a>
# 时间序列(Time Series)


<a name="SNN"></a>

# 脉冲网络
#### Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients
- Link：[https://arxiv.org/pdf/2503.03272](https://arxiv.org/pdf/2503.03272)
- Code：[https://github.com/ryime/PDSG-SDA](https://github.com/ryime/PDSG-SDA)

#### Inference - Scale Complexity in ANN - SNN Conversion for High - Performance and Low - Power Applications
- Link：[https://arxiv.org/pdf/2409.03368](https://arxiv.org/pdf/2409.03368)
- Code：https://github.com/putshua/Inference-scale-ANN-SNN


#### STAA - SNN: Spatial - Temporal Attention Aggregator for Spiking Neural Networks
- Link：[https://arxiv.org/pdf/2503.02689](https://arxiv.org/pdf/2503.02689)

<a name="IRetrieval"></a>
# 图像检索
#### A Comprehensive Survey on Composed Image Retrieval
- Link: [https://arxiv.org/pdf/2502.18495](https://arxiv.org/pdf/2502.18495)
- Code: [https://github.com/haokunwen/Awesome-Composed-Image-Retrieval](https://github.com/haokunwen/Awesome-Composed-Image-Retrieval)



# 其他(Others)

#### Decoder Gradient Shield: Provable and High-Fidelity Prevention of Gradient-Based Box-Free Watermark Removal
- Link：[https://arxiv.org/abs/2502.20924](https://arxiv.org/abs/2502.20924)


#### DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations
- Link：[https://arxiv.org/abs/2502.06029](https://arxiv.org/abs/2502.06029)
- Code：[https://github.com/ipsitmantri/DiTASK](https://github.com/ipsitmantri/DiTASK)


#### Full - DoF Egomotion Estimation for Event Cameras Using Geometric Solvers
- Link：[https://arxiv.org/pdf/2503.03307](https://arxiv.org/pdf/2503.03307)
- Code：[https://github.com/jizhaox/relpose-event](https://github.com/jizhaox/relpose-event)


#### Detecting Adversarial Data using Perturbation Forgery
- Link：[https://arxiv.org/pdf/2405.16226](https://arxiv.org/pdf/2405.16226)
- Code：[https://github.com/cc13qq/PFD](https://github.com/cc13qq/PFD)

#### PIDLoc: Cross - View Pose Optimization Network Inspired by PID Controllers
- Link：[https://arxiv.org/pdf/2503.02388](https://arxiv.org/pdf/2503.02388)
