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


| [3DGS(Gaussian Splatting)](#3DGS) | [Mamba / (SSM)](#Mamba) | [Avatars](#Avatars) | [Backbone](#Backbone) | [CLIP](#CLIP) | [MAE](#MAE) |[联邦学习(Federated Learning)](#FL) |  
|-------|-------|-------| --------|--------|--------|--------|
| [多模态大语言模型(MLLM)](#MLLM) | [大语言模型(LLM)](#LLM) | [视觉语言模型(VLM)](#VLM) | [多模态(Multi-modal)](#multimodal)  | [NAS](#NAS)   |  [OCR](#OCR)  |  [NeRF](#NeRF)  |   
| [视觉问答(Visual Question Answering)](#VQA) | [强化学习(Reinforcement Learning)](#RL) | [扩散模型(Diffusion Models)](#Diffusion) |  [ReID(重识别)](#ReID) |  [长尾分布(Long-Tail)](#Long-Tail) | [长尾分布(Long-Tail)](#Long-Tail)   |  [长尾分布(Long-Tail)](#Long-Tail)  |   
|[增量学习(Incremental Learning)](#IL) |[数据增强(Data Augmentation)](#DA) | [目标检测(Object Detection)](#Object-Detection)|[异常检测(Anomaly Detection)](#Anomaly-Detection) | [目标跟踪(Visual Tracking)](#VT)|[语义分割(Semantic Segmentation)](#Semantic-Segmentation) | [实例分割(Instance Segmentation)](#Instance-Segmentation)| 
|[医学图像(Medical Image)](#MI) |[医学图像分割(Medical Image Segmentation)](#MIS) |[视频目标分割(Video Object Segmentation)](#VOS) |[视频实例分割(Video Instance Segmentation)](#VIS) | [参考图像分割(Referring Image Segmentation)](#RIS) |  [图像抠图(Image Matting)](#Matting)| [图像编辑(Image Editing)](#Image-Editing)|
|[具身智能](Embodied-AI)|[Prompt](#Prompt) | [自监督学习(Self-supervised Learning)](#SSL)   |  [生物工程(bioengineering)](#bio)| [Low-level Vision](#LLV)|[超分辨率(Super-Resolution)](#SR) |[去模糊(Deblur)](#Deblur)|
|[生成对抗网络(GAN)](#GAN) |[3D点云(3D Point Cloud)](#3D-Point-Cloud) |[3D目标检测(3D Object Detection)](#3DOD) | [3D语义分割(3D Semantic Segmentation)](#3DSS)|[3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking) |[3D语义场景补全(3D Semantic Scene Completion)](#3DSSC) |[视频理解(Video Understanding)](#Video-Understanding)|
|[3D人体姿态估计(3D Human Pose Estimation)](#3D-Human-Pose-Estimation) |[3D人体Mesh估计(3D Human Mesh Estimation)](#3D-Human-Pose-Estimation) | [少样本学习(Few-Shot Learning)](#FewShot)| [图像生成(Image Generation)](#Image-Generation)|[视频生成(Video Generation)](#Video-Generation) |[3D生成(3D Generation)](#3D-Generation) | [图像压缩(Image Compression)](#IC)|
|[持续学习(Continual Learning)](#CL) |[行为识别(Action Recognition)](#Action-Recognition) | [行为检测(Action Detection)](#Action-Detection)|[人脸识别(Face Recognition)](#face-recognition) |[文本检测(Text Detection)](#Text-Detection) | [知识蒸馏(Knowledge Distillation)](#KD)|[三维重建(3D Reconstruction)](#3D-Reconstruction)
| [GNN](#GNN) | [DETR](#DETR)  |  [Vision Transformer](#Vision-Transformer) |[全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)| [去噪(Denoising)](#Denoising) |[自动驾驶(Autonomous Driving)](#Autonomous-Driving)| [3D配准(3D Registration)](#3D-Registration) | 
| [模型剪枝(Model Pruning)](#Pruning) |[深度估计(Depth Estimation)](#Depth-Estimation) |[轨迹预测(Trajectory Prediction)](#TP) |[车道线检测(Lane Detection)](#Lane-Detection) |[图像描述(Image Captioning)](#Image-Captioning) | [手语识别(Sign Language Recognition)](#SLR)|[视频预测(Video Prediction)](#Video-Prediction)  | 
|[新视点合成(Novel View Synthesis)](#NVS) |[Zero-Shot Learning(零样本学习)](#ZSL) |[立体匹配(Stereo Matching)](#Stereo-Matching) | [特征匹配(Feature Matching)](#Feature-Matching)| [场景图生成(Scene Graph Generation)](#SGG) |  [计数(Counting)](#Counting)|[隐式神经表示(Implicit Neural Representations)](#INR) | 
|[图像质量评价(Image Quality Assessment)](#IQA) |[视频质量评价(Video Quality Assessment)](#Video-Quality-Assessment) |[数据集(Datasets)](#Datasets) |[反学习(Machine Unlearning)](#Unlearning) |[新任务(New Tasks)](#New-Tasks) |[模型加速(Improving Reasoning)](#Improving-Reasoning) |[时间序列(Time Series)](#Time-Series) | 
|[其他(Others)](#Others) |[脉冲网络](#SNN) |[图像检索](#IRetrieval) | | | | | 



<a name="EmAI"></a>
# 具身智能（Embodied AI）

#### Universal Actions for Enhanced Embodied Foundation Models
- Link：[https://arxiv.org/pdf/2501.10105](https://arxiv.org/pdf/2501.10105)
- Code：[https://github.com/2toinf/UniAct](https://github.com/2toinf/UniAct)

#### A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning
- Link：[https://arxiv.org/pdf/2503.06960](https://arxiv.org/pdf/2503.06960)
- Code：[https://github.com/CVMI-Lab/SlotMIM](https://github.com/CVMI-Lab/SlotMIM)

#### VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation
- Link：[https://arxiv.org/pdf/2503.07135](https://arxiv.org/pdf/2503.07135)
- Code：[https://hanzhic.github.io/vidbot-project/](https://hanzhic.github.io/vidbot-project/)

#### InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions
- Link：[https://arxiv.org/pdf/2502.20390](https://arxiv.org/pdf/2502.20390)
- Code：[https://sirui-xu.github.io/InterMimic/](https://sirui-xu.github.io/InterMimic/)

#### UniGoal: Towards Universal Zero-shot Goal-oriented Navigation
- Link：[https://arxiv.org/pdf/2503.10630](https://arxiv.org/pdf/2503.10630)


#### DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness
- Link：[https://arxiv.org/pdf/2503.08257](https://arxiv.org/pdf/2503.08257)
- Code：[https://github.com/4DVLab/DexGrasp-Anything](https://github.com/4DVLab/DexGrasp-Anything)

#### Optimus - 2: Multimodal Minecraft Agent with Goal - Observation - Action Conditioned Policy
- Link：[https://arxiv.org/pdf/2502.19902](https://arxiv.org/pdf/2502.19902)
- Code：[https://cybertronagent.github.io/Optimus-2.github.io/](https://cybertronagent.github.io/Optimus-2.github.io/)

#### Reasoning in visual navigation of end-to-end trained agents: a dynamical systems approach
- Link：[https://europe.naverlabs.com/research/publications/reasoning-in-visual-navigation-of-end-to-end-trained-agents/](https://europe.naverlabs.com/research/publications/reasoning-in-visual-navigation-of-end-to-end-trained-agents/)
- Code：[https://europe.naverlabs.com/research/publications/reasoning-in-visual-navigation-of-end-to-end-trained-agents/](https://europe.naverlabs.com/research/publications/reasoning-in-visual-navigation-of-end-to-end-trained-agents/)


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

#### DirectTriGS: Triplane-based Gaussian Splatting Field Representation for 3D Generation
- Link：[https://arxiv.org/pdf/2503.06900](https://arxiv.org/pdf/2503.06900)


#### SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting
- Link：[https://arxiv.org/pdf/2503.07476](https://arxiv.org/pdf/2503.07476)


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


#### 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models
- Link：[https://arxiv.org/pdf/2503.10437](https://arxiv.org/pdf/2503.10437)
- Code：[https://4d-langsplat.github.io/](https://4d-langsplat.github.io/)

#### GaussHDR: High Dynamic Range Gaussian Splatting via Learning Unified 3D and 2D Local Tone Mapping
- Link：[https://arxiv.org/pdf/2503.10143](https://arxiv.org/pdf/2503.10143)
- Code：[https://liujf1226.github.io/GaussHDR](https://liujf1226.github.io/GaussHDR)

#### DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting
- Link：[https://arxiv.org/pdf/2503.00746](https://arxiv.org/pdf/2503.00746)
- Code：[https://dof-gaussian.github.io](https://dof-gaussian.github.io)

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


#### MUSt3R: Multi-view Network for Stereo 3D Reconstruction
- Link：[https://arxiv.org/pdf/2503.01661](https://arxiv.org/pdf/2503.01661)
- Code：[https://github.com/naver/must3r](https://github.com/naver/must3r)

#### FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views
- Link：[https://arxiv.org/pdf/2502.12138](https://arxiv.org/pdf/2502.12138)
- Code：[https://zhanghe3z.github.io/FLARE/](https://zhanghe3z.github.io/FLARE/)

<a name="Pruning"></a>
# 模型剪枝(Model Pruning)
#### COAP: Memory-Efficient Training with Correlation-Aware Gradient Projection
- Link：[https://arxiv.org/pdf/2412.00071](https://arxiv.org/pdf/2412.00071)
- Code：[https://byteaigc.github.io/coap/](https://byteaigc.github.io/coap/)

<a name="Depth-Estimation"></a>
# 深度估计(Depth Estimation)

#### DepthCues: Evaluating Monocular Depth Perception in Large Vision Models
- Link：[https://arxiv.org/pdf/2411.17385](https://arxiv.org/pdf/2411.17385)
- Code：[https://danier97.github.io/depthcues](https://danier97.github.io/depthcues)

#### Completion as Enhancement: A Degradation-Aware Selective Image Guided Network for Depth Completion
- Link：[https://arxiv.org/pdf/2412.19225](https://arxiv.org/pdf/2412.19225)


#### SVDC: Consistent Direct Time-of-Flight Video Depth Completion with Frequency Selective Fusion
- Link：[https://arxiv.org/pdf/2503.01257](https://arxiv.org/pdf/2503.01257)
- Code：[https://github.com/Lan1eve/SVDC](https://github.com/Lan1eve/SVDC)

<a name="TP"></a>
# 轨迹预测(Trajectory Prediction)

#### MoFlow: One-Step Flow Matching for Human Trajectory Forecasting via Implicit Maximum Likelihood Estimation based Distillation
- Link：[https://arxiv.org/pdf/2503.09950](https://arxiv.org/pdf/2503.09950)
- Code：[https://moflow-imle.github.io](https://moflow-imle.github.io)


<a name="Mamba"></a>

# Mamba / SSM

#### MAMBA4D: Efficient Long-Sequence Point Cloud Video Understanding with Disentangled Spatial-Temporal State Space Models
- Link：[https://arxiv.org/pdf/2405.14338](https://arxiv.org/pdf/2405.14338)
- Code：[https://github.com/IRMVLab/Mamba4D](https://github.com/IRMVLab/Mamba4D)

#### JamMa: Ultra - lightweight Local Feature Matching with Joint Mamba
- Link：[https://arxiv.org/pdf/2503.03437](https://arxiv.org/pdf/2503.03437)
- Code：https://leoluxxx.github.io/JamMa-page/

#### QMambaBSR: Burst Image Super-Resolution with Query State Space Model
- Link：[https://arxiv.org/pdf/2408.08665](https://arxiv.org/pdf/2408.08665)
- Code：
- 
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


#### SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment
- Link：[https://arxiv.org/pdf/2503.09594](https://arxiv.org/pdf/2503.09594)

#### UniScene: Unified Occupancy-centric Driving Scene Generation
- Link：[https://arxiv.org/pdf/2412.05435](https://arxiv.org/pdf/2412.05435)
- Code：[https://arlo0o.github.io/uniscene/](https://arlo0o.github.io/uniscene/)

<a name="Backbone"></a>

# Backbone

#### OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels
- Link：[https://arxiv.org/pdf/2502.20087](https://arxiv.org/pdf/2502.20087)
- Code：[https://bit.ly/4bdbmdl](https://bit.ly/4bdbmdl)

#### Transformers without Normalization
- Link：[https://arxiv.org/pdf/2503.10622](https://arxiv.org/pdf/2503.10622)
- Code：[https://jiachenzhu.github.io/DyT/](https://jiachenzhu.github.io/DyT/)

#### ShiftwiseConv: Small Convolutional Kernel with Large Kernel Effect
- Link：[https://arxiv.org/pdf/2401.12736](https://arxiv.org/pdf/2401.12736)
- Code：[https://github.com/lidc54/shift-wiseConv](https://github.com/lidc54/shift-wiseConv)


<a name="CLIP"></a>

# CLIP

#### CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation
- Link：[https://arxiv.org/pdf/2502.19842](https://arxiv.org/pdf/2502.19842)
- Code：[https://clip-analysis.github.io/](https://clip-analysis.github.io/)

#### CLIP is Strong Enough to Fight Back: Test - time Counterattacks towards Zero - shot Adversarial Robustness of CLIP
- Link：[https://arxiv.org/pdf/2503.03613](https://arxiv.org/pdf/2503.03613)
- Code：[https://github.com/Sxing2/CLIP-Test-time-Counterattacks](https://github.com/Sxing2/CLIP-Test-time-Counterattacks)

#### AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP
- Link：[https://arxiv.org/pdf/2503.06661](https://arxiv.org/pdf/2503.06661)
- Code：[https://github.com/Mwxinnn/AA-CLIP](https://github.com/Mwxinnn/AA-CLIP)
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

#### Reasoning to Attend: Try to Understand How <SEG> Token Works
- Link：[https://arxiv.org/pdf/2412.17741](https://arxiv.org/pdf/2412.17741)
- Code：[https://github.com/rui-qian/READ](https://github.com/rui-qian/READ)

#### ProAPO: Progressively Automatic Prompt Optimization for Visual Classification
- Link：[https://arxiv.org/pdf/2502.19844](https://arxiv.org/pdf/2502.19844)
- Code：[https://github.com/MorningStarOvO/ProAPO](https://github.com/MorningStarOvO/ProAPO)
<a name="LLM"></a>
# 大语言模型(LLM)



<a name="VLM"></a>
# 视觉语言模型(LLM)

#### GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks
- Link：[https://arxiv.org/pdf/2503.06514](https://arxiv.org/pdf/2503.06514)


#### Filter Images First, Generate Instructions Later: Pre-Instruction Data Selection for Visual Instruction Tuning
- Link：[https://arxiv.org/pdf/2503.07591](https://arxiv.org/pdf/2503.07591)
- Code：[https://bardisafa.github.io/PreSel](https://bardisafa.github.io/PreSel)

#### PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction
- Link：[https://arxiv.org/pdf/2410.17247](https://arxiv.org/pdf/2410.17247)
- Code：[https://github.com/Cooperx521/PyramidDrop](https://github.com/Cooperx521/PyramidDrop)

#### Reasoning to Attend: Try to Understand How <SEG> Token Work
- Link：[https://arxiv.org/pdf/2412.17741](https://arxiv.org/pdf/2412.17741)
- Code：[https://github.com/rui-qian/READ](https://github.com/rui-qian/READ)

#### Words or Vision: Do Vision-Language Models Have Blind Faith in Text?
- Link：[https://arxiv.org/pdf/2503.02199](https://arxiv.org/pdf/2503.02199)


#### Recurrence-Enhanced Vision-and-Language Transformers for Robust Multimodal Document Retrieval
- Link：[https://arxiv.org/pdf/2503.01980](https://arxiv.org/pdf/2503.01980)
- Code：[https://github.com/aimagelab/ReT](https://github.com/aimagelab/ReT)

#### Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key
- Link：[https://arxiv.org/pdf/2501.09695](https://arxiv.org/pdf/2501.09695)
- Code：[https://opa-dpo.github.io/](https://opa-dpo.github.io/)

#### LayoutVLM: Differentiable Optimization of 3D Layout via Vision - Language Models
- Link：[https://arxiv.org/pdf/2412.02193](https://arxiv.org/pdf/2412.02193)
- Code：[https://ai.stanford.edu/~sunfanyun/layoutvlm/](https://ai.stanford.edu/~sunfanyun/layoutvlm/)

<a name="MLLM"></a>

# 多模态大语言模型(MLLM)

#### Escaping Plato's Cave: Towards the Alignment of 3D and Text Latent Spaces
- Link：[https://arxiv.org/pdf/2503.05283](https://arxiv.org/pdf/2503.05283)




#### Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy
- Link：[https://arxiv.org/pdf/2502.19902](https://arxiv.org/pdf/2502.19902)

#### LLaVA-Critic: Learning to Evaluate Multimodal Models
- Link：[https://arxiv.org/pdf/2410.02712](https://arxiv.org/pdf/2410.02712)
- Code：[https://llava-vl.github.io/blog/2024-10-03-llava-critic](https://llava-vl.github.io/blog/2024-10-03-llava-critic)

#### AesthetiQ: Enhancing Graphic Layout Design via Aesthetic-Aware Preference Alignment of Multi-modal Large Language Models
- Link：[https://arxiv.org/pdf/2503.00591](https://arxiv.org/pdf/2503.00591)

#### EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions
- Link：[https://arxiv.org/pdf/2409.18042](https://arxiv.org/pdf/2409.18042)
- Code：[https://emova-ollm.github.io/](https://emova-ollm.github.io/)

#### DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation
- Link：[https://arxiv.org/pdf/2412.07589](https://arxiv.org/pdf/2412.07589)
- Code：[https://jianzongwu.github.io/projects/diffsensei/](https://jianzongwu.github.io/projects/diffsensei/)

#### Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training
- Link：[https://arxiv.org/pdf/2410.08202](https://arxiv.org/pdf/2410.08202)
- Code：[https://github.com/OpenGVLab/Mono-InternVL](https://github.com/OpenGVLab/Mono-InternVL)

<a name="multimodal"></a>

# 多模态

#### Towards Open - Vocabulary Audio - Visual Event Localization
- Link：[https://arxiv.org/pdf/2411.11278](https://arxiv.org/pdf/2411.11278)
- Code：[https://github.com/jasongief/OV-AVEL](https://github.com/jasongief/OV-AVEL)

#### Do computer vision foundation models learn the low-level characteristics of the human visual system?
- Link：[https://arxiv.org/pdf/2502.20256](https://arxiv.org/pdf/2502.20256)

#### Human Motion Instruction Tuning
- Link：[https://arxiv.org/pdf/2411.16805](https://arxiv.org/pdf/2411.16805)
- Code：[https://github.com/ILGLJ/LLaMo](https://github.com/ILGLJ/LLaMo)

#### Knowledge Bridger: Towards Training-free Missing Multi-modality Completion
- Link：[https://arxiv.org/pdf/2502.19834](https://arxiv.org/pdf/2502.19834)

#### MMRL: Multi-Modal Representation Learning for Vision-Language Models
- Link：[https://arxiv.org/pdf/2503.08497](https://arxiv.org/pdf/2503.08497)
- Code：[https://github.com/yunncheng/MMRL](https://github.com/yunncheng/MMRL)

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

#### V^2Dial: Unification of Video and Visual Dialog via Multimodal Experts
- Link：[https://arxiv.org/pdf/2503.02063](https://arxiv.org/pdf/2503.02063)
- Code：[https://www.collaborative-ai.org/publications/abdessaied25_cvpr/](https://www.collaborative-ai.org/publications/abdessaied25_cvpr/)

#### WeGen: A Unified Model for Interactive Multimodal Generation as We Chat
- Link：[https://arxiv.org/pdf/2503.01115](https://arxiv.org/pdf/2503.01115)
- Code：[https://github.com/hzphzp/WeGen](https://github.com/hzphzp/WeGen)

#### DynRefer: Delving into Region-level Multimodal Tasks via Dynamic Resolution
- Link：[https://arxiv.org/pdf/2405.16071](https://arxiv.org/pdf/2405.16071)
- Code：[https://github.com/callsys/DynRefer](https://github.com/callsys/DynRefer)

#### NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval
- Link：[https://arxiv.org/pdf/2503.10526](https://arxiv.org/pdf/2503.10526)
- Code：[https://github.com/zzezze/NeighborRetr](https://github.com/zzezze/NeighborRetr)

#### STiL: Semi-supervised Tabular-Image Learning for Comprehensive Task-Relevant Information Exploration in Multimodal Classification
- Link：[https://arxiv.org/pdf/2503.06277](https://arxiv.org/pdf/2503.06277)
- Code：[https://github.com/siyi-wind/STiL](https://github.com/siyi-wind/STiL)






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

#### AirRoom: Objects Matter in Room Reidentification
- Link：[https://arxiv.org/pdf/2503.01130](https://arxiv.org/pdf/2503.01130)
- Code：[https://sairlab.org/airroom/](https://sairlab.org/airroom/)

#### Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification
- Link：[https://arxiv.org/pdf/2503.09962](https://arxiv.org/pdf/2503.09962)
- Code：[https://github.com/sssaury/HAM](https://github.com/sssaury/HAM)

#### AG - VPReID: A Challenging Large - Scale Benchmark for Aerial - Ground Video - based Person Re - Identification
- Link：[https://arxiv.org/pdf/2503.08121](https://arxiv.org/pdf/2503.08121)
- Code：[https://github.com/agvpreid25/AG-VPReID-Net](https://github.com/agvpreid25/AG-VPReID-Net)

<a name="Long-Tail"></a>
# 长尾分布(Long-Tail)
#### Improve Representation for Imbalanced Regression through Geometric Constraints
- Link：[https://arxiv.org/pdf/2503.00876](https://arxiv.org/pdf/2503.00876)
- Code：[https://github.com/yilei-wu/imbalanced-regression](https://github.com/yilei-wu/imbalanced-regression)

#### Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness
- Link：[https://arxiv.org/pdf/2503.09487](https://arxiv.org/pdf/2503.09487)



<a name="Diffusion"></a>

# 扩散模型(Diffusion Models)

#### PTDiffusion: Free Lunch for Generating Optical Illusion Hidden Pictures with Phase-Transferred Diffusion Model
- Link：https://arxiv.org/pdf/2503.06186


#### Adversarial Diffusion Compression for Real-World Image Super-Resolution
- Link：[https://arxiv.org/pdf/2411.13383](https://arxiv.org/pdf/2411.13383)
- Code：[https://github.com/Guaishou74851/AdcSR](https://github.com/Guaishou74851/AdcSR)

#### Color Alignment in Diffusion
- Link：[https://arxiv.org/pdf/2503.06746](https://arxiv.org/pdf/2503.06746)


#### Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways
- Link：[https://arxiv.org/pdf/2503.07026](https://arxiv.org/pdf/2503.07026)

#### RayFlow: Instance-Aware Diffusion Acceleration via Adaptive Flow Trajectories
- Link：[https://arxiv.org/pdf/2503.07699](https://arxiv.org/pdf/2503.07699)


#### Attention Distillation: A Unified Approach to Visual Characteristics Transfer
- Link：[https://arxiv.org/pdf/2502.20235](https://arxiv.org/pdf/2502.20235)
- Code：https://github.com/xugao97/AttentionDistillation

#### Data-free Universal Adversarial Perturbation with Pseudo-semantic Prior
- Link: [https://arxiv.org/abs/2502.21048](https://arxiv.org/abs/2502.21048)

#### Optimizing for the Shortest Path in Denoising Diffusion Model
- Link：[https://arxiv.org/pdf/2503.03265](https://arxiv.org/pdf/2503.03265)
- Code：[https://github.com/UnicomAI/ShortDF](https://github.com/UnicomAI/ShortDF)

#### h-Edit: Effective and Flexible Diffusion-Based Editing via Doob's h-Transform
- Link：[https://arxiv.org/pdf/2503.02187](https://arxiv.org/pdf/2503.02187)
- Code：[https://github.com/nktoan/h-edit](https://github.com/nktoan/h-edit)

#### Silent Branding Attack: Trigger-free Data Poisoning Attack on Text-to-Image Diffusion Models
- Link：[https://arxiv.org/pdf/2503.09669](https://arxiv.org/pdf/2503.09669)
- Code：[https://silent-branding.github.io/](https://silent-branding.github.io/)

#### Denoising Functional Maps: Diffusion Models for Shape Correspondence
- Link：[https://arxiv.org/pdf/2503.01845](https://arxiv.org/pdf/2503.01845)
- Code：[https://alekseizhuravlev.github.io/denoising-functional-maps/](https://alekseizhuravlev.github.io/denoising-functional-maps/)

#### Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models
- Link：[https://arxiv.org/pdf/2503.01774](https://arxiv.org/pdf/2503.01774)
- Code：[https://research.nvidia.com/labs/toronto-ai/difix3d](https://research.nvidia.com/labs/toronto-ai/difix3d)

#### DesignDiffusion: High-Quality Text-to-Design Image Generation with Diffusion Models
- Link：[https://arxiv.org/pdf/2503.01645](https://arxiv.org/pdf/2503.01645)

#### CacheQuant: Comprehensively Accelerated Diffusion Models
- Link：[https://arxiv.org/pdf/2503.01323](https://arxiv.org/pdf/2503.01323)
- Code：[https://github.com/BienLuky/CacheQuant](https://github.com/BienLuky/CacheQuant)

#### Reconciling Stochastic and Deterministic Strategies for Zero-shot Image Restoration using Diffusion Model in Dual
- Link：[https://arxiv.org/pdf/2503.01288](https://arxiv.org/pdf/2503.01288)
- Code：[https://github.com/ChongWang1024/RDMD](https://github.com/ChongWang1024/RDMD)

#### DifIISR: A Diffusion Model with Gradient Guidance for Infrared Image Super-Resolution
- Link：[https://arxiv.org/pdf/2503.01187](https://arxiv.org/pdf/2503.01187)
- Code：[https://github.com/zirui0625/DifIISR](https://github.com/zirui0625/DifIISR)

#### Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes
- Link：[https://arxiv.org/pdf/2503.09993](https://arxiv.org/pdf/2503.09993)


#### Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models
- Link：[https://arxiv.org/pdf/2412.15341](https://arxiv.org/pdf/2412.15341)
- Code：[https://github.com/rezashkv/unlearn-ft](https://github.com/rezashkv/unlearn-ft)

#### Curriculum Direct Preference Optimization for Diffusion and Consistency Models
- Link：[https://arxiv.org/pdf/2405.13637](https://arxiv.org/pdf/2405.13637)
- Code：[https://github.com/CroitoruAlin/Curriculum-DPO](https://github.com/CroitoruAlin/Curriculum-DPO)

#### Reward Fine - Tuning Two - Step Diffusion Models via Learning Differentiable Latent - Space Surrogate Reward
- Link：[https://arxiv.org/pdf/2411.15247](https://arxiv.org/pdf/2411.15247)
- Code：[https://sites.google.com/view/lasro](https://sites.google.com/view/lasro)

<a name="Vision-Transformer"></a>

# Vision Transformer
#### Split Adaptation for Pre-trained Vision Transformers
- Link：[https://arxiv.org/pdf/2503.00441](https://arxiv.org/pdf/2503.00441)
- Code：[https://github.com/conditionWang/Split_Adaptation](https://github.com/conditionWang/Split_Adaptation)

#### Associative Transformer
- Link：[https://arxiv.org/pdf/2309.12862](https://arxiv.org/pdf/2309.12862)


<a name="VL"></a>

# 视觉和语言(Vision-Language)



<a name="Object-Detection"></a>

# 目标检测(Object Detection)

#### ReRAW: RGB - to - RAW Image Reconstruction via Stratified Sampling for Efficient Object Detection on the Edge
- Link：[https://arxiv.org/pdf/2503.03782](https://arxiv.org/pdf/2503.03782)
- Code：[https://anonymous.4open.science/r/ReRAW-0C87/](https://anonymous.4open.science/r/ReRAW-0C87/)
  
#### SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection
- Link：[https://arxiv.org/pdf/2503.00414](https://arxiv.org/pdf/2503.00414)
- Code：[https://github.com/Phil0212/SGC-Net](https://github.com/Phil0212/SGC-Net)

#### Solving Instance Detection from an Open-World Perspective
- Link：[https://arxiv.org/pdf/2503.00359](https://arxiv.org/pdf/2503.00359)
- Code：[https://shenqq377.github.io/IDOW/](https://shenqq377.github.io/IDOW/)
<a name="DA"></a>

## 数据增强(Data Augmentation)




<a name="Anomaly-Detection"></a>

# 异常检测(Anomaly Detection)

#### UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection
- Link：[https://arxiv.org/pdf/2412.03342](https://arxiv.org/pdf/2412.03342)
- Code：[https://github.com/FantasticGNU/UniVAD](https://github.com/FantasticGNU/UniVAD)

#### Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection
- Link: [https://arxiv.org/abs/2502.20981](https://arxiv.org/abs/2502.20981)

#### Towards Visual Discrimination and Reasoning of Real - World Physical Dynamics: Physics - Grounded Anomaly Detection
- Link：[https://arxiv.org/pdf/2503.03562](https://arxiv.org/pdf/2503.03562)

#### CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature Leveraging
- Link：[https://arxiv.org/pdf/2503.00325](https://arxiv.org/pdf/2503.00325)


<a name="VT"></a>

# 目标跟踪(Object Tracking)

####  Omnidirectional Multi - Object Tracking
- Link：[https://arxiv.org/abs/2503.04565](https://arxiv.org/abs/2503.04565)
- Code：[https://github.com/xifen523/OmniTrack](https://github.com/xifen523/OmniTrack)




<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

#### Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation
- Link：[https://arxiv.org/pdf/2412.04533](https://arxiv.org/pdf/2412.04533)
- Code：[https://github.com/hustvl/MaskAdapter](https://github.com/hustvl/MaskAdapter)

#### Towards Generalizable Scene Change Detection
- Link：[https://arxiv.org/abs/2409.06214](https://arxiv.org/abs/2409.06214)





<a name="Instance-Segmentation"></a>
# 实例分割(Instance Segmentation)
#### Audio-Visual Instance Segmentation
- Link：[https://arxiv.org/pdf/2310.18709](https://arxiv.org/pdf/2310.18709)
- Code：[https://github.com/ruohaoguo/avis](https://github.com/ruohaoguo/avis)


<a name="FewShot"></a>

# 少样本学习(Few-Shot Learning)

#### ProAPO: Progressively Automatic Prompt Optimization for Visual Classification
- Link：[https://arxiv.org/pdf/2502.19844](https://arxiv.org/pdf/2502.19844)
- Code：[https://github.com/MorningStarOvO/ProAPO](https://github.com/MorningStarOvO/ProAPO)



  
<a name="bio"></a>



<a name="MI"></a>

# 医学图像(Medical Image)

#### Q - PART: Quasi - Periodic Adaptive Regression with Test - time Training for Pediatric Left Ventricular Ejection Fraction Regression
- Link：[https://arxiv.org/pdf/2503.04131](https://arxiv.org/pdf/2503.04131)
- Code：[https://github.com/ljwztc/Q-PART](https://github.com/ljwztc/Q-PART)

#### MedUnifier: Unifying Vision - and - Language Pre - training on Medical Data with Vision Generation Task using Discrete Visual Representations
- Link：[https://arxiv.org/pdf/2503.01019](https://arxiv.org/pdf/2503.01019)

#### Volume Tells: Dual Cycle-Consistent Diffusion for 3D Fluorescence Microscopy De-noising and Super-Resolution
- Link：[https://arxiv.org/pdf/2503.02261](https://arxiv.org/pdf/2503.02261)
- Code：

#### EchoONE: Segmenting Multiple echocardiography Planes in One Model
- Link：[https://arxiv.org/pdf/2412.02993](https://arxiv.org/pdf/2412.02993)
- Code：[https://github.com/a2502503/EchoONE](https://github.com/a2502503/EchoONE)

#### Patient-Level Anatomy Meets Scanning-Level Physics: Personalized Federated Low-Dose CT Denoising Empowered by Large Language Model
- Link：[https://arxiv.org/pdf/2503.00908](https://arxiv.org/pdf/2503.00908)
- Code：[https://github.com/Zi-YuanYang/SCAN-PhysFed](https://github.com/Zi-YuanYang/SCAN-PhysFed)

#### DAMM-Diffusion: Learning Divergence-Aware Multi-Modal Diffusion Model for Nanoparticles Distribution Prediction
- Link：[https://arxiv.org/pdf/2503.09491](https://arxiv.org/pdf/2503.09491)
- Code：[https://github.com/JJ-ZHOU-Code/DAMM-Diffusion](https://github.com/JJ-ZHOU-Code/DAMM-Diffusion)

#### BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models
- Link：[https://arxiv.org/pdf/2411.15232](https://arxiv.org/pdf/2411.15232)
- Code：[https://github.com/HealthX-Lab/BiomedCoOp](https://github.com/HealthX-Lab/BiomedCoOp)

#### Interactive Medical Image Analysis with Concept - based Similarity Reasoning
- Link：[https://arxiv.org/pdf/2503.06873](https://arxiv.org/pdf/2503.06873)
- Code：[https://github.com/tadeephuy/InteractCSR](https://github.com/tadeephuy/InteractCSR)

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



<a name="Action-Detection"></a>
# 行为检测(Action Detection)
#### Precise Event Spotting in Sports Videos: Solving Long-Range Dependency and Class Imbalance
- Link：[https://arxiv.org/pdf/2503.00147](https://arxiv.org/pdf/2503.00147)

<a name="face-recognition"></a>

# 人脸识别(Face Recognition)

#### HSEmotion Team at ABAW - 8 Competition: Audiovisual Ambivalence/Hesitancy, Emotional Mimicry Intensity and Facial Expression Recognition
- Link：[https://arxiv.org/pdf/2503.10399](https://arxiv.org/pdf/2503.10399)



<a name="3D-Point-Cloud"></a>

# 3D点云(3D-Point-Cloud)

#### GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors
- Link：[https://arxiv.org/pdf/2502.19896](https://arxiv.org/pdf/2502.19896)

#### Floxels: Fast Unsupervised Voxel Based Scene Flow Estimation
- Link：[https://arxiv.org/pdf/2503.04718](https://arxiv.org/pdf/2503.04718)

#### Self - Supervised Large Scale Point Cloud Completion for Archaeological Site Restoration
- Link：[https://arxiv.org/pdf/2503.04030](https://arxiv.org/pdf/2503.04030)

#### GBlobs: Explicit Local Structure via Gaussian Blobs for Improved Cross-Domain LiDAR-based 3D Object Detection
- Link：[https://arxiv.org/pdf/2503.08639](https://arxiv.org/pdf/2503.08639)


#### ArcPro: Architectural Programs for Structured 3D Abstraction of Sparse Points
- Link：[https://arxiv.org/pdf/2503.02745](https://arxiv.org/pdf/2503.02745)
- Code：[https://vcc.tech/research/2025/ArcPro](https://vcc.tech/research/2025/ArcPro)

#### CMMLoc: Advancing Text - to - PointCloud Localization with Cauchy - Mixture - Model Based Framework
- Link：[https://arxiv.org/pdf/2503.02593](https://arxiv.org/pdf/2503.02593)
- Code：[https://github.com/kevin301342/CMMLoc](https://github.com/kevin301342/CMMLoc)

#### DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting
- Link：[https://arxiv.org/pdf/2503.00746](https://arxiv.org/pdf/2503.00746)
- Code：[https://dof-gaussian.github.io](https://dof-gaussian.github.io)

#### LiSu: A Dataset and Method for LiDAR Surface Normal Estimation
- Link：[https://arxiv.org/pdf/2503.08601](https://arxiv.org/pdf/2503.08601)


#### STAR-Edge: Structure-aware Local Spherical Curve Representation for Thin-walled Edge Extraction from Unstructured Point Clouds
- Link：[https://arxiv.org/pdf/2503.00801](https://arxiv.org/pdf/2503.00801)
- Code：[https://github.com/Miraclelzk/STAR-Edge](https://github.com/Miraclelzk/STAR-Edge)

#### Unlocking Generalization Power in LiDAR Point Cloud Registration
- Link：[https://arxiv.org/pdf/2503.10149](https://arxiv.org/pdf/2503.10149)
- Code：[https://github.com/peakpang/UGP](https://github.com/peakpang/UGP)

#### Parametric Point Cloud Completion for Polygonal Surface Reconstruction
- Link：[https://arxiv.org/pdf/2503.08363](https://arxiv.org/pdf/2503.08363)
- Code：[https://parametric-completion.github.io](https://parametric-completion.github.io)

#### Mitigating Ambiguities in 3D Classification with Gaussian Splatting
- Link：[https://arxiv.org/pdf/2503.08352](https://arxiv.org/pdf/2503.08352)


#### HOTFormerLoc: Hierarchical Octree Transformer for Versatile Lidar Place Recognition Across Ground and Aerial Views
- Link：[https://arxiv.org/pdf/2503.08140](https://arxiv.org/pdf/2503.08140)
- Code：[https://csiro-robotics.github.io/HOTFormerLoc](https://csiro-robotics.github.io/HOTFormerLoc)

<a name="SSL"></a>
# 自监督学习(Self-supervised Learning)

#### Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach
- Link：[https://arxiv.org/pdf/2502.19691](https://arxiv.org/pdf/2502.19691)
- Code：[https://github.com/chenchenzong/EAOA](https://github.com/chenchenzong/EAOA)

#### Do We Always Need the Simplicity Bias? Looking for Optimal Inductive Biases in the Wild
- Link：https://arxiv.org/pdf/2503.10065


<a name="bio"></a>
# 生物工程(bioengineering)

#### Electromyography-Informed Facial Expression Reconstruction for Physiological-Based Synthesis and Analysis
- Link：[https://arxiv.org/pdf/2503.09556](https://arxiv.org/pdf/2503.09556)
- Code：[https://eifer-mam.github.io/](https://eifer-mam.github.io/)



<a name="FL"></a>
# 联邦学习(Federated Learning)

#### Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning
- Link：[https://arxiv.org/pdf/2503.06457](https://arxiv.org/pdf/2503.06457)
- Code：[https://github.com/WeiDai-David/2025CVPR_GGEUR](https://github.com/WeiDai-David/2025CVPR_GGEUR)

#### Handling Spatial - Temporal Data Heterogeneity for Federated Continual Learning via Tail Anchor
- Link：[https://arxiv.org/pdf/2412.18355](https://arxiv.org/pdf/2412.18355)

#### Extrapolating and Decoupling Image-to-Video Generation Models: Motion Modeling is Easier Than You Think
- Link：[https://arxiv.org/pdf/2503.00948](https://arxiv.org/pdf/2503.00948)
- Code：[https://github.com/Chuge0335/EDG](https://github.com/Chuge0335/EDG)

#### FedBiP: Heterogeneous One-Shot Federated Learning with Personalized Latent Diffusion Models
- Link：[https://arxiv.org/pdf/2410.04810](https://arxiv.org/pdf/2410.04810)
- Code：[https://github.com/HaokunChen245/FedBiP](https://github.com/HaokunChen245/FedBiP)

#### dFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data Analysis
- Link：[https://arxiv.org/pdf/2503.10412](https://arxiv.org/pdf/2503.10412)



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

#### SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal
- Link：[https://arxiv.org/pdf/2409.07041](https://arxiv.org/pdf/2409.07041)


#### SCSA: A Plug - and - Play Semantic Continuous - Sparse Attention for Arbitrary Semantic Style Transfer
- Link：[https://arxiv.org/pdf/2503.04119](https://arxiv.org/pdf/2503.04119)
- Code：[https://github.com/scn-00/SCSA](https://github.com/scn-00/SCSA)

#### K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs
- Link：[https://arxiv.org/pdf/2502.18461](https://arxiv.org/pdf/2502.18461)
- Code：[https://k-lora.github.io/K-LoRA.io/](https://k-lora.github.io/K-LoRA.io/)
#### Zero-Shot Head Swapping in Real-World Scenarios
- Link：[https://arxiv.org/pdf/2503.00861](https://arxiv.org/pdf/2503.00861)



<a name="Image-Inpainting"></a>

# 图像补全/图像修复(Image Inpainting)


<a name="GAN"></a>

# 生成对抗网络(GAN)




<a name="Video-Editing"></a>

# 视频编辑(Video Editing)



<a name="LLV"></a>

# Low-level Vision

#### Feature Fusion Attention Network with CycleGAN for Image Dehazing, De-Snowing and De-Raining
- Link：[https://arxiv.org/pdf/2503.06107](https://arxiv.org/pdf/2503.06107)



#### One Model for ALL: Low-Level Task Interaction Is a Key to Task-Agnostic Image Fusion
- Link：[https://arxiv.org/pdf/2502.19854](https://arxiv.org/pdf/2502.19854)
- Code：[https://github.com/AWCXV/GIFNet](https://github.com/AWCXV/GIFNet)

#### UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion
- Link：[https://arxiv.org/pdf/2501.11515](https://arxiv.org/pdf/2501.11515)
- Code：[https://openimaginglab.github.io/UltraFusion/](https://openimaginglab.github.io/UltraFusion/)


<a name="SR"></a>

# 超分辨率(Super-Resolution)
#### Adaptive Rectangular Convolution for Remote Sensing Pansharpening
- Link：[https://arxiv.org/pdf/2503.00467](https://arxiv.org/pdf/2503.00467)
- Code：[https://github.com/WangXueyang-uestc/ARConv.git](https://github.com/WangXueyang-uestc/ARConv.git)

#### Arbitrary-steps Image Super-resolution via Diffusion Inversion
- Link：[https://arxiv.org/pdf/2412.09013](https://arxiv.org/pdf/2412.09013)
- Code：[https://github.com/zsyOAOA/InvSR](https://github.com/zsyOAOA/InvSR)



#### Complexity Experts are Task-Discriminative Learners for Any Image Restoration
- Link：[https://arxiv.org/pdf/2411.18466](https://arxiv.org/pdf/2411.18466)
- Code：[https://eduardzamfir.github.io/moceir/](https://eduardzamfir.github.io/moceir/)

<a name="Denoising"></a>

# 去噪(Denoising)

## 图像去噪(Image Denoising)

<a name="3D-Human-Pose-Estimation"></a>



<a name="Image-Generation"></a>

# 图像生成(Image Generation)

#### Finding Local Diffusion Schrödinger Bridge using Kolmogorov-Arnold Network
- Link：[https://arxiv.org/pdf/2502.19754](https://arxiv.org/pdf/2502.19754)
- Code：[https://github.com/Qiu-XY/LDSB](https://github.com/Qiu-XY/LDSB)

#### Towards Improved Text-Aligned Codebook Learning: Multi-Hierarchical Codebook-Text Alignment with Long Text
- Link：[https://arxiv.org/pdf/2503.01261](https://arxiv.org/pdf/2503.01261)

#### Tiled Diffusion
- Link：[https://arxiv.org/pdf/2412.15185](https://arxiv.org/pdf/2412.15185)


#### ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation
- Link：[https://arxiv.org/pdf/2503.10358](https://arxiv.org/pdf/2503.10358)


#### Adv - CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks
- Link：[https://arxiv.org/pdf/2503.08269](https://arxiv.org/pdf/2503.08269)



<a name="Video-Generation"></a>

# 视频生成(Video Generation)

#### Mind the Time: Temporally-Controlled Multi-Event Video Generation
- Link：[https://arxiv.org/pdf/2412.05263](https://arxiv.org/pdf/2412.05263)
- Code：[https://mint-video.github.io/](https://mint-video.github.io/)
- 
#### Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition
- Link：[https://arxiv.org/pdf/2503.06984](https://arxiv.org/pdf/2503.06984)
- Code：[https://wjc2830.github.io/MelQCD/](https://wjc2830.github.io/MelQCD/)

#### AR-Diffusion: Asynchronous Video Generation with Auto-Regressive Diffusion
- Link：[https://arxiv.org/pdf/2503.07418](https://arxiv.org/pdf/2503.07418)
- Code：[https://github.com/iva-mzsun/AR-Diffusion](https://github.com/iva-mzsun/AR-Diffusion)

#### GEN3C: 3D - Informed World - Consistent Video Generation with Precise Camera Control
- Link：https://arxiv.org/pdf/2503.03751
- Code：[https://research.nvidia.com/labs/toronto-ai/GEN3C/](https://research.nvidia.com/labs/toronto-ai/GEN3C/)

#### KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation
- Link：[https://arxiv.org/pdf/2503.01715](https://arxiv.org/pdf/2503.01715)

#### AnyMoLe: Any Character Motion In-betweening Leveraging Video Diffusion Models
- Link：[https://arxiv.org/pdf/2503.08417](https://arxiv.org/pdf/2503.08417)
- Code：[https://kwanyun.github.io/AnyMoLe_page/](https://kwanyun.github.io/AnyMoLe_page/)

#### ObjectMover: Generative Object Movement with Video Prior
- Link：[https://arxiv.org/pdf/2503.08037](https://arxiv.org/pdf/2503.08037)
- Code：[https://xinyu-andy.github.io/ObjMover](https://xinyu-andy.github.io/ObjMover)
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

#### MEAT: Multiview Diffusion Model for Human Generation on Megapixels with Mesh Attention
- Link：[https://arxiv.org/pdf/2503.08664](https://arxiv.org/pdf/2503.08664)
- Code：[https://johann.wang/MEAT/](https://johann.wang/MEAT/)

#### Make - It - Animatable: An Efficient Framework for Authoring Animation - Ready 3D Characters
- Link：[https://arxiv.org/pdf/2411.18197](https://arxiv.org/pdf/2411.18197)
- Code：[https://jasongzy.github.io/Make-It-Animatable/](https://jasongzy.github.io/Make-It-Animatable/)

<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)

#### BIMBA: Selective-Scan Compression for Long-Range Video Question Answering
- Link：[https://arxiv.org/pdf/2503.09590](https://arxiv.org/pdf/2503.09590)
- Code：[https://sites.google.com/view/bimba-mllm](https://sites.google.com/view/bimba-mllm)

#### Narrating the Video: Boosting Text-Video Retrieval via Comprehensive Utilization of Frame-Level Captions
- Link：[https://arxiv.org/pdf/2503.05186](https://arxiv.org/pdf/2503.05186)
- Code：[https://multimodal-understanding-group.github.io/NarVid/](https://multimodal-understanding-group.github.io/NarVid/)

#### VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary
- Link：[https://arxiv.org/pdf/2503.09402](https://arxiv.org/pdf/2503.09402)
- Code：[https://github.com/showlab/VLog](https://github.com/showlab/VLog)

#### HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding
- Link：[https://arxiv.org/pdf/2503.08585](https://arxiv.org/pdf/2503.08585)
- Code：[https://sacrcv.github.io/HierarQ-website/](https://sacrcv.github.io/HierarQ-website/)

<a name="3D-Human-Pose-Estimation"></a>
# 3D人体姿态估计(3D Human Pose Estimation)

#### HumanMM: Global Human Motion Recovery from Multi-shot Videos
- Link：[https://arxiv.org/pdf/2503.07597](https://arxiv.org/pdf/2503.07597)
- Code：[https://zhangyuhong01.github.io/HumanMM/](https://zhangyuhong01.github.io/HumanMM/)

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

#### SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance
- Link：[https://arxiv.org/pdf/2503.01291](https://arxiv.org/pdf/2503.01291)
- Code：[https://4dvlab.github.io/project_page/semgeomo/](https://4dvlab.github.io/project_page/semgeomo/)

#### HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation
- Link：[https://arxiv.org/pdf/2503.01175](https://arxiv.org/pdf/2503.01175)
- Code：[https://star-uu-wang.github.io/HOP/](https://star-uu-wang.github.io/HOP/)

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

#### Taming Large Multimodal Agents for Ultra-low Bitrate Semantically Disentangled Image Compression
- Link：[https://arxiv.org/pdf/2503.00399](https://arxiv.org/pdf/2503.00399)
- Code：[https://github.com/yang-xidian/SEDIC](https://github.com/yang-xidian/SEDIC)


<a name="Stereo-Matching"></a>

# 立体匹配(Stereo Matching)

#### FoundationStereo: Zero-Shot Stereo Matching
- Link：[https://arxiv.org/pdf/2501.09898](https://arxiv.org/pdf/2501.09898)
- Code：[https://nvlabs.github.io/FoundationStereo/](https://nvlabs.github.io/FoundationStereo/)

<a name="SGG"></a>

# 场景图生成(Scene Graph Generation)

#### Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing
- Link：[https://arxiv.org/pdf/2503.00548](https://arxiv.org/pdf/2503.00548)


<a name="Counting"></a>

# 计数(Counting)




<a name="INR"></a>

# 隐式神经表示(Implicit Neural Representations)

#### RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings
- Link：[https://arxiv.org/pdf/2502.19781](https://arxiv.org/pdf/2502.19781)

#### Dynamic Neural Surfaces for Elastic 4D Shape Representation and Analysis
- Link：[https://arxiv.org/pdf/2503.03132](https://arxiv.org/pdf/2503.03132)
- Code：[https://4d-dsns.github.io/DSNS/](https://4d-dsns.github.io/DSNS/)

#### MetricGrids: Arbitrary Nonlinear Approximation with Elementary Metric Grids based Implicit Neural Representation
- Link：[https://arxiv.org/pdf/2503.10000](https://arxiv.org/pdf/2503.10000)
- Code：[https://github.com/wangshu31/MetricGrids](https://github.com/wangshu31/MetricGrids)

<a name="IQA"></a>

# 图像质量评价(Image Quality Assessment)

#### SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model
- Link：[https://arxiv.org/pdf/2412.04292](https://arxiv.org/pdf/2412.04292)
- Code：[https://hzlsaber.github.io/projects/SIDA/](https://hzlsaber.github.io/projects/SIDA/)

#### Q - Eval - 100K: Evaluating Visual Quality and Alignment Level for Text - to - Vision Content
- Link：[https://arxiv.org/pdf/2503.02357](https://arxiv.org/pdf/2503.02357)
- Code：https://github.com/zzc-1998/Q-Eval


<a name="Video-Quality-Assessment"></a>

# 视频质量评价(Video Quality Assessment)

<a name="Datasets"></a>

# 数据集(Datasets)
#### AIM-Fair: Advancing Algorithmic Fairness via Selectively Fine-Tuning Biased Models with Contextual Synthetic Data
- Link：[https://arxiv.org/pdf/2503.05665](https://arxiv.org/pdf/2503.05665)
- Code：[https://github.com/zengqunzhao/AIM-Fair](https://github.com/zengqunzhao/AIM-Fair)

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

#### AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark
- Link：[https://arxiv.org/pdf/2406.00783](https://arxiv.org/pdf/2406.00783)
- Code：[https://github.com/Purdue-M2/AI-Face-FairnessBench](https://github.com/Purdue-M2/AI-Face-FairnessBench)

#### Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?
- Link：[https://arxiv.org/pdf/2503.09122](https://arxiv.org/pdf/2503.09122)
- Code：[https://github.com/xieyc99/TrainProVe](https://github.com/xieyc99/TrainProVe)

<a name="Unlearning"></a>
# 反学习(Machine Unlearning)

#### MINT - Demo: Membership Inference Test Demonstrator
- Link：[https://arxiv.org/pdf/2503.08332](https://arxiv.org/pdf/2503.08332)

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


#### Rashomon Sets for Prototypical-Part Networks: Editing Interpretable Models in Real-Time
- Link：[https://arxiv.org/pdf/2503.01087](https://arxiv.org/pdf/2503.01087)
