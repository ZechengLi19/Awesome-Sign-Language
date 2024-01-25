# Awesome-Sign-Language
This repository collects the common datasets and paper list related to the research on **Sign Language**🤟

This repository is continuously updating🎉

If this repository brings you some inspiration, I would be very honored😊

If you have any suggestions, feel free to contact me with: lizecheng19@gmail.com📮

## Contents
- [Datasets](#Datasets)
- [Isolated sign language recognition](#islr_2016)
- [Continue sign language recognition](#cslr_2016)
- [Sign language translation](#slt_2018)
- [Sign Generation](#sg_2018)
- [Sign Language Retrieval](#slr_2022)
- [Pre-training](#pt_2020)

## Popular Datasets
<a id="Datasets"></a>
- Isolated sign language recognition datasets:
  - **WLASL**: 14,289, 3,916, and 2,878 video segments in the train, dev, and test splits, respectively. [[Link](https://dxli94.github.io/WLASL/)]
  - **MSASL**: 16,054, 5,287, and 4,172 video segments in the train, dev, and test splits, respectively. [[Link](https://www.microsoft.com/en-us/research/project/ms-asl/)]
  - **NMFs-CSL**: 25,608 and 6,402 video segments in the train and test splits, respectively. [[Link](https://ustc-slr.github.io/datasets/2020_nmfs_csl/)]
  - **SLR500**: 90,000 and 35,000 video segments in the train and test splits, respectively. [[Link](http://home.ustc.edu.cn/~hagjie/)]
  - **Slovo**: 15,300 and 5,100 video segments in the train and test splits, respectively. [[Link](https://github.com/hukenovs/slovo)]

- Continue sign language recognition datasets:
  - **Phoenix-2014**: 5,672, 540 and 629 video segments in the train, dev, and test splits, respectively. [[Link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)]
  - **Phoenix-2014T**: 7,096, 519 and 642 video segments in train, dev and test splits, respectively. [[Link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)]

- Sign language translation datasets:
  - **Phoenix-2014T**: 7,096, 519 and 642 video segments in
train, dev and test splits, respectively. [[Link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)]
  - **CSL-Daily**: 18,401, 1,077 and 1,176 video segments in train, dev
and test splits, respectively. [[Link](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/)]
  - **OpenASL**: 96,476, 966 and 975 video segments in train, val and test splits, respectively. [[Link](https://github.com/chevalierNoir/OpenASL/)]
  - **How2Sign**: 31,128, 1,741, 2,322 video segments in train, val and test splits, respectively. [[Link](https://how2sign.github.io/)]

## Paper List
### Isolated sign language recognition
### <a id="islr_2016">2016</a>
  - **Iterative Reference Driven Metric Learning for Signer Independent Isolated Sign**. *ECCV 2016*. [[Paper](http://vipl.ict.ac.cn/uploadfile/upload/2018112115134267.pdf)]

### <a id="islr_2019">2019</a>
  - **Skeleton-Based Gesture Recognition Using Several Fully Connected Layers with Path Signature Features and Temporal Transformer Module**. *AAAI 2019*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4878)]

### <a id="islr_2020">2020</a>
  - **Transferring Cross-Domain Knowledge for Video Sign Language Recognition**. *CVPR 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Transferring_Cross-Domain_Knowledge_for_Video_Sign_Language_Recognition_CVPR_2020_paper.html)]
  - **BSL-1K: Scaling up co-articulated sign language recognition using mouthing cues**. *ECCV 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1279_ECCV_2020_paper.php)]
  - **Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison**. *WACV 2020*. [[Paper](https://openaccess.thecvf.com/content_WACV_2020/html/Li_Word-level_Deep_Sign_Language_Recognition_from_Video_A_New_Large-scale_WACV_2020_paper.html)][[Code](https://github.com/dxli94/WLASL)] 
  - **FineHand: Learning Hand Shapes for American Sign Language Recognition**. *FG 2020*. [[Paper](https://ieeexplore.ieee.org/document/9320289)]

### <a id="islr_2021">2021</a>
  - **Hand-Model-Aware Sign Language Recognition**. *AAAI 2021*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16247)]
  - **Global-Local Enhancement Network for NMF-Aware Sign Language Recognition**. *TMM 2021*. [[Paper](https://dl.acm.org/doi/10.1145/3436754)]
  - **Hand Pose Guided 3D Pooling for Word-level Sign Language Recognition**. *WACV 2021*. [[Paper](https://openaccess.thecvf.com/content/WACV2021/html/Hosain_Hand_Pose_Guided_3D_Pooling_for_Word-Level_Sign_Language_Recognition_WACV_2021_paper.html)]
  - **Pose-based Sign Language Recognition using GCN and BERT**. *WACVW 2021*. [[Paper](https://openaccess.thecvf.com/content/WACV2021W/HBU/html/Tunga_Pose-Based_Sign_Language_Recognition_Using_GCN_and_BERT_WACVW_2021_paper.html)]
  - **Skeleton Aware Multi-modal Sign Language Recognition**. *CVPRW 2021*. [[Paper](https://arxiv.org/pdf/2103.08833.pdf)][[Code](https://github.com/jackyjsy/CVPR21Chal-SLR)]
  - **Sign Language Recognition via Skeleton-Aware Multi-Model Ensemble**. *Arxiv 2021*. [[Paper](https://arxiv.org/pdf/2110.06161.pdf)][[Code](https://github.com/jackyjsy/SAM-SLR-v2)]

### <a id="islr_2023">2023</a>
  - **Isolated Sign Language Recognition based on Tree Structure Skeleton Images**. *CVPRW 2023*. [[Paper](https://arxiv.org/pdf/2304.05403.pdf)][[Code](https://github.com/davidlainesv/SL-TSSI-DenseNet)]
  - **Natural Language-Assisted Sign Language Recognition**. *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zuo_Natural_Language-Assisted_Sign_Language_Recognition_CVPR_2023_paper.html)][[Code](https://github.com/FangyunWei/SLRT/tree/main/NLA-SLR)]
  - **Human Part-wise 3D Motion Context Learning for Sign Language Recognition**. *ICCV 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Human_Part-wise_3D_Motion_Context_Learning_for_Sign_Language_Recognition_ICCV_2023_paper.pdf)]
  
### Continue sign language recognition
### <a id="cslr_2016">2016</a>
  - **Deep Sign: Hybrid CNN-HMM for Continuous Sign Language Recognition**. *BMVC 2016*. [[Paper](https://bmva-archive.org.uk/bmvc/2016/papers/paper136/index.html)]

### <a id="cslr_2017">2017</a>
  - **SubUNets: End-To-End Hand Shape and Continuous Sign Language Recognition**. *ICCV 2017*. [[Paper](https://openaccess.thecvf.com/content_iccv_2017/html/Camgoz_SubUNets_End-To-End_Hand_ICCV_2017_paper.html)]
  - **Recurrent Convolutional Neural Networks for Continuous Sign Language Recognition by Staged Optimization**. *CVPR 2017*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Cui_Recurrent_Convolutional_Neural_CVPR_2017_paper.html)]

### <a id="cslr_2018">2018</a>
  - **Deep Sign: Enabling Robust Statistical Continuous Sign Language Recognition via Hybrid CNN-HMMs**. *IJCV 2018*. [[Paper](https://link.springer.com/article/10.1007/s11263-018-1121-3)]

### <a id="cslr_2019">2019</a>
  - **Iterative Alignment Network for Continuous Sign Language Recognition**. *CVPR 2019*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Pu_Iterative_Alignment_Network_for_Continuous_Sign_Language_Recognition_CVPR_2019_paper.html)]

### <a id="cslr_2020">2020</a>
  - **Boosting Continuous Sign Language Recognition via Cross Modality Augmentation**. *ACM MM 2020*. [[Paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413931)]
  - **Stochastic Fine-grained Labeling of Multi-state Sign Glosses for Continuous Sign Language Recognition**. *ECCV 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2527_ECCV_2020_paper.php)]
  - **Fully Convolutional Networks for Continuous Sign Language Recognition**. *ECCV 2020*. [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4763_ECCV_2020_paper.php)]
  - **Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition**. *AAAI 2020*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/7001)]

### <a id="cslr_2021">2021</a>
  - **Visual Alignment Constraint for Continuous Sign Language Recognition**. *ICCV 2021*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html)][[Code](https://github.com/ycmin95/VAC_CSLR)] 
  - **Self-Mutual Distillation Learning for Continuous Sign Language Recognition**. *ICCV 2021*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Hao_Self-Mutual_Distillation_Learning_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html)]

### <a id="cslr_2022">2022</a>
  - **Signing Outside the Studio: Benchmarking Background Robustness for Continuous Sign Language Recognition**. *BMVC 2022*. [[Paper](https://bmvc2022.mpi-inf.mpg.de/322/)][[Code](https://github.com/art-jang/Signing-Outside-the-Studio)]
  - **Temporal Lift Pooling for Continuous Sign Language Recognition**. *ECCV 2022*. [[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/160_ECCV_2022_paper.php)][[Code](https://github.com/hulianyuyy/Temporal-Lift-Pooling)]
  - **C2SLR: Consistency-Enhanced Continuous Sign Language Recognition**. *CVPR 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zuo_C2SLR_Consistency-Enhanced_Continuous_Sign_Language_Recognition_CVPR_2022_paper.html)]

### <a id="cslr_2023">2023</a>
  - **AdaBrowse: Adaptive Video Browser for Efficient Continuous Sign Language Recognition**. *ACM MM 2023*. [[Paper](https://dl.acm.org/doi/10.1145/3581783.3611745)][[Code](https://github.com/hulianyuyy/AdaBrowse)]
  - **CoSign: Exploring Co-occurrence Signals in Skeleton-based Continuous Sign Language Recognition**. *ICCV 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.html)]
  - **Improving Continuous Sign Language Recognition with Cross-Lingual Signs**. *ICCV 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Wei_Improving_Continuous_Sign_Language_Recognition_with_Cross-Lingual_Signs_ICCV_2023_paper.html)]
  - **C2ST: Cross-modal Contextualized Sequence Transduction for Continuous Sign Language Recognition**. *ICCV 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_C2ST_Cross-Modal_Contextualized_Sequence_Transduction_for_Continuous_Sign_Language_Recognition_ICCV_2023_paper.html)]
  - **CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment**. *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_CVT-SLR_Contrastive_Visual-Textual_Transformation_for_Sign_Language_Recognition_With_Variational_CVPR_2023_paper.html)][[Code](https://github.com/binbinjiang/CVT-SLR)]
  - **Continuous Sign Language Recognition with Correlation Network**. *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Continuous_Sign_Language_Recognition_With_Correlation_Network_CVPR_2023_paper.html)][[Code](https://github.com/hulianyuyy/CorrNet)]
  - **Distilling Cross-Temporal Contexts for Continuous Sign Language Recognition**. *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Guo_Distilling_Cross-Temporal_Contexts_for_Continuous_Sign_Language_Recognition_CVPR_2023_paper.html)]
  - **Self-Emphasizing Network for Continuous Sign Language Recognition**. *AAAI 2023*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25164)][[Code](https://github.com/hulianyuyy/SEN_CSLR)]

### Sign language translation
### <a id="slt_2018">2018</a>
  - **Neural Sign Language Translation**. *CVPR 2018*. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Camgoz_Neural_Sign_Language_CVPR_2018_paper.html)][[Code](https://github.com/neccam/nslt)]

### <a id="slt_2020">2020</a>
  - **Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation**. *CVPR 2020*. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Camgoz_Sign_Language_Transformers_Joint_End-to-End_Sign_Language_Recognition_and_Translation_CVPR_2020_paper.html)][[Code](https://github.com/neccam/slt)]
  - **TSPNet: Hierarchical Feature Learning via Temporal Semantic Pyramid for Sign Language Translation**. *NeurIPS 2020*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/8c00dee24c9878fea090ed070b44f1ab-Abstract.html)][[Code](https://github.com/verashira/TSPNet)]
  - **Neural Sign Language Translation by Learning Tokenization**. *FG 2020*. [[Paper](https://ieeexplore.ieee.org/document/9320278?denied=)]

### <a id="slt_2021">2021</a>
  - **Spatial-Temporal Multi-Cue Network for Sign Language Recognition and Translation**. *TMM 2021*. [[Paper](https://ieeexplore.ieee.org/document/9354538)]
  - **How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language**. *CVPR 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Duarte_How2Sign_A_Large-Scale_Multimodal_Dataset_for_Continuous_American_Sign_Language_CVPR_2021_paper.html)][[Project](https://how2sign.github.io/)]
  - **Improving Sign Language Translation with Monolingual Data by Sign Back-Translation**. *CVPR 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Model-Aware_Gesture-to-Gesture_Translation_CVPR_2021_paper.html)]
  - **Skeleton-Aware Neural Sign Language Translation**. *ACM MM 2021*. [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475577)][[Code](https://github.com/SignLanguageCode/SANet)]
  - **SimulSLT: End-to-End Simultaneous Sign Language Translation**. *ACM MM 2021*. [[Paper](https://arxiv.org/abs/2112.04228)][[Code](https://github.com/Robert0125/SimulSLT)]

### <a id="slt_2022">2022</a>
  - **Prior Knowledge and Memory Enriched Transformer for Sign Language Translation**. *ACL 2022*. [[Paper](https://aclanthology.org/2022.findings-acl.297/)][[Code](https://github.com/hugddygff/PET)]
  - **Open-Domain Sign Language Translation Learned from Online Video**. *EMNLP 2022*. [[Paper](https://aclanthology.org/2022.emnlp-main.427/)][[Code](https://github.com/chevalierNoir/OpenASL)]
  - **Automatic Gloss-level Data Augmentation for Sign Language Translation**. *LREC 2022*. [[Paper](https://aclanthology.org/2022.lrec-1.734.pdf)]
  - **A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation**. *CVPR 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_A_Simple_Multi-Modality_Transfer_Learning_Baseline_for_Sign_Language_Translation_CVPR_2022_paper.html)][[Code](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork)]
  - **MLSLT: Towards Multilingual Sign Language Translation**. *CVPR 2022*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Yin_MLSLT_Towards_Multilingual_Sign_Language_Translation_CVPR_2022_paper.html)][[Code](https://github.com/MLSLT/SP-10)]
  - **Two-Stream Network for Sign Language Recognition and Translation**. *NeurIPS 2022*. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6cd3ac24cdb789beeaa9f7145670fcae-Abstract-Conference.html)][[Code](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork)]
  - **Sign Language Translation With Hierarchical Spatio-Temporal Graph Neural Network**. *WACV 2022*. [[Paper](https://openaccess.thecvf.com/content/WACV2022/html/Kan_Sign_Language_Translation_With_Hierarchical_Spatio-Temporal_Graph_Neural_Network_WACV_2022_paper.html)]
  - **Sign Language Translation based on Transformers for the How2Sign Dataset**. *Report 2022*. [[Paper](https://imatge.upc.edu/web/sites/default/files/pub/xCabot22.pdf)]

### <a id="slt_2023">2023</a>
  - **Gloss-Free End-to-End Sign Language Translation**. *ACL 2023*. [[Paper](https://aclanthology.org/2023.acl-long.722/)][[Code](https://github.com/HenryLittle/GloFE)]
  - **Neural Machine Translation Methods for Translating Text to Sign Language Glosses**. *ACL 2023*. [[Paper](https://aclanthology.org/2023.acl-long.700/)]
  - **Considerations for meaningful sign language machine translation based on glosses**. *ACL 2023*. [[Paper](https://aclanthology.org/2023.acl-short.60/)]
  - **ISLTranslate: Dataset for Translating Indian Sign Language**. *ACL 2023*. [[Paper](https://aclanthology.org/2023.findings-acl.665/)][[Code](https://github.com/Exploration-Lab/ISLTranslate)]
  - **Gloss Attention for Gloss-free Sign Language Translation**. *CVPR 2023*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Yin_Gloss_Attention_for_Gloss-Free_Sign_Language_Translation_CVPR_2023_paper.html)][[Code](https://github.com/YinAoXiong/GASLT)]
  - **Sign Language Translation with Iterative Prototype**. *ICCV 2023*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Yao_Sign_Language_Translation_with_Iterative_Prototype_ICCV_2023_paper.html)]
  - **Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining**. *ICCV 2023*. [[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.html)][[Code](https://github.com/zhoubenjia/GFSLT-VLP)]
  - **SLTUNET: A Simple Unified Model for Sign Language Translation**. *ICLR 2023*. [[paper](https://openreview.net/forum?id=EBS4C77p_5S)][[Code](https://github.com/bzhangGo/sltunet)]
  - **Cross-modality Data Augmentation for End-to-End Sign Language Translation**. *EMNLP 2023*. [[paper](https://arxiv.org/pdf/2305.11096.pdf)][[Code](https://github.com/Atrewin/SignXmDA)]

### <a id="slt_2024">2024</a>
  - **Sign2GPT: Leveraging Large Language Models for Gloss-Free Sign Language Translation**. *ICLR 2024*. [[paper](https://openreview.net/forum?id=LqaEEs3UxU)]
  - **Conditional Variational Autoencoder for Sign Language Translation with Cross-Modal Alignment**. *AAAI 2024*. [[paper](https://arxiv.org/pdf/2312.15645.pdf)][[Code](https://github.com/rzhao-zhsq/CV-SLT)]

### Sign Generation
### <a id="sg_2018">2018</a>
  - **GestureGAN for Hand Gesture-to-Gesture Translation in the Wild**. *ACM MM 2018*. [[Paper](https://dl.acm.org/doi/abs/10.1145/3240508.3240704)]

### <a id="sg_2020">2020</a>
  - **Text2Sign: Towards Sign Language Production Using Neural Machine Translation and Generative Adversarial Networks**. *IJCV 2020*. [[Paper](https://link.springer.com/article/10.1007/s11263-019-01281-2#citeas)]

### <a id="sg_2021">2021</a>
  - **Model-Aware Gesture-to-Gesture Translation**. *CVPR 2021*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Model-Aware_Gesture-to-Gesture_Translation_CVPR_2021_paper.html)]

### <a id="sg_2024">2024</a>
  - **Sign Language Production with Latent Motion Transformer**. *WACV 2024*. [[Paper](https://arxiv.org/pdf/2312.12917.pdf)]

### Sign Language Retrieval
### <a id="slr_2022">2022</a>
  - **Sign Language Video Retrieval with Free-Form Textual Queries**. *CVPR 2022*. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Duarte_Sign_Language_Video_Retrieval_With_Free-Form_Textual_Queries_CVPR_2022_paper.pdf)][[Project](https://imatge-upc.github.io/sl_retrieval/)]

### <a id="slr_2023">2023</a>
  - **CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning**. *CVPR 2023*. [[paper](https://arxiv.org/pdf/2303.12793.pdf)][[Code](https://github.com/FangyunWei/SLRT)]

### Pre-training
### <a id="pt_2021">2021</a>
  - **SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition**. *ICCV 2021*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_SignBERT_Pre-Training_of_Hand-Model-Aware_Representation_for_Sign_Language_Recognition_ICCV_2021_paper.html)]

### <a id="pt_2023">2023</a>
  - **BEST: BERT Pre-Training for Sign Language Recognition with Coupling Tokenization**. *AAAI 2023*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25470)]
  - **SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding**. *TPAMI 2023*. [[Paper](https://ieeexplore.ieee.org/document/10109128)][[Project](https://signbert-zoo.github.io/)]

