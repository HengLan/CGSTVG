[![PWC](https://img.shields.io/badge/State_of_the_Art-STVG_on_HCSTVGv1-orange?logo=AMP)](https://paperswithcode.com/sota/spatio-temporal-video-grounding-on-hc-stvg1)
[![PWC](https://img.shields.io/badge/State_of_the_Art-STVG_on_HCSTVGv2-pink?logo=AMP)](https://paperswithcode.com/sota/spatio-temporal-video-grounding-on-hc-stvg2)
[![PWC](https://img.shields.io/badge/State_of_the_Art-STVG_on_VidSTG-yellow?logo=AMP)](https://paperswithcode.com/sota/spatio-temporal-video-grounding-on-vidstg)

# Context-Guided Spatio-Temporal Video Grounding
🔮 Welcome to the official code repository for [**CG-STVG: Context-Guided Spatio-Temporal Video Grounding**](https://arxiv.org/abs/2401.01578). We're excited to share our work with you, please bear with us as we prepare code. Stay tuned for the reveal!

## Illustration of Idea
![CG-STVG](https://github.com/HengLan/CGSTVG/blob/main/assets/idea.png)
**Figure:** Comparison between (a) existing methods that localize the target using object information from text query and (b) our CG-STVG
that enjoys object information from text query and guidance from mined instance context for STVG. 

## Framework
![CG-STVG](https://github.com/HengLan/CGSTVG/blob/main/assets/framework.png)
**Figure:** Overview of our method, which consists of a multimodal encoder for feature extraction and a context-guided decoder by cascading
a set of decoding stages for grounding. In each decoding stage, instance context is mined to guide query learning for better localization. More details can be seen in the [**paper**](https://arxiv.org/abs/2401.01578).

## Get Started
### Training/Evaluation
Coming soon ...

### Pre-trained models
Coming soon ...

## Experiments
CG-STVG achieves state-of-the-art performance on three challenging benchmarks, including [**HCSTVG-v1**](https://github.com/tzhhhh123/HC-STVG), [**HCSTVG-v2**](https://github.com/tzhhhh123/HC-STVG), and [**VidSTG**](https://github.com/Guaranteer/VidSTG-Dataset), as shown below. Note that, the baseline is our CG-STVG without context generation and refinement.

### Results on HCSTVG-v1
|  Methods   | M_tIoU | m_vIoU | vIoU@0.3 | vIoU@0.5  |
|  ----:  | :----:  | :----: | :----: | :----: |
|STGVT<sub>[TCSVT'2021]</sub> | - |  18.2 | 26.8 | 9.5|
|STVGBert<sub>[ICCV'2021]</sub> | - | 20.4 | 29.4 |  11.3|
|TubeDETR<sub>[CVPR'2022]</sub> | 43.7 | 32.4 | 49.8 | 23.5|
|STCAT<sub>[NeurIPS'2022]</sub> | 49.4 | 35.1 | 57.7 | 30.1|
|CSDVL<sub>[CVPR'2023]</sub> | - | 36.9 | **62.2** | 34.8|
|Baseline (ours) | 50.4 | 36.5 | 58.6 | 32.3 |
|CG-STVG (ours)|**52.8**<sub>+2.4</sub> | **38.4**<sub>+1.9</sub> | 61.5<sub>+2.9</sub> | **36.3**<sub>+4.0</sub>|

### Results on HCSTVG-v2
|  Methods   | M_tIoU | m_vIoU | vIoU@0.3 | vIoU@0.5  |
|  ----:  | :----:  | :----: | :----: | :----: |
|PCC<sub>[arxiv'2021]</sub> | - |  30.0 | - | - |
|2D-Tan<sub>[arxiv'2021]</sub>  | - | 30.4 |  50.4 | 18.8 |
