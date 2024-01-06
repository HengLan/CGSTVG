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
Training/Evaluation will be coming soon ...

## Experiments
CG-STVG achieves state-of-the-art performance on three challenging benchmarks, including HCSTVG-v1, HCSTVG-v2, and VidSTG, as shown below.
