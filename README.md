[![PWC](https://img.shields.io/badge/State_of_the_Art-STVG_on_HCSTVGv1-orange?logo=AMP)](https://paperswithcode.com/sota/spatio-temporal-video-grounding-on-hc-stvg1)
[![PWC](https://img.shields.io/badge/State_of_the_Art-STVG_on_HCSTVGv2-pink?logo=AMP)](https://paperswithcode.com/sota/spatio-temporal-video-grounding-on-hc-stvg2)
[![PWC](https://img.shields.io/badge/State_of_the_Art-STVG_on_VidSTG-yellow?logo=AMP)](https://paperswithcode.com/sota/spatio-temporal-video-grounding-on-vidstg)

# Context-Guided Spatio-Temporal Video Grounding
🔮 Welcome to the official code repository for [**CG-STVG: Context-Guided Spatio-Temporal Video Grounding**](https://arxiv.org/abs/2401.01578). We're excited to share our work with you, please bear with us as we prepare code. Stay tuned for the reveal!

## Illustration of Idea
![CG-STVG](https://github.com/HengLan/CGSTVG/blob/main/assets/idea.png)
**Figure:** Comparison between (a) existing methods that localize the target using object information from text query and (b) our CG-STVG
that enjoys object information from text query and guidance from instance context for STVG. 

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
|MMN<sub>[AAAI'2022]</sub> | - | 30.3 | 49.0 | 25.6|
|TubeDETR<sub>[CVPR'2022]</sub> | - | 36.4 | 58.8 | 30.6|
|CSDVL<sub>[CVPR'2023]</sub> | 58.1 | 38.7 | **65.5** | 33.8|
|Baseline (ours) | 58.6 | 37.8 | 62.4 | 32.1|
|CG-STVG (ours) | **60.0**<sub>+1.4</sub> | **39.5**<sub>+1.7</sub> | 64.5<sub>+2.1</sub> | **36.3**<sub>+4.2</sub>|

### Results on VidSTG
<table>
  <tr>
    <td rowspan="2" align="right"><b>Methods</b></td>
    <td colspan="4" align="center"><b>Declarative Sentences</b></td>
    <td colspan="4" align="center"><b>Interrogative Sentences</b></td>
  </tr>
  <tr>
    <td align="center"><b>M_tIoU</b></td>
    <td align="center"><b>m_vIoU</b></td>
    <td align="center"><b>vIoU@0.3</b></td>
    <td align="center"><b>vIoU@0.5</b></td>
    <td align="center"><b>M_tIoU</b></td>
    <td align="center"><b>m_vIoU</b></td>
    <td align="center"><b>vIoU@0.3</b></td>
    <td align="center"><b>vIoU@0.5</b></td>
  </tr>
  <tr>
    <td align="right">STGRN<sub>[CVPR'2020]</sub></td>
    <td align="center">48.5</td>
    <td align="center">19.8</td>
    <td align="center">25.8</td>
    <td align="center">14.6</td>
    <td align="center">47.0 </td>    
    <td align="center">18.3</td>
    <td align="center">21.1</td>
    <td align="center">12.8</td>
  </tr>
  <tr>
    <td align="right">OMRN<sub>[IJCAI'2020]</sub></td>
    <td align="center">50.7</td>
    <td align="center">23.1</td>
    <td align="center">32.6</td>
    <td align="center">16.4</td>
    <td align="center">49.2</td>    
    <td align="center">20.6</td>
    <td align="center">28.4</td>
    <td align="center">14.1</td>
  </tr>
  <tr>
    <td align="right">STGVT<sub>[TCSVT'2021]</sub></td>
    <td align="center">-</td>
    <td align="center">21.6</td>
    <td align="center">29.8</td>
    <td align="center">18.9</td>
    <td align="center">-</td>    
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="right">STVGBert<sub>[ICCV'2021]</sub></td>
    <td align="center">-</td>
    <td align="center">24.0</td>
    <td align="center">30.9</td>
    <td align="center">18.4</td>
    <td align="center">-</td>    
    <td align="center">22.5</td>
    <td align="center">26.0</td>
    <td align="center">16.0</td>
  </tr>
  <tr>
    <td align="right">TubeDETR<sub>[CVPR'2022]</sub></td>
    <td align="center">48.1</td>
    <td align="center">30.4</td>
    <td align="center">42.5</td>
    <td align="center">28.2</td>
    <td align="center">46.9</td>    
    <td align="center">25.7</td>
    <td align="center">35.7</td>
    <td align="center">23.2</td>
  </tr>
  <tr>
    <td align="right">STCAT<sub>[NeurIPS'2022]</sub></td>
    <td align="center">50.8</td>
    <td align="center">33.1</td>
    <td align="center">46.2</td>
    <td align="center">32.6</td>
    <td align="center">49.7</td>    
    <td align="center">28.2</td>
    <td align="center">39.2</td>
    <td align="center">26.6</td>
  </tr>
  <tr>
    <td align="right">CSDVL<sub>[CVPR'2023]</sub></td>
    <td align="center">-</td>
    <td align="center">33.7</td>
    <td align="center">47.2</td>
    <td align="center">32.8</td>
    <td align="center">-</td>    
    <td align="center">28.5</td>
    <td align="center">39.9</td>
    <td align="center">26.2</td>
  </tr>
  <tr>
    <td align="right">Baseline (ours)</td>
    <td align="center">49.7</td>
    <td align="center">32.4</td>
    <td align="center">45.0</td>
    <td align="center">31.4</td>
    <td align="center">48.8</td>    
    <td align="center">27.7</td>
    <td align="center">38.7</td>
    <td align="center">25.6</td>
  </tr>
  <tr>
    <td align="right">CG-STVG (ours)</td>
    <td align="center"><b>51.4</b> (+1.7)</td>
    <td align="center">32.4</td>
    <td align="center">45.0</td>
    <td align="center">31.4</td>
    <td align="center">48.8</td>    
    <td align="center">27.7</td>
    <td align="center">38.7</td>
    <td align="center">25.6</td>
  </tr>
</table>


