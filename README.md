<div align="center">

# Content-Aware Transformer for All-in-one Image Restoration

[Gang Wu](https://scholar.google.com/citations?user=JSqb7QIAAAAJ), [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun), [Kui Jiang](https://homepage.hit.edu.cn/jiangkui), and [Xianming Liu](http://homepage.hit.edu.cn/xmliu)

[AIIA Lab](https://aiialabhit.github.io/team/), Harbin Institute of Technology.


</div>

## Overview

> Image restoration has witnessed significant advancements with the development of deep learning models. Although Transformer architectures have progressed considerably in recent years, challenges remain—particularly the limited receptive field in window-based self-attention. In this work, we propose DSwinIR, a Deformable Sliding window Transformer for Image Restoration. DSwinIR introduces a novel deformable sliding window self-attention that adaptively adjusts receptive fields based on image content, enabling the attention mechanism to focus on important regions and enhance feature extraction aligned with salient features. Additionally, we introduce a central ensemble pattern to reduce the inclusion of irrelevant content within attention windows. In this way, the proposed DSwinIR model integrates the deformable sliding window Transformer and central ensemble pattern to amplify the strengths of both CNNs and Transformers while mitigating their limitations. Extensive experiments on various image restoration tasks demonstrate that DSwinIR achieves state-of-the-art performance. For example, in image deraining, compared to DRSformer on the SPA dataset, DSwinIR achieves a 0.66 dB PSNR improvement. In all-in-one image restoration, compared to PromptIR, DSwinIR achieves over a 0.66 dB and 1.04 dB improvement on three-task and five-task settings, respectively. 


<div align="center">
<a href="https://www.imagehub.cc/image/Motivation.CNC9WO"><img src="https://s1.imagehub.cc/images/2024/11/18/2cc9c26c3487167a531af447c370dab4.png" alt="Motivation" border="0" width="60%"></a>
</div>

### Framework
<div align="center">
<a href="https://www.imagehub.cc/image/framework.CNCvda"><img src="https://s1.imagehub.cc/images/2024/11/18/d22e55d3bb561d9dc5250b9aded16b18.png" alt="framework" border="0"></a>


<a href="https://www.imagehub.cc/image/DSwin.CNCTyA"><img src="https://s1.imagehub.cc/images/2024/11/18/db37ed9079a54c662af20c76f4951abc.png" alt="DSwin" border="0"></a>
</div>

### Requirements
Our project is based on Basicsr, for the basic requirements, you can install this project
```
pip install basicsr
```

To achieve the fast implementation, we adopt the NATTEN implementation.
```
pip3 install natten==0.17.1+torch220cu118 -f https://shi-labs.com/natten/wheels/
```

## Datasets

### All-in-One Image Restoration

|Setting| Dataset|
|---|---|
|Noise-Rain-Haze|[WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar),[BSD](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing),[Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing),[OTS](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)|
|AllWeather|[Download](https://drive.google.com/file/d/1tfeBnjZX1wIhIFPl6HOzzOKOyo0GdGHl/view?usp=sharing)|

## Results
<div align="center">
<a href="https://www.imagehub.cc/image/improvement.CNCVqk"><img src="https://s1.imagehub.cc/images/2024/11/18/069cf923af8df50976af515a923ed54c.png" alt="improvement" border="0" width="90%"></a>
</div>


### All-in-one Image Restoration
|AllWeather| Outdoor-Rain | RainDrop | Snow-L| Snow-S|
|---|---|---|---|---|
|DSwinIR| [Download](https://drive.google.com/drive/folders/1V6V4jcnyUeUE_iqF0JDbQsT0Fd4Dm-ee?usp=sharing)|[Download](https://drive.google.com/drive/folders/1V6V4jcnyUeUE_iqF0JDbQsT0Fd4Dm-ee?usp=sharing)|[Download](https://drive.google.com/drive/folders/1V6V4jcnyUeUE_iqF0JDbQsT0Fd4Dm-ee?usp=sharing)|[Download](https://drive.google.com/drive/folders/1V6V4jcnyUeUE_iqF0JDbQsT0Fd4Dm-ee?usp=sharing)|



### Single-Task Image Restoration
|Task|PSNR/SSIM|Results|Pretrained|
|---|---|---|---|
|Rain100L|38.19/0.984|[Results](https://drive.google.com/file/d/1jCBqB3C32OfqyOyfUYYAd7asUisfl_Ox/view?usp=sharing)|[Model](https://drive.google.com/file/d/1yQcnuRUepg9NW5DVh3MmBFlA-H1r7y7E/view?usp=sharing)|

|Task|PSNR/SSIM|Results|
|---|---|---|
|SPA|49.19/0.993|[Results](https://drive.google.com/file/d/1qj2RkgTSDnA_1a-jzztvdVd4gavDb-3B/view?usp=sharing)|


## Acknowledgment

We think a lot for their nice sharing, including [DRSformer](https://github.com/cschenxiang/DRSformer?tab=readme-ov-file), [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet), [Transweather](https://github.com/jeya-maria-jose/TransWeather), [Histoformer](https://github.com/sunshangquan/Histoformer/tree/main), and [BasicSR](https://github.com/XPixelGroup/BasicSR).


