# DSwinIR
> **Paper**: DSwinIR: Rethinking Window-based Attention for Image Restoration
> 
> **Author**: [Gang Wu](https://scholar.google.com/citations?user=JSqb7QIAAAAJ), [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun)*, [Kui Jiang](https://homepage.hit.edu.cn/jiangkui), [Xianming Liu](http://homepage.hit.edu.cn/xmliu), and [Liqiang Nie](https://liqiangnie.github.io/)


## Overview
> **TL;DR:** We rethink window-based attention and propose **Deformable Sliding Window (DSwin) Attention** — a token-centric, content-aware mechanism that eliminates boundary artifacts and adapts receptive fields to image content.

- 🎯 **Token-Centric Sliding Window**: Every token attends within its own centered neighborhood, eliminating boundary blindness.
- 🧠 **Content-Aware Deformable Sampling**: Learned offsets dynamically shape the receptive field to focus on informative regions.


## Datasets

### All-in-One Image Restoration

|Setting| Dataset|
|---|---|
|Noise-Rain-Haze|[WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar),[BSD](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing),[Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing),[OTS](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)|
|AllWeather|[Download](https://drive.google.com/file/d/1tfeBnjZX1wIhIFPl6HOzzOKOyo0GdGHl/view?usp=sharing)|

## Results 


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

We thank a lot for their nice sharing, including [DRSformer](https://github.com/cschenxiang/DRSformer?tab=readme-ov-file), [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet), [Transweather](https://github.com/jeya-maria-jose/TransWeather), [Histoformer](https://github.com/sunshangquan/Histoformer/tree/main), and [BasicSR](https://github.com/XPixelGroup/BasicSR).



## 📖 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{wu2025dswinir,
  title={DSwinIR: Rethinking Window-based Attention for Image Restoration},
  author={Wu, Gang and Jiang, Junjun and Jiang, Kui and Liu, Xianming and Nie, Liqiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```

