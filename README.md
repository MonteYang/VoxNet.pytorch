# VoxNet.pytorch

A PyTorch implement of "VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition".

## Prepare Data

1. Download `ModelNet10` dataset, unzip in `data/`.

    like `data/ModelNet10/bathtub` ...

2. Convert `*.off` file to `*.binvox` file.
   ```shell
   cd utils
   python off2binvox.py
   ```

## Train
Train VoxNet and the model weights will output in `cls/`
```shell
python train.py
```

## Result
```shell
final accuracy 0.8588807785888077
```

## Reference
```
@inproceedings{maturana_iros_2015,
    author = "Maturana, D. and Scherer, S.",
    title = "{VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition}",
    booktitle = "{IROS}",
    year = "2015",
    pdf = "/extra/voxnet_maturana_scherer_iros15.pdf",
}
```