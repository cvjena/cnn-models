# Residual Networks Models by CVGJ 

## Description
This folder contains Residual Network (ResNet) [1] models trained on ImageNet by Marcel Simon at the Computer Vision Group Jena (CVGJ) using the Caffe framework. Each subfolder contains a model of different depth. We use the pre-activation variant and mostly followed the FAIR implementation [2], which is available for Torch. The generation script was based on a previous implementation of Oscar Beijbom [3].

## How to use
**No mean subtraction is required** for the pre-trained models! We use a batch-normalization layer which basically does the same. 

The pre-trained models can be obtained by the download link written in `model_download_link.txt`. 

The network definition files were generated using the Python 3 script `resnet_preact.py`. You can generate your own network file using the following Python 3 code: 
    
```
import resnet_preact
proto = resnet_preact.residual_net(50)
open('train.prototxt','w').write(str(proto.to_proto()))
```

If you want to train a generated model, copy the solver from our pre-trained models and simply execute `caffe train --solver train.solver --gpu 0 2> train.log` to start the training and write the output to the log file `train.log`.

To evaluate the final model, execute `caffe train --solver test.solver --gpu 0 2> test.log`.


## Accuracy on ImageNet
**Single-crop** error rates on the validation set of the ILSVRC 2012--16 classification task.

| Model             | Top-1 error  (vs. original) |  Top-5 error  (vs. original) |
| ------------- |-------------|---------|-------------|---------|
| ResNet10_cvgj    | **36.1%**    | **14.8%**  |
| ResNet50_cvgj    | **24.6%** (vs. 24.7%)   | **7.6%** (vs. 7.8%)|


## Citation
Please cite the following [technical report](https://arxiv.org/abs/1612.01452 "ImageNet pre-trained models with batch normalization by Marcel Simon et al on arxiv.") if our models helped your research:

```
@article{simon2016cnnmodels,
  Author = {Simon, Marcel and Rodner, Erik and Denzler, Joachim},
  Journal = {arXiv preprint arXiv:1612.01452},
  Title = {ImageNet pre-trained models with batch normalization},
  Year = {2016}
}
```

The report also contains an overview and analysis of the models shown here.

## References
[1]: He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015).
[2]: https://github.com/facebook/fb.resnet.torch
[3]: https://github.com/beijbom/beijbom_vision_lib

