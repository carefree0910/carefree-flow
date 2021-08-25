![carefree-flow][socialify-image]

Deep Learning with [OneFlow](https://github.com/OneFlow-Inc/oneflow) made easy 🚀 !


## Carefree?

`carefree-learn` aims to provide **CAREFREE** usages for both users and developers.

### User Side

#### Computer Vision 🖼️

```python
# MNIST classification task with LeNet

import cflow

import numpy as np
import oneflow.data as data


(x_train, y_train), (x_test, y_test) = data.load_mnist()
x_train, x_test = np.concatenate(x_train, axis=0), np.concatenate(x_test, axis=0)
y_train = np.concatenate(y_train, axis=0)[..., None]
y_test = np.concatenate(y_test, axis=0)[..., None]

data = cflow.cv.TensorData(x_train, y_train, x_test, y_test)
m = cflow.cv.CarefreePipeline(
    "clf",
    dict(
        in_channels=1,
        num_classes=10,
        img_size=28,
        latent_dim=128,
        encoder1d="lenet",
    ),
    fixed_epoch=5,
    loss_name="cross_entropy",
    metric_names=["acc", "auc"],
    tqdm_settings={"use_tqdm": True, "use_step_tqdm": True},
)
m.fit(data, cuda=0)
```

### Developer Side

> This is a WIP section :D


## Installation

`carefree-flow` requires Python 3.6 or higher.

### Pre-Installing OneFlow

`carefree-flow` requires `oneflow>=0.4.0`. Please refer to [OneFlow](https://github.com/OneFlow-Inc/oneflow) for pre-installation.

### pip installation

After installing OneFlow, installation of `carefree-flow` would be rather easy:

```bash
git clone https://github.com/carefree0910/carefree-flow
cd carefree-flow
pip install -e .
```


## Citation

If you use `carefree-flow` in your research, we would greatly appreciate if you cite this library using this Bibtex:

```
@misc{carefree-flow,
  year={2021},
  author={Yujian He},
  title={carefree-flow, Deep Learning with OneFlow made easy},
  howpublished={\url{https://https://github.com/carefree0910/carefree-flow/}},
}
```


## License

`carefree-flow` is MIT licensed, as found in the [`LICENSE`](https://github.com/carefree0910/carefree-flow/blob/main/LICENSE) file.


[socialify-image]: https://socialify.git.ci/carefree0910/carefree-flow/image?description=1&descriptionEditable=Deep%20Learning%20%E2%9D%A4%EF%B8%8F%20OneFlow&forks=1&issues=1&language=1&owner=1&stargazers=1&theme=Light