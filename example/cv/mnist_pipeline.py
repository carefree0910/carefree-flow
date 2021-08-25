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
