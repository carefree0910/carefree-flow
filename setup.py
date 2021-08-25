from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = "Deep Learning with OneFlow made easy"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-flow",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "pykeops",
        "lmdb",
        "einops",
        "albumentations",
        "scikit-image",
        "mlflow",
        "onnx",
        "onnx-simplifier",
        "onnxruntime",
        "plotly",
        "dill",
        "future",
        "psutil",
        "cython>=0.29.12",
        "numpy>=1.19.2",
        "scipy>=1.2.1",
        "scikit-learn>=0.23.1",
        "matplotlib>=3.0.3",
        "carefree-toolkit>=0.2.7",
    ],
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    url="https://github.com/carefree0910/carefree-flow",
    download_url=f"https://github.com/carefree0910/carefree-flow/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python machine-learning deep-learning solution OneFlow",
)
