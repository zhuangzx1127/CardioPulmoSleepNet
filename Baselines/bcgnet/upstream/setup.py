from setuptools import setup, find_namespace_packages

VERSION = "0.1.0"
DESCRIPTION = (
    "BCGNet: Pretrained Polysomnogram AI-Model Helps Ballistocardiogram for Contactless Sleep Monitoring Built on 595,947 Hours of Sleep Recordings."
)
LONG_DESCRIPTION = (
    "BCGNet: Pretrained Polysomnogram AI-Model Helps Ballistocardiogram for Contactless Sleep Monitoring Built on 595,947 Hours of Sleep Recordings."
)

setup(
    name="wuji-dl",
    version=VERSION,
    author="Yuxin Chen",
    author_email="itsme_yxchen@outlook.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_namespace_packages(include=("wuji_*",)),
    install_requires=[
        "astropy<6",
        "fire",
        "hrv-analysis",
        "lightning",
        "omegaconf",
        "pandas",
        "rich",
        "torchmetrics",
        "wandb",
        "wuji",
        'pywavelets',
        'torchaudio',
        'transformers'
    ],
    keywords=["python", "deep learning"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    entry_points={
        "console_scripts": [
            "wuji-dl = wuji_dl.cli:main",
            "wuji-fit = wuji_dl.cli:fit",
            "wuji-test = wuji_dl.cli:test",
            "wuji-preprocess = wuji_preprocess.cli:main",
            "wuji-index = wuji_index.cli:main",
            "wuji-export-onnx = wuji_dl.cli:export_onnx"
        ]
    },
)
