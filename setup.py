from setuptools import setup, find_packages

setup(
    name="curriculum-augmentation",
    version="0.1.0",
    author="Aseel Qazqzy",
    description="Curriculum-Style Augmentation for Image Classification",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "timm>=0.9.0",
        "albumentations>=1.3.0",
        "rich>=13.0.0",
    ],
)