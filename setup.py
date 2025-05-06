from setuptools import setup, find_packages

setup(
    name="mka-design",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.1",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tqdm>=4.65.0",
        "numpy<2.0.0",
        "pytest>=7.3.1",
        "accelerate>=0.20.0",
        "deepspeed>=0.9.0",
    ],
    python_requires=">=3.10",
) 