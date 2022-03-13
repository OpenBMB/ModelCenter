from setuptools import setup, find_packages
import os

def main():
    setup(
        name='model_center',
        version='0.1.0',
        description="example codes for big models using bmtrain",
        author="Weilin Zhao",
        author_email="acha131441373@gmail.com",
        packages=find_packages(),
        url="https://github.com/OpenBMB/BModelCenter",
        install_requires=[
            "bmtrain",
            "cpm_kernels",
            "transformers",
        ],
        keywords="CPM, cuda, AI, model, transformer",
        license='Apache 2.0',
    )

if __name__ == '__main__':
    main()