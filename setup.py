from setuptools import setup, find_packages
import os

def main():
    setup(
        name='model_center',
        version='0.1.0',
        description="example codes for big models using bmtrain",
        packages=find_packages(),
        install_requires=[
            "bmtrain",
            "cpm_kernels",
            "transformers",
        ],
    )

if __name__ == '__main__':
    main()