from setuptools import find_packages, setup


setup(
    name="stmc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "hydra-core",
        "torch",
        "wandb",
        "loguru",
        "omegaconf",
        "qqdm",
        "pillow",
        "ramapy",
    ],
    entry_points={
        "console_scripts": [
            "track=tools.track:main",
        ],
    },
    author="Fabian Herzog",
    author_email="fabian.herzog@tum.de",
    description="Spatial-Temporal Multi-Cuts for Online Multiple-Camera Vehicle Tracking",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fubel/stmc",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
