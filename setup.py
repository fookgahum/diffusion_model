from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="diffusion_model",
    version="0.1.0",
    author="fookgahum",
    author_email="your.email@example.com",
    description="扩散模型研究与实现",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fookgahum/diffusion_model.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
)
 