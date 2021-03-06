import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FF-ANN",
    version="0.0.1",
    author="Armin Najarpour Foroushani",
    author_email="armin.najarpour@gmail.com",
    description="A package for building feed forward neural networks from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Armin12/FF_ANN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)