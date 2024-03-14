from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="CNVoyant",
    version="1.0.34",
    author="Rob Schuetz",
    author_email="robert.schuetz@nationwidechildrens.org",
    description="Copy Number Variant Pathogenicity Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nch-igm/CNVoyant", 
    project_urls={
        "Bug Tracker": "https://github.com/nch-igm/CNVoyant/issues",
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'onnxruntime',
        'pandas',
        'progressbar',
        'requests',
        'pyvcf3',
        'pyBigWig',
        'pybedtools',
        'scikit-learn==1.3.2',
        'pickleshare',
        'uuid',
        'pysam',
        'shap',
        'pyarrow',
        'matplotlib',
        'tqdm'
    ],
    python_requires=">=3.6",
    include_package_data=True
)
