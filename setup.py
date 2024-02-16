from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="CNVoyant",
    version="1.0.18",
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
        # 'numpy==1.23.4',
        # 'onnxruntime==1.12.1',
        # 'pandas==1.5.0',
        # 'progressbar==2.5',
        # 'requests==2.28.1',
        # # 'setuptools==65.6.3',
        # 'xgboost==1.7.1',
        # 'pyvcf3==1.0.3',
        # 'scikit-learn==1.1.1',
        # 'pickleshare==0.7.5',
        # 'uuid==1.30',
        # 'pysam==0.22.0'
        'numpy',
        'onnxruntime',
        'pandas',
        'progressbar',
        'requests',
        # 'setuptools==65.6.3',
        # 'xgboost',
        'pyvcf3',
        'scikit-learn==1.3.2',
        'pickleshare',
        'uuid',
        'pysam',
        'shap',
        'pyarrow',
        'matplotlib'
    ],
    python_requires=">=3.6",
    include_package_data=True
)
