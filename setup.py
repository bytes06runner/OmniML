from setuptools import setup, find_packages

setup(
    name="anomallm",
    version="0.1.0",
    description="Explainable Anomaly Detection SDK for Multivariate Time Series using LLMs",
    author="Hackathon Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0",
        "statsmodels>=0.13.0",
        "networkx>=2.6",
        "openai>=1.0.0"
    ],
    python_requires=">=3.8",
)
