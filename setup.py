from setuptools import setup, find_packages

setup(
    name="brism",
    version="0.1.0",
    description="Bayesian Reciprocal ICD-Symptom Model",
    author="Sean",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    python_requires=">=3.8",
)
