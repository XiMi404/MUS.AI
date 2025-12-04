"""Setup script for museum RAG pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="museum-rag-pipeline",
    version="1.0.0",
    author="Museum RAG Team",
    author_email="contact@museum-rag.com",
    description="RAG pipeline for personalized museum recommendations in Moscow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/museum-rag/museum-rag-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "museum-rag=src.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)