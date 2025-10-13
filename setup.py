from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="financial-timeseries-forecast",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Framework modulare per previsioni finanziarie con modelli time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/financial-forecast",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "forecast=src.cli:main",  # Se implementi una CLI
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
)
