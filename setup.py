from setuptools import setup, find_packages

setup(
    name="minillmlib",
    version="0.1.0",
    author="Quentin Feuillade--Montixi",
    description="Minimalist Library for prompting LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "openai",
        "anthropic",
        "mistralai",
        "together",
        "colorama",
        "requests",
        "python-dotenv",
        "json_repair",
    ],
    extras_require={
        "huggingface": [
            "torch",
            "transformers",
        ],
        "dev": [
            "pytest"
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
