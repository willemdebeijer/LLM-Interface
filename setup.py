from setuptools import find_packages, setup

setup(
    name="llm_interface",
    version="0.1.0",
    author="Willem de Beijer",
    description="A simple LLM interface",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.5.0",
        "aiohttp>=3.9.0",
    ],
)
