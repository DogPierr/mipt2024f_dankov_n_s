from setuptools import setup, find_packages

setup(
    name="barcode-localizer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["opencv-python", "numpy", "pandas"],
    author="Dankov Nikita",
    author_email="vegaspro1097@mail.ru",
    description="A package to localize barcodes in images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/barcode-localizer",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
