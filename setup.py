import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sbirl",
    version="1.0.0",
    author="Alex J. Chan",
    author_email="ajc340@cam.ac.uk",
    description="Scalable Bayesian Inverse Reinforcement Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XanderJC/scalable-birl",
    packages=setuptools.find_packages(),
    install_requires=[
        'dm-haiku @ git+https://github.com/deepmind/dm-haiku@7e84ed108fcc644f3a65327fba818bf20b225a8e',
        'gym==0.17.2',
        'jax==0.1.72',
        'jaxlib==0.1.48',
        'numpy==1.19.1',
        'tqdm==4.48.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)