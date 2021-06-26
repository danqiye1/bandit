import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bandits", # Replace with your own username
    version="0.0.1",
    author="Ye Danqi",
    author_email="john@example.com",
    description="An implementation of common multi-armed bandit and reinforcement learning algorithms to gridworld, cartpole, and stock trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn'
    ],
    python_requires=">=3.9",
)