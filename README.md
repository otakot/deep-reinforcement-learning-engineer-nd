# Sumamry

This repository contains the source code implementation for the projects of Deep Reinforcement Learning Engineer Nano Degree educational course.

For detailed description of each project and project specific execution environemnt setup instructions please refer to README.MD file located in corresponding project folder

## Common execution environment setup

To be able to execute the source code of any of projects in this repository, before applying project specific execution/training environment setup instructions, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__:
	```bash
	conda create --name drlnd python=3.6
	activate drlnd
	```
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/).

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.

4. Clone this repository (if you haven't already), and navigate to the `python/` folder.  Then, install several dependencies.
    ```bash
    git clone https://github.com/otakot/deep-reinforcement-learning-engineer-nd.git
    cd deep-reinforcement-learning-engineer-nd/python
    pip install .
    ```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

6. Before running project code in a Jupyter notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.
