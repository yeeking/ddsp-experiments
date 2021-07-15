
# How to set up for the ddsp workshop on mac

Important: this might not work on m1 macs as m1 macs do not currently fully support tensorflow. 

## Install a compatible version of python

Check your python version. Run a terminal and type this command:

```
python3
```
You want 3.8 or less. If you have 3.9, DDSP will probably not work. 

### If your python version is 3.9+

Install home brew:

https://brew.sh/

Run this in a terminal:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
That takes a while. Then install python 3.7:

```
brew install python@3.7
```

Now - if you already had 3.9 installed, you need to make sure you are running the correct version later. So in the following python commands, use this python:

```
/usr/local/opt/python@3.7/bin/python3
```

## Set up a virtual environment

Now you need to set up a python virtual environment. This allows you to have a local set of packages that you can easily manage, instead of installing everything globally. Run this to setup your virtualenv:


```
python3.7 -m venv ~/Python/ddsp-env
```

Or

```
/usr/local/opt/python@3.7/bin/python -m venv ~/Python/ddsp-env
```

Then enter your virtualenv:

```
source ~/Python/ddsp-env/bin/activate
```
You should see a slightly different command prompt, like:

```
(ddsp-env) matthews-imac:ddsp-workshop matthewyk$ 
```

## Setup the ddsp packages

Now we have a working python and a working python virtual environment, we are ready to install ddsp. 

First off, in your virtualenv, upgrade the pip package manager:

```
pip install --upgrade pip
```

Then just go for it:

```
pip install ddsp
```

For some reason, this installs tensorflow 2.4 as a dependency, when it actually wants tf 2.5, so do this:

```
pip install --upgrade tensorflow
```

I saw a warning about crepe wanting h5py 3.0 but TF 2.5 wants 3.1. I ignored this. 

Test it out:

```
python
import ddsp
ddsp.__version__
1.6.0
```

