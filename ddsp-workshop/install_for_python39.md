
First off, set up libs so that the python you are about to build has the right stuff for ddsp:

```
sudo apt install libbz2-dev libssl-dev liblzma-dev
```

Install pyenv:
https://github.com/pyenv/pyenv-installer

https://stackoverflow.com/questions/59549829/how-do-i-downgrade-my-version-of-python-from-3-7-5-to-3-6-5-on-ubuntu

```
curl https://pyenv.run | bash

# install 3.7.0
~/.pyenv/bin/pyenv install 3.7.0
# activate 3.7.0
~/.pyenv/bin/pyenv local 3.7.0
# create a virtual env with 3.7.0
~/.pyenv/bin/pyenv virtualenv myvirtualenv
# activate it 
~/.pyenv/bin/pyenv activate myvirtualenv
# verify python version
python --version
Python 3.7.0

```
Now I found that next time I ran it I could not get the virtualenv to start up.
So instead I just install 3.7.0 with pyenv, then use the following commands to add it to tha path:

```
export PATH=$PATH:~/.pyenv/bin:~/.pyenv/versions/3.7.0/bin/
#eval "$(pyenv init -)"  
#eval "$(pyenv virtualenv-init -)"
```



# go for it!
pip install ddsp
# I found this installed TF 2.4 so I also upgraded that
pip install --upgrade tensorflow
# finally:
python
import ddsp
ddsp.__version__
1.2.0
```


