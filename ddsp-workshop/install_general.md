# General instructions to run our scripts

Once you have followed the setup instructions for DDSP, you are ready to run our scripts, nearly! 

To run our network-bending scripts, install some extra packages inside your virtualenv:

```
pip install websockets sounddevice
```

## Get the code for the workshop

Now clone our git repository somewhere:

```
cd ~/src
git clone https://github.com/yeeking/ddsp-experiments.git
```
Or download the zip from the git repo:

https://github.com/yeeking/ddsp-experiments

and unzip it. You should end with a folder called ddsp-experiments. 

## Run the script

Go into that folder:

```
cd ddsp-experiments/ddsp-workshop
```

Now you will need an example audio file. Record something on your mic like some singing and then save it as test.wav in that ddsp-workshop folder. Then run the script like this:

```
python gui_ws.py -m ../models/Flute2021New/ddsp-solo-instrument -i test.wav 
```

Notes: 
-m is the model. We have included a flute model in the repo
-i is the input audio file. For now, we will resynthesize that file with the flute model. Later we will try other models and some other inputs.

