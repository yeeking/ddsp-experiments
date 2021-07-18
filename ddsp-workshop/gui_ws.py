#### Thsi script written by Louis McCallum and Matthew Yee-King
#### Original version here: https://github.com/Louismac/network-bending

from tkinter import *
from tkinter.ttk import Button, Entry, Style
DRIVE_DIR = ''
import os
from NetworkBender import Generator
from expand_gpu_memory import expand
import numpy as np
import threading
import json
import sys, argparse
import ws_server as ws

class NetworkBenderFrame(Frame):

    def __init__(self, input_args):
        super().__init__()
        self.input_args = input_args
        self.bg = "white"
        self.params = {
           "ablate":[],
           "invert":[],
           "oscillate":["depth","freq"],
           "threshold":["thresh"]
        }
        self.init_ui()
        self.update_ui_from_file()

    def set_audio_features(self, audio_features):
        """
        pass the sent audio features to the generator
        so that it starts using them for resynthesis
        should assert you are not sending it garbage
        """
        # assert type(audio_features) == np.ndarray, "set_audio_features:: audio features should be numpy.ndarray but are " + str(type(audio_features))
        # assert len(audio_features) > 0, "set_audio_features:: audio features should be len > 0 but are " + str(len(audio_features))
        # assert type(audio_features[0]) == dict, "set_audio_features:: audio features should be array of dicts but are " + str(type(audio_features[0]))
        # assert "loudness_db" in audio_features[0], "set_audio_features:: audio features should have loudness_db key but keys are:" + str(audio_features[0].keys())
        # assert "f0_hz" in audio_features[0], "set_audio_features:: audio features should have f0_hz key but keys are:" + str(audio_features[0].keys)
        # assert type(audio_features[0]["loudness_db"]) == np.ndarray, "set_audio_features:: f0_hz should be np.ndarray but is" + str(type(audio_features[0]["loudness_db"]))
        #
        # assert type(audio_features[0]["f0_hz"]) == np.ndarray, "set_audio_features:: f0_hz should be np.ndarray but is" + str(type(audio_features[0]["f0_hz"]))
        #print("sending audio features from gui class", audio_features)
        #Update the audio features so they get pulled into a buffer
        self.g.new_audio_features(audio_features)

    def get_audio_features(self):
        """
        retrieve the set of audio features currently being used for resynthesis
        """
        return self.g.audio_features

    def stop(self):
        self.g.stop = True;

    def run(self):
        model_name = self.input_args["model"]
        DRIVE_DIR = '.'
        if DRIVE_DIR:
            MODEL_DIR = model_name # os.path.join(DRIVE_DIR, 'Models/' + model_name)
            AUDIO_DATA_DIR = "" #os.path.join(DRIVE_DIR, 'audio_data/')

        expand()

        class RunModelTask:
            def __init__(self):
                self._running = True
            def terminate(self):
                self._running = False
            def run(self, action):
                action()

        def run_model():
            print("gui_ws run_model")
            samplerate = 16000
            input_file = AUDIO_DATA_DIR + self.input_args["input_audio"]
            config = {}
            config["model_dir"] = MODEL_DIR
            config["midi_port"] = self.input_args["midi_port"] 
            config["sample_rate"] = samplerate
            #pick how much of input file to do (0->1)
            config["features"] = {"file_name":input_file, "start":0, "end":1}
            #add boost to loudness feature of input
            config["db_boost"] = 10
            #4 secs at 16000
            config["audio_callback_buffer_length"] = self.input_args["audio_callback_buffer_length"]
    
            # config["audio_callback_buffer_length"] = int(samplerate) * 5
            #Number of frames put into the model at a time
            config["frames"] = 250
            self.update_transforms_from_ui()
            config["transforms"] = self.transforms
            if not hasattr(self, 'g'):
                self.g = Generator()
                self.g.on_update_transforms = self.update_ui_from_dict
                self.g.check_config(config)
                # step 1: write features to CSV file
                self.feature_csvfile = self.g.extract_features_and_write_to_file(input_file)
                # step 2: do the resynthesis
                audio_gen = self.g.start_midi(self.feature_csvfile, input_file, config)

            else:
                self.g.update_config(config)

        c = RunModelTask()
        t = threading.Thread(target = c.run, args = (run_model,))
        t.start()

    def update_ui_from_file(self):
        try:
            with open('data.json',) as f:
                data = json.load(f)
                self.update_ui_from_dict(data)
                f.close()
        except:
            print("error opening file")

    def update_ui_from_dict(self, data):

        """
        Update UI given a config object, either from json file or elsewhere
        """
        for i in range(self.NUM_TRANSFORMS):
            for j in range(2):
                self.gui_elements[i]["param"][j]["param_label"].config(text = "NA")
        for i, t in enumerate(data):
            self.gui_elements[i]["layer"].set(t["layer"])
            self.gui_elements[i]["transform"].set(t["function"])
            self.gui_elements[i]["unit_value"].set(t["units"]["value"])
            params = self.params[t["function"]]
            if "midi" in t["units"].keys():
                self.gui_elements[i]["unit_midi"].set(t["units"]["midi"]["cc"])

            for j, p in enumerate(t["params"]):
                self.gui_elements[i]["param"][j]["param_value"].set(p["value"])
                self.gui_elements[i]["param"][j]["param_label"].config(text = params[j])
                if "midi" in p.keys():
                    self.gui_elements[i]["param"][j]["param_midi"].set(p["midi"]["cc"])
                    self.gui_elements[i]["param"][j]["param_min"].set(p["midi"]["min"])
                    self.gui_elements[i]["param"][j]["param_max"].set(p["midi"]["max"])
                if "lfo" in p.keys():
                    self.gui_elements[i]["param"][j]["param_lfo"].set(p["lfo"]["freq"])
                    self.gui_elements[i]["param"][j]["param_min"].set(p["lfo"]["min"])
                    self.gui_elements[i]["param"][j]["param_max"].set(p["lfo"]["max"])

    def update_transforms_from_ui(self):

        """
        Update the config object from the values in the GUI
        """

        config = []

        for row in self.gui_elements:
            c = {}
            layer = row["layer"].get()
            if not layer == "None" and len(layer) > 0:
                c["layer"] = layer
                c["function"] = row["transform"].get()
                if not c["function"] == "None" and len(c["function"]) > 0:
                    c["units"] = {}
                    c["units"]["value"] = float(row["unit_value"].get())
                    unit_midi = row["unit_midi"].get()
                    if len(unit_midi) > 0:
                        c["units"]["midi"] = {}
                        c["units"]["midi"]["cc"] = int(unit_midi)
                    c["params"] = []
                    for i, param in enumerate(self.params[c["function"]]):
                        p = {}
                        p["name"] = param
                        p["value"] = float(row["param"][i]["param_value"].get())
                        param_midi = row["param"][i]["param_midi"].get()
                        if not param_midi == "":
                            p["midi"] = {}
                            p["midi"]["cc"] = int(param_midi)
                            p["midi"]["min"] = float(row["param"][i]["param_min"].get())
                            p["midi"]["max"] = float(row["param"][i]["param_max"].get())
                        param_lfo = row["param"][i]["param_lfo"].get()
                        print("param_lfo",param_lfo)
                        if not param_lfo == "":
                            p["lfo"] = {}
                            p["lfo"]["freq"] = float(param_lfo)
                            p["lfo"]["min"] = float(row["param"][i]["param_min"].get())
                            p["lfo"]["max"] = float(row["param"][i]["param_max"].get())
                        c["params"].append(p)

                    config.append(c)
        self.transforms = config
        jsonString = json.dumps(config)
        jsonFile = open("data.json", "w")
        jsonFile.write(jsonString)
        jsonFile.close()
        print(self.transforms)

    def option_changed(self, *args):
        #print(f'You selected: {args}')
        self.update_transforms_from_ui()
        self.update_ui_from_dict(self.transforms)

    def init_ui(self):

        self.master.title("Louis")

        Style().configure("TButton", padding=(0, 5, 0, 5),
            font='serif 10')

        self.NUM_TRANSFORMS = 5

        layer_options = [
            "None",
            "FC1",
            "GRU",
            "FC2",
        ]

        transform_options = [
            "None",
            "ablate",
            "oscillate",
            "threshold",
            "invert",
        ]

        self.gui_elements= []

        button = Button(
           self,
           text="Update",
           command=self.run
        )
        button.grid(column=0, row = self.NUM_TRANSFORMS)

        button = Button(
           self,
           text="Stop",
           command=self.stop
        )
        button.grid(column=0, row = 0)

        Label(self, text="layer").grid(row=0,column=1)
        Label(self, text="transform").grid(row=0,column=2)
        Label(self, text="units/value").grid(row=0,column=3)
        Label(self, text="midi").grid(row=0,column=4)
        Label(self, text="lfo_freq").grid(row=0,column=5)
        Label(self, text="min").grid(row=0,column=6)
        Label(self, text="max").grid(row=0,column=7)

        for i in range(self.NUM_TRANSFORMS):
            var_dict = {}
            var_dict["layer"] = StringVar()
            var_dict["transform"] = StringVar()
            var_dict["unit_value"] = StringVar()
            var_dict["unit_value"].set("0.5")
            var_dict["unit_midi"] = StringVar()
            var_dict["param"] = []
            for j in range(2):
                param_dict = {}
                param_dict["param_value"] = StringVar()
                param_dict["param_value"].set("0.5")
                param_dict["param_midi"] = StringVar()
                param_dict["param_min"] = StringVar()
                param_dict["param_max"] = StringVar()
                param_dict["param_lfo"] = StringVar()
                param_dict["param_label"] = Label(self, text="NA")
                var_dict["param"].append(param_dict)
            self.gui_elements.append(var_dict)

            r = (i*3)+1
            layer_menu = OptionMenu(self, var_dict["layer"] , *layer_options, command=self.option_changed)
            layer_menu.config(width=10)
            layer_menu.grid(row=r, column=1)
            transform_menu = OptionMenu(self, var_dict["transform"] , *transform_options, command=self.option_changed)
            transform_menu.config(width=10)
            transform_menu.grid(row=r, column=2)

            ENTRY_WIDTH = 8

            unit_value = Entry(self, textvariable = var_dict["unit_value"], width=ENTRY_WIDTH)
            unit_value.grid(row=r, column=3)
            unit_midi = Entry(self, textvariable = var_dict["unit_midi"], width=ENTRY_WIDTH)
            unit_midi.grid(row=r, column=4)

            for j in range(2):
                label = var_dict["param"][j]["param_label"]
                label.grid(row=r+(j+1),columnspan=2,column=1)
                param_value = Entry(self, textvariable = var_dict["param"][j]["param_value"], width=ENTRY_WIDTH)
                param_value.grid(row=r+(j+1), column=3)
                param_midi = Entry(self, textvariable = var_dict["param"][j]["param_midi"], width=ENTRY_WIDTH)
                param_midi.grid(row=r+(j+1), column=4)
                param_lfo = Entry(self, textvariable = var_dict["param"][j]["param_lfo"], width=ENTRY_WIDTH)
                param_lfo.grid(row=r+(j+1), column=5)
                param_min = Entry(self, textvariable = var_dict["param"][j]["param_min"], width=ENTRY_WIDTH)
                param_min.grid(row=r+(j+1), column=6)
                param_max = Entry(self, textvariable = var_dict["param"][j]["param_max"], width=ENTRY_WIDTH)
                param_max.grid(row=r+(j+1), column=7)

        self.pack()


def ws_msg_to_audio_features(msg):
    """
    parses messages received from the websocket client
    assumes they are JSON like this: "{"f0_hz":[100, 200], "loudness_db":[-10, -50]}"
    """
    data = {}
    try:
        data = json.loads(msg)
    except:
        print("gui_ws.py::ws_message msg is not good JSON. Here's what you sent", msg)
        return
    # # should be good
    # print(type(data))
    # if "f0_hz" in data.keys():
    #     print("f0_hz good")
    # else:
    #     print("f0_hz bad")
    #     return
    # if "loudness_db" in data.keys():
    #     print("loudness_db good")
    # else:
    #     print("loudness_db bad")
    #     return
    # # set up the types
    data['f0_hz'] = data['f0_hz']
    data['loudness_db'] = data['loudness_db']

    return data


def main(config):
    root = Tk()
    app = NetworkBenderFrame(config)
    print("gui_ws.py::main running the websocket")
    # define a callback in same scope so can access app
    def ws_responder(msg):
        """
        callback for the websocket - putting it here so we have
        the app object in scope, since app has all the interesting stuff in it
        """
        #print("ws_responder: ", msg)
        if hasattr(app, 'g'):
            #af = app.get_audio_features()
            # print("app ready - feature type:", type(af))
            # print("app ready - feature [0] type:", type(af[0]))
            # print("app ready - feature [0] keys:", af[0].keys())
            # print("app ready - feature [0] len:", af[0]["f0_hz"])
            af = ws_msg_to_audio_features(msg)
            #af = np.array([{"f0_hz":np.array([400, 400]), "loudness_db":np.array([-20, -40])}])
            if af is not None:
                app.set_audio_features(af)
        else:
            print("app not ready yet")
    # fire up the websocket with our callback
    ws.run_websocket(ws_responder)

    root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_audio", help="name of file in audio_data directory")
    parser.add_argument("-p","--midi_port", help="name of midi port to connect to")
    parser.add_argument("-m","--model",help="name of folder containing model checkpoint in Models folder")
    args = parser.parse_args()
    config = {
        "midi_port":"",
        "model":"Flute2021New",
        "input_audio":"i_will_always_1min.wav",
       "audio_callback_buffer_length":16000 * 10
    }
    print(args)
    if args.input_audio:
        config["input_audio"] = args.input_audio
    if args.model:
        config["model"] = args.model
    if args.midi_port:
        config["midi_port"] = args.midi_port
    # do some asserts:
    #assert os.path.exists(config["input_audio"]), "Input audio does not exist: " + config["input_audio"]
    assert os.path.exists(config["model"]), "Model folder does not exist: " + config["model"]
    # check for the gin file
    gin_file = os.path.join(config["model"], "operative_config-0.gin")
    assert os.path.exists(gin_file), "Model gin file not in folder " + gin_file
    print("gui_ws.py:: running main")
    main(config)
