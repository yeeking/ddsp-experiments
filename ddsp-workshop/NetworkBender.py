
import ddsp
import ddsp.training
import gin
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavutils
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import json
import librosa
import os
import sounddevice as sd
import sys
from datetime import datetime
from time import sleep
import time
import threading
import mido
import datetime

class UnitProvider():
    def __init__(self):
        self.unit_list = []
        self.shuffled_list = []
        self.units = 1;

    def get_units(self, s):
        if len(self.unit_list) == 0:
            self.shuffled_list = np.arange(s)
            np.random.shuffle(self.shuffled_list)
        self.unit_list = self.shuffled_list[:int(s * self.units)]
        return self.unit_list

class BendingParam():
    def __init__(self):
        self.t = 0
        #number of vals in block
        self.res = 1000
        self.unit_list = []
        self.value = 0
        self.freq = 1
        self.len = 1
        self.lfo = {}

    #return 1 block of params
    def get_values(self):
        vals = []
        if len(self.lfo.keys())>0:
            r = (self.lfo["max"] - self.lfo["min"]) / 2
            vals = np.array([self.step_lfo() for i in range(self.res)])
            vals = vals + (1 + self.lfo["min"])
            vals = vals * r
        # elif self.ramp:
        #     vals = np.linspace(self.min, self.max, self.len * self.res)[self.t:self.t+self.res]
        #     self.t = self.t + self.res
        else:
            vals = np.ones(self.res) * self.value
        return vals

    def step_lfo(self):
        increment = (self.lfo["freq"] / self.res) * (np.pi * 2)
        val = np.sin(self.t)
        self.t = self.t + increment
        return val

class BendingTransforms():
    def __init__(self):
        super().__init__()
        self.t = 0
        self.res = 1000

    def ablate(self, src, units):
        src = src.numpy()
        src = src.reshape((src.shape[1], src.shape[2]))
        M, N = src.shape
        units = units.get_units(N)
        src[:,units] = 0
        return src.reshape((1, M, N))

    def invert(self, src, units):
        src = src.numpy()
        src = src.reshape((src.shape[1], src.shape[2]))
        M, N = src.shape
        units = units.get_units(N)
        src[:,units] = 1 - src[:,units]
        return src.reshape((1, M, N))

    def threshold(self, src, thresh, units):
        thresh = thresh.get_values()
        #apply in axis 1 (time)
        thresh = thresh.reshape((thresh.shape[0], 1))
        src = src.numpy()
        one, M, N = src.shape
        src = src.reshape((M, N))
        units = units.get_units(N)
        #print(src[src < t], t, src)
        selected = src[:,units]
        selected[selected < thresh] = src.min()
        selected[selected > thresh] = src.max()
        src[:,units] = selected
        return src.reshape((1, M, N))

    def step_osc(self, f = 1.0):
        increment = (f / self.res) * (np.pi * 2)
        self.t = self.t + increment
        return np.sin(self.t)

    def oscillate(self, src, freq, depth, units):
        src = src.numpy()
        src = src.reshape((src.shape[1], src.shape[2]))
        M, N = src.shape
        f = freq.get_values()
        d = depth.get_values()
        b = np.array([self.step_osc(f[i]) for i in range(0,self.res)]) * d
        #apply in axis 1 (time)
        b = b.reshape(b.shape[0], 1)
        units = units.get_units(N)
        src[:,units] = src[:,units] + b
        return src.reshape((1, M, N))

    # def reflect(self, src, r, units):
    #     alpha = r
    #     a = np.array([[np.cos(2*alpha), np.sin(2*alpha)],
    #                   [np.sin(2*alpha), -np.cos(2*alpha)]])
    #     return self.linear_transformation(src, a)
    #
    # def rotate(self, src, radians, units):
    #     alpha = radians
    #     a = np.array([[np.cos(alpha), -np.sin(alpha)],
    #                   [np.sin(alpha), np.cos(alpha)]])
    #     return self.linear_transformation(src, a)
    #
    # def linear_transformation(self, src, a):
    #     src = src.numpy()
    #     src = src.reshape((src.shape[1], src.shape[2]))
    #     M, N = src.shape
    #     points = np.mgrid[0:N, 0:M].reshape((2, M*N))
    #     new_points = np.linalg.inv(a).dot(points).round().astype(int)
    #     x, y = new_points.reshape((2, M, N), order='F')
    #     indices = x + N*y
    #     wrap = np.take(src, indices, mode='wrap').reshape((1, M, N))
    #     t = tf.constant(wrap)
    #     return t

class BendingDecoder(ddsp.training.decoders.RnnFcDecoder):
    def __init__(self):
        super().__init__()
        print("bending init called")

    def init_params(self):
        print("BendingDecoder init_params")
        self.clear_transforms()

    def clear_transforms(self):
        self.t = {}
        self.t["FC1"] = {}
        self.t["FC2"] = {}
        self.t["GRU"] = {}

    def update_transform(self, layer, name, f, a):
        self.t[layer][name] = tf.keras.layers.Lambda(f, arguments = a)

    def add_transform(self, layer, name, f, a):
        #print("adding transform", layer, name, f, a)
        self.t[layer][name] = tf.keras.layers.Lambda(f, arguments = a)

    def compute_output(self, *inputs):
      # Initial processing.
      #print("BendingDecoder compute_output")
      inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]

      # Run an RNN over the latents.
      x = tf.concat(inputs, axis=-1)
      for k,v in self.t["FC1"].items():
          x = v(x)
      x = self.rnn(x)
      for k,v in self.t["GRU"].items():
            x = v(x)
      x = tf.concat(inputs + [x], axis=-1)

      # Final processing.
      x = self.out_stack(x)
      for k,v in self.t["FC2"].items():
            x = v(x)
      return x

class Generator():
    def __init__(self):
        super().__init__()
        self.layers = ["FC1", "GRU", "FC2"]
        self.transforms = {}
        self.buf_length = 16000
        for l in self.layers:
            self.transforms[l] = BendingTransforms()

    # setup tensorflow, the feature extractor and the model
    def setup_resynthesis(self, model_dir):
        """
        initialises the resynthesis models
        and reset the crepe feature extractor
        """
        #self.setup_tensorflow()
        ddsp.spectral_ops.reset_crepe()
        self.setup_model(model_dir)
        print("setup_resynthesis::resynthesis ready probably")
        self.model.decoder.__class__ = BendingDecoder
        self.model.decoder.init_params()

    def setup_tensorflow(self):
        config = tf.compat.v1.ConfigProto()
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)
        print("setup_tensorflow")

    def setup_model(self, model_dir):
        gin_file = os.path.join(model_dir, 'operative_config-0.gin')

        if os.path.isfile(gin_file) != True:
            print("setup_model::Gin file not found: ", gin_file)
            return

         # Parse gin config,
        with gin.unlock_config():
            gin.parse_config_file(gin_file, skip_unknown=True)

        # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
        ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
        ckpt_name = ckpt_files[0].split('.')[0]
        ckpt = os.path.join(model_dir, ckpt_name)

        # Ensure dimensions and sampling rates are equal
        #time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
        time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
        #n_samples_train = gin.query_parameter('Additive.n_samples')
        n_samples_train = gin.query_parameter('Harmonic.n_samples')
        hop_size = int(n_samples_train / time_steps_train)

        time_steps = int(self.buf_length / hop_size)
        required_input_samples = time_steps * hop_size
        print("time steps", time_steps, time_steps_train)
        print("input_samples", required_input_samples, n_samples_train)

        gin_params = [
            'RnnFcDecoder.input_keys = ("f0_scaled", "ld_scaled", "z")',
            'Additive.n_samples = {}'.format(required_input_samples),
            'FilteredNoise.n_samples = {}'.format(required_input_samples),
            'DefaultPreprocessor.time_steps = {}'.format(time_steps),
        ]

        # with gin.unlock_config():
        #     gin.parse_config(gin_params)

        # Set up the model just to predict audio given new conditioning
        self.model = ddsp.training.models.Autoencoder()
        self.model.restore(ckpt)
        # gin_file = os.path.join(model_dir, 'operative_config-0.gin')
        # gin.parse_config_file(gin_file)
        # self.model = ddsp.training.models.Autoencoder()
        # self.model.restore(model_dir)

    def resynth_batch(self, data_dir):
        TRAIN_TFRECORD = data_dir + '/train.tfrecord'
        TRAIN_TFRECORD_FILEPATTERN = TRAIN_TFRECORD + '*'
        data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
        dataset = data_provider.get_batch(batch_size=1, shuffle=False)

        try:
          batch = next(iter(dataset))
        except OutOfRangeError:
          raise ValueError(
              'TFRecord contains no examples. Please try re-running the pipeline with '
              'different audio file(s).')
        print(batch["f0_hz"].shape)
        audio_gen = self.model(batch, training=False)
        return audio_gen, batch['audio']

    @staticmethod
    def load_audio_data(audio_filename):
        """
        reads all samples from the senf audio_filename
        returns a numpy array of the samples and the sample rate
        """
        signal, sr=librosa.load(audio_filename, sr=16000, mono = True,)
        print("loaded audio file", len(signal))
        return np.array(signal), sr

    def extract_features_and_write_to_file(self, audio_filename):
        """
        looks for a file called audio_filename.csv
        if it does not exist, extracts features
        using the function ddsp.training.metrics.compute_audio_features
        and writes them to that file
        returns the csv filename
        """
        audio_signal, sr = self.load_audio_data(audio_filename)
        feature_filaname = audio_filename + ".csv"
        if not os.path.exists(feature_filaname):
            print("load_features::extracting features from ", audio_filename, ' (slow on CPU!)')
            #audio_features = self.extract_audio_file_features(audio_sig, sr)
            start_time = time.time()
            print('Extracting features (may take a while). Sig length ', len(audio_signal))
            audio_features = ddsp.training.metrics.compute_audio_features(audio_signal, sample_rate=sr)
            print('extract_input_fetures:: Audio features took %.1f seconds' % (time.time() - start_time))
            audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
            stacked = np.stack((audio_features["f0_hz"], audio_features["loudness_db"],audio_features["f0_confidence"]), axis=1)
            df = pd.DataFrame(stacked,columns=["f0_hz","loudness_db","f0_confidence"])
            df.to_csv(feature_filaname)
        else:
            print("features already extracted, found csv")

        return feature_filaname

    def write_file(self, output, config = None, normalise = False, sample_rate = 16000):
        complete_output = np.zeros((2, len(output)))
        complete_output[0] = complete_output[1] = output

        print("main:: synthesis ends..." + str(len(output)))

        now = datetime.datetime.now()
        output_root = now.strftime("%Y%m%d%H%M%S")
        output_audio_file = output_root + ".wav"
        output_json_file = output_root + ".json"
        boost_left = boost_right = 1
        if normalise:
          boost_left = self.get_normalise_scalar(complete_output[0])
          boost_right = self.get_normalise_scalar(complete_output[1])

        if not config == None:
          output_file = os.path.join(AUDIO_DATA_DIR, output_json_file);
          print("writing config to json", output_file)
          with open(output_file, 'w') as outfile:
              json.dump(config, outfile)

        complete_output[0] = complete_output[0] * boost_left
        complete_output[1] = complete_output[1] * boost_right

        amplitude = np.iinfo(np.int16).max
        complete_output = complete_output * amplitude
        #now rotate it from [[ch1...], [ch2...] to [[c1, c2], [c1, c2] ..]
        complete_output = np.rot90(complete_output, 3) # 3 as 1 is reversed
        #complete_output = np.array([int((x + 1) * 32768) for x in complete_output])
        output_path = os.path.join(AUDIO_DATA_DIR, output_audio_file);
        wavutils.write(output_path, sample_rate, complete_output.astype(np.int16))

        print("main:: wrote result to ", output_file)

    def get_normalise_scalar(self, buffer):
        max = 0
        for i in range(len(buffer)):
            if np.abs(buffer[i]) > max:
                max = np.abs(buffer[i])
        scalar = 1/max
        return scalar



    def combine_features_and_audio(self, csv_file, audio_file, samplerate = 16000, start = 0.0, end = 1.0):
        """
        creates a basic data structure containing
        features and audio signal
        assumes the csv_file exists
        more processing is needed before the data can be fed
        to the model. That is done by load_and_prepare_features_for_model
        which actually calls me
        """
        #df = pd.read_csv(os.path.join(AUDIO_DATA_DIR,name + ".csv"))
        df = pd.read_csv(csv_file)
        total = np.array(df["f0_hz"]).shape[0]
        start = int(total * start)
        end = int(total * end)
        print("loaded features from {s} to {e}".format(s=start, e=end))
        features = {}
        features["f0_hz"] = np.array(df["f0_hz"])[start:end]
        features["loudness_db"] = np.array(df["loudness_db"])[start:end]
        features["f0_confidence"] = np.array(df["f0_confidence"])[start:end]
        ## note that I don't think we need to add the original
        ## audio signal to the features that are fed to the model
        #features["audio"] = audio_signal[start:end]
        # audio_signal,samplerate = self.load_audio_data(audio_file)
        # total = len(audio_signal)
        # start = int(total * start)
        # end = int(total * end)
        # we do need the sample rate though
        features["sr"] = samplerate
        return features

    def load_and_prepare_features_for_model(self, csv_file, audio_file, config, floor = True):
        """
        gets the input ready for the model
        loads in the features and the signal
        then prepares it in blocks
        """
        audio_features = self.combine_features_and_audio(
          #config["features"]["file_name"],
          csv_file,
          audio_file,
          16000,
          config["features"]["start"],
          config["features"]["end"]
        )
        self.buf_length = config["audio_callback_buffer_length"]
        self.frames = config["frames"]

        db_boost = config["db_boost"]
        r = np.floor if floor else np.ceil
        steps = r(len(audio_features["f0_hz"]) / self.frames )
        def get_dict(start, af):
            d = {}
            f_start = int(start * self.frames )
            s_start = int(start * self.buf_length)
            d["f0_hz"] = af["f0_hz"][f_start:f_start+self.frames]
            d["loudness_db"] = af["loudness_db"][f_start:f_start+self.frames ] + db_boost
            d["f0_confidence"] = af["f0_confidence"][f_start:f_start+self.frames]
            delta = self.frames - len(d["f0_hz"])
            if delta > 0:
               d["f0_hz"] = np.append(d["f0_hz"], np.zeros(delta))
               d["f0_confidence"] = np.append(d["f0_confidence"], np.zeros(delta))
               d["loudness_db"] = np.append(d["loudness_db"], np.zeros(delta))
            ## note I don't think we need to put the original
            ## audio signal into the features that are fed into the model
            #d["audio"] = [af["audio"][s_start:s_start+self.buf_length]]
            return d

        split = [get_dict(i, audio_features) for i in np.arange(steps)]
        return np.array(split), steps

    def append_audio_features(self, audio_features):
        """
        Write the most recent audio features into the audio_features buffer
        Should happen every 1/self.frames seconds
        """
        self.audio_features[0]["f0_hz"][self.feature_ptr] = audio_features["f0_hz"][0]
        self.audio_features[0]["loudness_db"][self.feature_ptr] = audio_features["loudness_db"][0]
        self.feature_ptr = (self.feature_ptr + 1) % self.frames

    def new_audio_features(self, audio_features):
        """
        Update the latest audio features to pull into audio features
        """
        self.features_to_poll = audio_features;

    @staticmethod
    def check_config(config):
        """
        verify the sent config has the correct fields
        uses assert so it will end execution if anything is missing
        """
        want_keys = ["features", "audio_callback_buffer_length", "frames", "db_boost", "model_dir", "frames"]
        for key in want_keys:
            assert key in config.keys(), "missing config key "+key
            print("check_config::Config has key", key)
        print("check_config::Config looks good")


    def get_transform_args(self, config, existing = None):
        """
        Gets the arguments to pass to the transforms lambda layer
        Optionally takes an existing dicionary (if updating not initialising)
        """
        transform_arguments = {}
        units = 1;
        #What percentage of units to transform
        if "units" in config.keys():
            units = config["units"]["value"]
        if existing == None:
            transform_arguments["units"] = UnitProvider()
        else:
            transform_arguments["units"] = existing["units"]

        transform_arguments["units"].units = units
        if "params" in config.keys():
            #Each transform has different named parameters e.g. thresh, freq etc...
            for p in config["params"]:
                if existing == None:
                    transform_arguments[p["name"]] = BendingParam()
                else:
                    transform_arguments[p["name"]] = existing[p["name"]]
                #transform_arguments[p["name"]].res = self.frames
                transform_arguments[p["name"]].len = int(np.ceil(self.duration))
                transform_arguments[p["name"]].value = p["value"]
                if "lfo" in p.keys():
                    transform_arguments[p["name"]].lfo = p["lfo"]
                #Inherit the properties from the dict and set on the BendingParam object
                if "args" in p.keys():
                    for k,v in p["args"].items():
                        #print(k, v)
                        setattr(transform_arguments[p["name"]], k, v)
        #print("transform_arguments",transform_arguments)
        return transform_arguments

    def update_transforms(self, config):
        """
        updates the network bending transforms to the network
        as specified by config
        """

        self.on_update_transforms(config["transforms"])

        for i, c in enumerate(config["transforms"]):
            args = self.get_transform_args(c)
            function = getattr(self.transforms[c["layer"]], c["function"])
            self.model.decoder.update_transform(c["layer"], str(i), function, args)

    def add_transforms(self, config):
        """
        adds the network bending transforms to the network
        as specified by config
        """

        for i, c in enumerate(config["transforms"]):
            args = self.get_transform_args(c)
            function = getattr(self.transforms[c["layer"]], c["function"])
            self.model.decoder.add_transform(c["layer"], str(i), function, args)

    def start_midi(self,feature_csv_filename, audio_filename, config):
        self.start_realtime(feature_csv_filename, audio_filename, config, True)

    def update_config(self, config):
        self.config = config
        self.model.decoder.clear_transforms()
        self.add_transforms(config)

    def run_features_through_model(self, audio_features):
        """
        runs the sent features through the model one block at a time
        and concatenates the result
        returns an audio signal that is the result
        """
        output = [self.run_feature_block_through_model(i) for i in audio_features]
        faded = []
        output = np.array(output).flatten()
        return output

    def run_feature_block_through_model(self, ft):
        """
        runs a single block of features through the model
        returns and audio signal which is the result
        """
        ft["f0_hz"] = np.array(ft["f0_hz"])
        ft["loudness_db"] = np.array(ft["loudness_db"])
        outputs = self.model(ft, training=False)
        audio = self.model.get_audio_from_outputs(outputs)
        return audio

    def start_realtime(self, feature_csv_filename, audio_filename, config, midi = False):
        """
        This is the key function where se select the audio features
        and send them to the model for resynthesis
        """
        sd.default.samplerate = config["sample_rate"]
        sd.default.channels = 1 # only one channel for now!
        self.config = config
        self.stop = False;

        # setup the model
        self.setup_resynthesis(config["model_dir"])
        # get the features ready
        self.audio_features, duration = self.load_and_prepare_features_for_model(feature_csv_filename, audio_filename, config)
        self.feature_ptr = 0;
        self.features_to_poll = {
            "f0_hz":[100],
            "loudness_db":[-20]
        }
        self.audio_features = [{
            "f0_hz":np.ones(self.frames)*100,
            "loudness_db":np.ones(self.frames)*-20
        }]
        self.duration = duration
        # setup the bending transforms
        # for l in self.layers:
        #     self.transforms[l].res = self.frames;
        self.add_transforms(self.config)

        audio_callback_buffer_length = self.config["audio_callback_buffer_length"]
        self.config["audio_callback_buffer_length"] = audio_callback_buffer_length

        audio_ptr = [0]
        prev_signal = [self.run_feature_block_through_model(self.audio_features[audio_ptr[0]])]
        output_signal = [self.run_feature_block_through_model(self.audio_features[audio_ptr[0]])]

        if midi and not config["midi_port"] == "":
            def receive_message(message):
                cc = message.control
                did_update = False
                for t in self.config["transforms"]:
                    if "midi" in t["units"].keys():
                        if cc == t["units"]["midi"]["cc"]:
                            t["units"]["value"] = (message.value/127)
                            print("updated units to ", t["units"]["value"])
                            did_update = True
                    if "params" in t.keys():
                        for p in t["params"]:
                            if "midi" in p.keys():
                                if cc == p["midi"]["cc"]:
                                    r = p["midi"]["max"] - p["midi"]["min"]
                                    p["value"] = ((message.value / 127) * r ) + p["midi"]["min"]
                                    print("updated", p["name"], "to ", p["value"])
                                    did_update = True
                if did_update:
                    self.update_transforms(self.config)

            #self.inport = mido.open_input('Akai MPD32 Port 1')

            self.inport = mido.open_input(config["midi_port"])
            self.inport.callback = receive_message

        #Called on AppendAudioFeaturesTask Thread to poll in audio_features
        def append_audio_features(ctr):
            self.append_audio_features(self.features_to_poll)
            sleep(1/self.frames)

        d=[0]
        #This thread is called every 1/self.frames seconds
        #Collects in the most recently updated audio features, ready for generation
        class AppendAudioFeaturesTask:
            def __init__(self):
                self._running = True
            def terminate(self):
                self._running = False
            def run(self, action):
                while self._running:
                    action(1)
        try:
            d[0].terminate()
        except:
            print("no d yet")
        d[0] = AppendAudioFeaturesTask()
        appendThread = threading.Thread(target = d[0].run, args = (append_audio_features,))
        appendThread.start()

        class GenerateAudioTask:
            def __init__(self):
                self._running = True
            def terminate(self):
                self._running = False
            def run(self, action):
                action(1)

        def generate_audio(ctr):
            audio_ptr[0] = (audio_ptr[0] + 1) % len(self.audio_features)

            input = self.audio_features[audio_ptr[0]].copy()
            #Only input features up til the point we have collected them
            input["f0_hz"] = input["f0_hz"][:self.feature_ptr]
            input["loudness_db"] = input["loudness_db"][:self.feature_ptr]
            print("generating from ", self.feature_ptr, "frames")
            prev_signal = output_signal.copy()
            #Overwrite audio_features with self.frames worth of the last received feature
            #This essentially helps with note continuation
            self.audio_features = [{
                "f0_hz":np.ones(self.frames) * input["f0_hz"][len(input["f0_hz"])-1],
                "loudness_db":np.ones(self.frames) * input["loudness_db"][len(input["loudness_db"])-1]
            }]
            self.feature_ptr = 0
            t1 = datetime.datetime.now()
            output_signal[0] = self.run_feature_block_through_model(input)
            t2 = datetime.datetime.now()
            print("done generating block", (t2 - t1).total_seconds())

        c = [0]

        def audio_callback(outdata, frames, time, status):
            if status:
                print(status)
            xfade = 1500
            chop = 500
            prev = np.reshape(prev_signal[0], (-1))
            out = np.array(np.reshape(output_signal[0], (-1)))
            #Chop off the beginning (attempting to remove pop?)
            out = out[chop:]
            fadeout = prev[-xfade:]*np.linspace(1,0,xfade)
            fadein = out[0:xfade]*np.linspace(0,1,xfade)
            #Replace beginning with xfade from prev
            out[0:xfade] = fadeout+fadein
            #Remove end (will be xfaded into next buffer)
            out = out[:-xfade]
            # make sure the out array is the same size as the outdata array
            if out.shape[0] != outdata.shape[0]:
                #print("audio_callback:: shape mistmatch out shape, outdata shape ", out.shape, outdata.shape)
                out = np.concatenate((out, np.zeros(outdata.shape[0] - out.shape[0])))   
            outdata[:] = np.reshape(out,(out.shape[0],1))
            try:
                c[0].terminate()
            except:
                print("no c yet")
            c[0] = GenerateAudioTask()
            t = threading.Thread(target = c[0].run, args = (generate_audio,))
            t.start()

        with sd.OutputStream(channels=1, samplerate=self.config["sample_rate"], blocksize=audio_callback_buffer_length, callback=audio_callback):
            while not self.stop:
                sd.sleep(int(1 * 1000))

    def resynthesize(self, feature_csv_filename, audio_filename, config):
        """
        top level function that does resynthesis
        it prepares the basic models, reads and prepares the features
        adds transformations then calls
        assumes that the features have already been extracted from
        the input file (audio_filename)
        """
        # setup the model
        self.setup_resynthesis(config["model_dir"])
        # get the features ready
        audio_features, duration = self.load_and_prepare_features_for_model(feature_csv_filename, audio_filename, config)
        # setup the bending transforms
        # for l in self.layers:
        #     self.transforms[l].res = self.frames;
        self.add_transforms(config, duration)
        # resynthesize
        output = self.run_features_through_model(audio_features)
        print("DONE")
        return output
