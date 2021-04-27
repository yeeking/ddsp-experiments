#%tensorflow_version 2.x
#!pip install -qU ddsp[data_preparation]

# Initialize global path for using google drive. 
DRIVE_DIR = ''
import os
import ddsp
import ddsp.training
import gin
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavutils
import tensorflow as tf
import tensorflow_probability as tfp
import time
import datetime
import json
import librosa
#from ddsp.colab.colab_utils import play, specplot
#!git clone https://github.com/Louismac/network-bending/


class UnitProvider():
  def __init__(self):
    self.unit_list = []
    self.units = 1;

  def get_units(self, s):
    if len(self.unit_list) == 0:
        self.unit_list = np.arange(s)
        np.random.shuffle(self.unit_list)
        self.unit_list = self.unit_list[:int(s * self.units)]
    #print(len(self.unit_list))
    return self.unit_list

class BendingParam():
    def __init__(self):
        self.t = 0
        #number of vals in block
        self.res = 1000
        self.unit_list = []
        self.lfo = False
        self.ramp = False
        self.scalar = 0
        self.min = 0
        self.max = 1
        self.freq = 1
        self.len = 1
    
    #return 1 block of params
    def get_values(self):
        vals = []
        if self.lfo:
          r = (self.max - self.min) / 2
          vals = np.array([self.step_lfo() for i in range(self.res)])
          vals = vals + (1 + self.min)
          vals = vals * r
        elif self.ramp:
          vals = np.linspace(self.min, self.max, self.len * self.res)[self.t:self.t+self.res]
          self.t = self.t + self.res
        else:
          vals = np.ones(self.res) * self.scalar
        return vals
 
    def step_lfo(self):
        increment = (self.freq / self.res) * (np.pi * 2)
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
        src[:,units][src[:,units] < thresh] = 0
        src[:,units][src[:,units] >= thresh] = 1
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

    def reflect(self, src, r, units):
        alpha = r
        a = np.array([[np.cos(2*alpha), np.sin(2*alpha)],
                      [np.sin(2*alpha), -np.cos(2*alpha)]])
        return self.linear_transformation(src, a)
    
    def rotate(self, src, radians, units):
        alpha = radians
        a = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return self.linear_transformation(src, a)
    
    def linear_transformation(self, src, a):
        src = src.numpy()
        src = src.reshape((src.shape[1], src.shape[2]))
        M, N = src.shape
        points = np.mgrid[0:N, 0:M].reshape((2, M*N))
        new_points = np.linalg.inv(a).dot(points).round().astype(int)
        x, y = new_points.reshape((2, M, N), order='F')
        indices = x + N*y
        wrap = np.take(src, indices, mode='wrap').reshape((1, M, N))
        t = tf.constant(wrap)
        return t

class BendingDecoder(ddsp.training.decoders.RnnFcDecoder):
    def __init__(self):
        super().__init__()
        print("bending init called")

    def init_params(self):
        print("init_params")
        self.t = {}
        self.t["FC1"] = []
        self.t["FC2"] = []
        self.t["GRU"] = []
        
    def add_transform(self, layer, f, a):
        self.t[layer].append(tf.keras.layers.Lambda(f, arguments = a))
        
    def decode(self, conditioning):
        # Initial processing.
        inputs = [conditioning[k] for k in self.input_keys]
        #print(conditioning["f0_hz"].shape)
        inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]
        # Run an RNN over the latents.
        x = tf.concat(inputs, axis=-1)
        #print(x.shape)
        for f in self.t["FC1"]:
            x = f(x)
        x = self.rnn(x)
        #print(x.shape)
        for f in self.t["GRU"]:
            x = f(x)
        x = tf.concat(inputs + [x], axis=-1)
        #print(x.shape)
        # Final processing.
        x = self.out_stack(x)
        #print(x.shape)
        for f in self.t["FC2"]:
            x = f(x)
        return self.dense_out(x)

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
        initialisesm the resynthesis models
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
        print(len(signal))
        return np.array(signal), sr
        
    def extract_features_and_write_to_file(self, audio_filename):
        """
        looks for a file called audio_filename.csv
        if it does not exist, extracts features
        using the function ddsp.training.metrics.compute_audio_features
        and writes them to that file
        returns the csv filename
        """
        audio_sig, sr = self.load_audio_data(audio_filename)
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
        self.buf_length = config["input_buf_length"]
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
    
    @staticmethod
    def check_config(config):
        """
        verify the sent config has the correct fields
        uses assert so it will end execution if anything is missing
        """
        want_keys = ["features", "input_buf_length", "frames", "db_boost", "model_dir", "frames"]
        for key in want_keys:
            assert key in config.keys(), "missing config key "+key
            print("check_config::Config has key", key)
        print("check_config::Config looks good")


    def add_transforms(self, config, duration):
        """
        adds the network bending transforms to the network
        as specified by config
        """
        for l in self.layers:
        #if transforms given for layer l
            if l in config.keys():
                c = config[l]
                for f in c:
                    arg = {}
                    units = 1;
                    if "units" in f.keys():
                        units = f["units"]
                    arg["units"] = UnitProvider()
                    arg["units"].units = units
                    if "params" in f.keys():
                        for p in f["params"]:
                            arg[p["name"]] = BendingParam()
                            arg[p["name"]].res = self.frames
                            arg[p["name"]].len = int(np.ceil(duration))
                            if "args" in p.keys():
                                for k,v in p["args"].items():
                                    setattr(arg[p["name"]], k, v)
                        self.model.decoder.add_transform(l, getattr(self.transforms[l], f["function"]), arg)


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
        print("getting next block")
        outputs = self.model(ft, training=False)
        audio = self.model.get_audio_from_outputs(outputs)
        return audio


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
        for l in self.layers:
            self.transforms[l].res = self.frames;
        self.add_transforms(config, duration)
        # resynthesize
        output = self.run_features_through_model(audio_features)
        print("DONE")
        return output

def main(input_file, output_file, model_dir, samplerate = 16000):
    """
    resynthesize the pitches and amplitudes in input_file using the model in
    model_dir and write the result to output_file
    """
    ##See Instructions (https://github.com/Louismac/network-bending/blob/main/README.md)
    config = {}
    config["model_dir"] = model_dir
    #pick how much of input file to do (0->1)
    config["features"] = {"file_name":input_file, "start":0, "end":1}
    #add boost to loudness feature of input
    config["db_boost"] = 10
    #4 secs at 16000
    config["input_buf_length"] = 4 * samplerate
    config["frames"] = 1000
    #transforms for first layer
    config["FC1"] = [
    {
        "function":"oscillate",
        "units":0.0,
        "params":[
            {"name":"depth",
            "args":{
                "lfo":True,
                "freq":1,
                "min":0.1,
                "max":0.4,
                }
            },
            {"name":"freq",
            "args":{
                "lfo":True,
                "freq":0.5,
                "min":3,
                "max":5,
                }
            }
        ]
    }
    ]
    config["FC1"] = [
        {}
    ]
    g = Generator()
    g.check_config(config)
    
    # step 1: write features to CSV file
    feature_csvfile = g.extract_features_and_write_to_file(input_file)

    np.set_printoptions(threshold=np.inf)
    # step 2: do the resynthesis
    audio_gen = g.resynthesize(feature_csvfile, input_file, config)
    wavutils.write(output_file, samplerate, np.array(audio_gen))
    return

#main('../audio_data/test_input.wav', '../audio_data/test_output.wav', '../Models/Flute2021New/')
main('../audio_data/test_input.wav', '../audio_data/test_output.wav', '../../models/Keeley16kV1/ddsp-solo-instrument/')
#main('../audio_data/test_input.wav', '../audio_data/test_output.wav', '../../models/Whitney16kV4/ddsp-solo-instrument/')



