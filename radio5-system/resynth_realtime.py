import sounddevice as sd # https://python-sounddevice.readthedocs.io/en/0.3.15/

import warnings
warnings.filterwarnings("ignore")

import copy
import os
import os.path
import time
import datetime
import crepe
import ddsp
import ddsp.training
import gin
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds
import scipy.io.wavfile as wavutils

### Top level stuff that you might want to edit

def setup_tensorflow():
    config = tf.compat.v1.ConfigProto(gpu_options = 
                                    tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
                    # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    print("setup_tensorflow::GPU configured.")
    return

# set up a ddsp model with the sent name
# assumes that model is available in ../Models
# input_signal_buf_length is the number of samples
# in the input buffer that you are planning to feed to the model 
# returns the model and some info about the model. 
# The info is needed to correctly trim the audio features fed into the model
# @return model
# @return time_steps : for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
#        audio_features[key] = audio_features[key][:time_steps]
# @return required_input_samples:     audio_features['audio'] = audio_features['audio'][:, :required_input_samples]

def setup_model(model_name, input_signal_buf_length):

    PRETRAINED_DIR = '../models/'+model_name    
    model_dir = PRETRAINED_DIR
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
    #F0LoudnessPreprocessor
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    #n_samples_train = gin.query_parameter('Additive.n_samples')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(input_signal_buf_length / hop_size)
    required_input_samples = time_steps * hop_size
    gin_params = [
        'RnnFcDecoder.input_keys = ("f0_scaled", "ld_scaled")',
        #'Additive.n_samples = {}'.format(required_input_samples),
        'Harmonic.n_samples = {}'.format(required_input_samples),
        'FilteredNoise.n_samples = {}'.format(required_input_samples),
     #   'DefaultPreprocessor.time_steps = {}'.format(time_steps),
        'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps)
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Set up the model just to predict audio given new conditioning
    model = ddsp.training.models.Autoencoder()
    model.restore(ckpt)
    print("setup_model: ", model)
    return model, time_steps, required_input_samples

# setup tensorflow, the feature extractor and the model
def setup_resynthesis(model_name, input_buffer_length):
    setup_tensorflow()
    ddsp.spectral_ops.reset_crepe()
    model, time_steps, required_input_samples = setup_model(model_name, input_buffer_length)
    print("setup_resynthesis::resynthesis ready probably")
    return model, time_steps, required_input_samples


# Assumes audio signal is a 1D np array
# with samples in
def extract_audio_features(audio_signal, model_time_steps, model_buffer_length):
    # Setup the session.
    # Compute features.
 #   start_time = time.time()

    audio_signal = np.array([audio_signal])
    #print('extract_audio_features:: audio signal shape ', audio_signal.shape)
    audio_features = ddsp.training.eval_util.compute_audio_features(audio_signal)
    audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
    #audio_features_mod = None

    # Trim all input vectors to correct lengths 
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:model_time_steps]
    #print("extract_audio_features:: audio len: ", audio_features['audio'].shape)
    #print("extract_audio_feature:: should be: ", model_buffer_length)
    
#    audio_features['audio'] = audio_features['audio'][:, :model_buffer_length]
    return audio_features


def fix_features(audio_features, model_name):
    return audio_features

def resynthesize(model, model_name, input_signal, model_time_steps, model_buffer_length, pitch_transpose = 0):
    start_time = time.time()

    audio_features = extract_audio_features(input_signal, model_time_steps, model_buffer_length)
    # here we massage the features in various ways
    audio_features = fix_features(audio_features, model_name)

    if pitch_transpose > 0:
        f0_scalar = 2 # try the octave first
        audio_features['f0_hz'] = audio_features['f0_hz'] * f0_scalar

    output_signal = model(audio_features, training=False)
    # convert back to a simple 1D array
    print('resynthesis ::  samples  took %.1f seconds' %  (time.time() - start_time))

    return output_signal[0]


def get_normalise_scalar(buffer):
    max = 0
    for i in range(len(buffer)):
        if np.abs(buffer[i]) > max:
            max = np.abs(buffer[i])
    scalar = 1/max
    return scalar

def main(config):
  
    sd.default.samplerate = config["sample_rate"]
    sd.default.channels = 1 # only one channel for noe!

    model, model_time_steps, model_buffer_length = setup_resynthesis(
        config["model_name"], config["callback_buffer_length"])

    print("config buffer len ", config["callback_buffer_length"], "model buffer len ", model_buffer_length)
    if model_buffer_length != config["callback_buffer_length"]:
        config["callback_buffer_length"] = model_buffer_length
    
    # set up the recording arrays
    recording_pos = [0] # not sure why but this needs to be a list not a simple var
    # a buffer to store input and output
    complete_output = np.zeros((2, config["duration"] * config["sample_rate"]))
    total_frames = (config["duration"] * config["sample_rate"]) / config["callback_buffer_length"]
    print("main:: initialised output array to ", complete_output.shape)
    
    def audio_callback(indata, outdata, frames, time, status):
        print("audio_callback:: output_ind ", round(recording_pos[0] / config["callback_buffer_length"]), " of ", round(total_frames))
        
        if status:
            print(status)
        #print('callback:: make a block', time)
        #outdata[:] = indata
        input_signal = np.rot90(indata)[0] # [[s1], [s2], [s3]] to [s1,s2,s3]
        output_signal = resynthesize(model, config["model_name"], input_signal, model_time_steps, model_buffer_length, 12)  
        outdata [:] = np.reshape(output_signal, (-1, 1)) # [s1,s2,s3] to [[s1], [s2], [s3]] 
        #print(indata)
        # copy input_signal and output_signal
        # into the complete_output array
        start_pos = recording_pos[0]
        #print("audio_callback:: copy range ", start_pos, " to ", start_pos + len(output_signal), " available ", len(complete_output[0]))

        complete_output[0][start_pos:start_pos+len(output_signal)] = input_signal
        complete_output[1][start_pos:start_pos+len(output_signal)] = output_signal
        recording_pos[0] = recording_pos[0] + len(output_signal)
        #something_else = something_else + len(output_signal)

    print("main:: starting the callback with buf_length", config["callback_buffer_length"], "then waiting for ", config["duration"], "seconds ...")
    with sd.Stream(channels=1, samplerate=config["sample_rate"], blocksize=config["callback_buffer_length"], callback=audio_callback):
        sd.sleep(int(config["duration"] * 1000))

    print("main:: synthesis ends...")
    now = datetime.datetime.now()
    output_file =  now.strftime("%Y%m%d%H%M%S") + "_live_using_" + config["model_name"] + ".wav"
    
    boost_left = get_normalise_scalar(complete_output[0])
    boost_right = get_normalise_scalar(complete_output[1])
    complete_output[0] = complete_output[0] * boost_left
    complete_output[1] = complete_output[1] * boost_right

    amplitude = np.iinfo(np.int16).max
    complete_output = complete_output * amplitude
    # now rotate it from [[ch1...], [ch2...]]
    # to [[c1, c2], [c1, c2] ..]
    complete_output = np.rot90(complete_output, 3) # 3 as 1 is reversed
    #complete_output = np.array([int((x + 1) * 32768) for x in complete_output])
    #wavutils.write('../audio/incoming/'+output_file, config["sample_rate"], complete_output.astype(np.int16))

    #print("main:: wrote result to ", output_file)



# config = {"model_name":'Lauda_novella', 
#             "sample_rate":16000, 
#             "callback_buffer_length":16000*10, 
#             "duration": 60}

config = {"model_name":'Flute', 
            "sample_rate":16000, 
            "callback_buffer_length":16000*5, 
            "duration": 60}

# config = {"model_name":'Tenor_Saxophone', 
#             "sample_rate":16000, 
#             "callback_buffer_length":16000*5, 
#             "duration": 30}
main(config)
