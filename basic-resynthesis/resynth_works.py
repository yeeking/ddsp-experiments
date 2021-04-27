# Derived from the Gooogle Magenta DDSP example
# Copyright 2020 Google LLC. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#def imports():
# Ignore a bunch of deprecation warnings
import warnings
warnings.filterwarnings("ignore")

import copy
import os
import time

import crepe
import ddsp
import ddsp.training
#   from ddsp.colab.colab_utils import (download, play, record, specplot, upload,
#                                       DEFAULT_SAMPLE_RATE)
import gin
#from google.colab import files
import librosa
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow.compat.v2 as tf
import tensorflow as tf
#import tensorflow_datasets as tfds

import scipy.io.wavfile as wavutils


# Helper Functions
#sample_rate = DEFAULT_SAMPLE_RATE  # 16000


import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                                 tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
                    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

ddsp.spectral_ops.reset_crepe()

print('resynth_works:: GPU configured. ')

# generates a 1D audio signal
# def get_audio_input():
#     audio_sig = np.array([[np.sin(x) for x in np.arange(0, np.pi, 0.001)]], dtype=np.float32)
#     return audio_sig

def get_audio_input_file(file_name, sr=16000):
    """
    Load in the file and put it in the right range -1, 1, returns mono
    """
    assert os.path.exists(file_name), "get_audio_input_file:: file not found " + file_name
    #fs,signal=wavutils.read(file_name)
    signal, sr = librosa.load(file_name, sr=sr)
    #signal = np.copy(signal[0:22050]) [:,0] # first column
    print(signal.shape)
    #signal = np.copy(signal) [:,0] # first column
    #signal = np.copy()
    #signal = np.array([signal / 32767])
    signal = np.array([signal])
    
    return signal


def extract_input_features(audio_signal, sr=16000):
    # Setup the session.
    # Compute features.
    start_time = time.time()
    audio_features = ddsp.training.metrics.compute_audio_features(audio_signal, sample_rate=sr)
    #audio_features = ddsp.training.eval_util.compute_audio_features(audio_signal)
    audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
    audio_features_mod = None
    print('extract_input_fetures:: Audio features took %.1f seconds' % (time.time() - start_time))
    return audio_features

def plot_audio_features(audio_features):
        # Plot Features.
    fig, ax = plt.subplots(nrows=3, 
                        ncols=1, 
                        sharex=True,
                        figsize=(6, 8))
    ax[0].plot(audio_features['loudness_db'])
    ax[0].set_ylabel('loudness_db')

    ax[1].plot(librosa.hz_to_midi(audio_features['f0_hz']))
    ax[1].set_ylabel('f0 [midi]')

    ax[2].plot(audio_features['f0_confidence'])
    ax[2].set_ylabel('f0 confidence')
    _ = ax[2].set_xlabel('Time step [frame]')
    plt.show()
    return plt
#imports()

def find_model_dir(dir_name):
  # Iterate through directories until model directory is found
  for root, dirs, filenames in os.walk(dir_name):
    for filename in filenames:
      if filename.endswith(".gin") and not filename.startswith("."):
        model_dir = root
        break
  return model_dir 

def prep_model_simple(audio_signal, audio_features, model_name = 'Sax', models_dir = '../models/'):
    model_dir = models_dir + model_name + "/ddsp-solo-instrument/" 
    assert os.path.exists(model_dir), "prep_model_simple:: model not found " + model_dir
    gin_file = os.path.join(model_dir, 'operative_config-0.gin')
    assert os.path.exists(gin_file), "prep_model_simple:: gin config not found " + gin_file
  
    # Parse gin config,
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
    ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = os.path.join(model_dir, ckpt_name)

    # Ensure dimensions and sampling rates are equal
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio_signal.shape[1] / hop_size)
    n_samples = time_steps * hop_size
    gin_params = [
        'RnnFcDecoder.input_keys = ("f0_scaled", "ld_scaled")',
        'Harmonic.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Trim all input vectors to correct lengths 
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]

    print("prep_model_simple:: resizing audio feature audio ", )
    print("prep_model_simple:: input audio shape ", audio_sig.shape)
    
    print("prep_model_simple:: old audio shape ", audio_features['audio'].shape)
    print("prep_model_simple:: n_samples, time_steps, hopsize", n_samples, time_steps, hop_size)
    
    audio_features['audio'] = audio_features['audio'][:, :n_samples]
    
    print("prep_model_simple:: new audio shape ", audio_features['audio'].shape)
    print("prep_model_simple:: About to create model")
    # Set up the model just to predict audio given new conditioning
    model = ddsp.training.models.Autoencoder()
    print("prep_model_simple:: About to restore model")
    model.restore(ckpt)
    print("prep_model_simple::Running a batch through the model")
    
    # Build model by running a batch through it.
    #start_time = time.time()
    #_ = model(audio_features, training=False)
    return model

## Helper functions.
def shift_ld(audio_features, ld_shift=0.0):
  """Shift loudness by a number of ocatves."""
  audio_features['loudness_db'] += ld_shift
  return audio_features


def shift_f0(audio_features, f0_octave_shift=0.0):
  """Shift f0 by a number of ocatves."""
  audio_features['f0_hz'] *= 2.0 ** (f0_octave_shift)
  audio_features['f0_hz'] = np.clip(audio_features['f0_hz'], 
                                    0.0, 
                                    librosa.midi_to_hz(110.0))
  return audio_features


def mask_by_confidence(audio_features, confidence_level=0.1):
  """For the violin model, the masking causes fast dips in loudness. 
  This quick transient is interpreted by the model as the "plunk" sound.
  """
  mask_idx = audio_features['f0_confidence'] < confidence_level
  audio_features['f0_hz'][mask_idx] = 0.0
  # audio_features['loudness_db'][mask_idx] = -ddsp.spectral_ops.LD_RANGE
  return audio_features


def smooth_loudness(audio_features, filter_size=3):
  """Smooth loudness with a box filter."""
  smoothing_filter = np.ones([filter_size]) / float(filter_size)
  audio_features['loudness_db'] = np.convolve(audio_features['loudness_db'], 
                                           smoothing_filter, 
                                           mode='same')
  return audio_features




def adjust_input(audio_features,  model):
  #  if auto_adjust:
    if model in ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Saxophone', 'Tenor_Saxophone', 'AGrande16']:
        # Adjust the peak loudness.
        l = audio_features['loudness_db']
        model_ld_avg_max = {
            'Violin': -34.0,
            'Flute': -45.0,
            'Flute2': -44.0,
            'Trumpet': -52.3,
            'Tenor_Saxophone': -31.2, 
            'AGrande16': -45,

        }[model]
        ld_max = np.max(audio_features['loudness_db'])
        ld_diff_max = model_ld_avg_max - ld_max
        audio_features_mod = shift_ld(audio_features_mod, ld_diff_max)

        # Further adjust the average loudness above a threshold.
        l = audio_features_mod['loudness_db']
        model_ld_mean = {
            'Violin': -44.0,
            'Flute': -51.0,
            'Flute2': -53.0,
            'Trumpet': -69.2,
            'Tenor_Saxophone': -50.8, 
            'AGrande16': -51
        }[model]
        ld_thresh = -70.0
        ld_mean = np.mean(l[l > ld_thresh])
        ld_diff_mean = model_ld_mean - ld_mean
        audio_features_mod = shift_ld(audio_features_mod, ld_diff_mean)

        # Shift the pitch register.
        model_p_mean = {
            'Violin': 73.0,
            'Flute': 81.0,
            'Flute2': 74.0,
            'Trumpet': 65.8,
            'Tenor_Saxophone': 57.8, 
            'AGrande16': 60, 
        }[model]
        p = librosa.hz_to_midi(audio_features['f0_hz'])
        p[p == -np.inf] = 0.0
        p_mean = p[l > ld_thresh].mean()
        p_diff = model_p_mean - p_mean
        p_diff_octave = p_diff / 12.0
        round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
        p_diff_octave = round_fn(p_diff_octave)
        audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)

    f0_octave_shift =  0 #@param {type:"slider", min:-2, max:2, step:1}
    f0_confidence_threshold =  0 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
    loudness_db_shift = 0 #@param {type:"slider", min:-20, max:20, step:1}

    # audio_features_mod = {k: v.copy() for k, v in audio_features.items()}
 
    # audio_features_mod = shift_ld(audio_features_mod, loudness_db_shift)
    # audio_features_mod = shift_f0(audio_features_mod, f0_octave_shift)
    # audio_features_mod = max  sk_by_confidence(audio_features_mod, f0_confidence_threshold)


    return audio_features

# all models are at 16000
sample_rate = 16000

input_file = "./keeley_cut1.wav"
model_name = "Flute2021New"
model_name = "Keeley16kV1"

audio_sig = get_audio_input_file(input_file)
#audio_sig,sr = librosa.load(input_file, sr=sample_rate)
#audio_sig = [audio_sig]
# get audio features
print('resynth_works:: Extracting features. Sig length ', len(audio_sig[0]))
audio_features = extract_input_features(audio_sig)

# get the model and pre-process features
model = prep_model_simple(audio_sig, audio_features, model_name)
print('resynth_works:: Model might be ready... adjusting input features')


#model = prep_model(audio_sig, audio_features, 'Flute')
audio_features = adjust_input(audio_features, model)
print('resynth_works:: Audio features adjusted. resynthesize now??')
outputs = model(audio_features, training=False)
audio_gen = model.get_audio_from_outputs(outputs)

print('resynth_works:: generated audio')
print(audio_gen)
output_file = input_file + "_using_" + model_name + ".wav"
print('resynth_works:: writing output file', output_file)

wavutils.write(output_file, sample_rate, np.array(audio_gen[0]))



