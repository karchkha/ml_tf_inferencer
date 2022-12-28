# ml_tf_inferencer

"ml_tf_inferencer" is a MAX/MSP extension object that can load and run TensorFlow/Keras neural network by running the Python server. The python server will run Tensorflow library that is supporting GPU use. Thus, inference will be acquisitively fast use the GPU if available (and all drivers properly installed) on the machine. The object takes the network's name, input matrix sizes and input data as arguments. For the object to function you need to have Python installed on your machine, with some libraries as: Tensorflow, pythonosc, numpy.


## Requirements

* WIndows 10
* Max 8
* Python 3.7 or later 
	* argparse
	* numpy
	* pythonosc
	* tensorflow

## Installation

For the instalation you treat this file as any other MAX package. Just place folder "ml_tf_inferencer" with all its subfolders into directory: C:\Users\username\Documents\Max 8\Packages\
After this ml_tf_inferencer object must be seen form any MAX patch. Please see "ml_tf_inferencer" help patch to see it in action.

#### Python and its libraries

Please make sure you have python installed on your computer. If not please downoad and install latest version of python from: https://www.python.org/downloads/

You will need few additional libriries in Python. Please, install folloving libraries by executing following commands in your command prompt:

pip install argparse
pip install numpy
pip install pip install python-osc
pip install tensorflow  / pip install tensorflow-gpu

#### GPU support

Please first verify if you have a CUDA-capable GPU. Go to Windows Device Manager, where you will find the vendor name and model of your graphics card(s). If you have an NVIDIA card that is listed in http://developer.nvidia.com/cuda-gpus, that GPU is CUDA-capable. The Release Notes for the CUDA Toolkit (that you need to isntall) also contain a list of supported products.

If your GPU is cuda capable this means you can utilise GPU power while inferencing the Tensorflow model, which makes inference process faster. You will need to isntall CUDA core drivers and few more software on your machine.
The following NVIDIA® software must be installed on your system:

NVIDIA® GPU drivers —CUDA® 11.2 requires 450.80.02 or higher.
CUDA® Toolkit —TensorFlow supports CUDA® 11.2 (TensorFlow >= 2.5.0)
CUPTI ships with the CUDA® Toolkit.
cuDNN SDK 8.1.0 cuDNN versions).

Please see following links for the complete guid to CUDA drivers instalation and making GPU work on your computer:

https://www.tensorflow.org/install/gpu#windows_setup
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/
https://towardsdatascience.com/how-to-finally-install-tensorflow-gpu-on-windows-10-63527910f255

Not all cuda divers and cuDNN-s are compatible. For more details see:

https://www.tensorflow.org/install/source_windows

Here is step by step video guide for CUDA installation:

https://www.youtube.com/watch?v=hHWkvEcDBO0&t=379s&ab_channel=AladdinPersson


## Getting Started

The “ml_tf-inferencer” help file demonstrates the use case for the object. In this particular case, it is used for trigger word detection. The network is trained to detect the word “start” in the audio by looking at FFT data using a recurrent neural network (RNN). The audio is 10 seconds long, and FFT analysis is done with the following parameters: FFT size 256, sample rate 44100, overlap between windows 128. We use only the frequency response side of FFT, resulting in a data matrix from audio file with dimensions (3444, 128). This becomes data input for the network. At the same time, the output is 858 predictions corresponding to ~12 ms of audio. The network will output 1-s in response to the trigger word “start” in audio. Otherwise, the network will output zeros. In the help file, there is given 10-sec long buffer containing audio file with words “start”. It is transformed into (3444, 128) jitter matrix and sent to the neural network that is run through “ml_tf-inferencer” object.

If everything is installed correctly in response to “server 1” message “ml_tf-inferencer” must:

* Start command prompt window running python and displaying the server hosting IP address and the port number it is listening to. 
*It should display the CPU and GPU availability of the hosting machine. 
*The server should display input_size given as an argument to the “ml_tf-inferencer” object. it must be (1, 3444, 128) in this particular case.
* The “ml_tf-inferencer” must load network “Model_for_MAX5.h5” given as arguments. The network is saved in the same folder as the help patch. The server must display a network’s summary.

If all above happened without problem, the server is ready to receive data for prediction. By clicking on the bang button, you can send data for prediction. The “jit.iter” object spills data and sends it to the “ml_tf-inferencer” object. We prepend message “data” that lets the “ml_tf-inferencer” know that this is input data and the object saves this data into inner buffer. The data flow ends with the message “end” that triggers the process of sending data to the server. This is followed by the prediction on the server and by receiving and spilling out prediction from “ml_tf-inferencer.”

If the process goes well, you must see >>> sign on the server, which means the prediction went successfully and “ml_tf-inferencer” must output data. The data is imported into the visualized buffer beneath the original buffer. You should see 1-s following the trigger word “start in the original file. You can listen to the original file from the left upper corner of the help patch to make sure the prediction happened correctly. You can also run trigger word detection live by starting sound recording next to the player and triggering prediction manually or automatically with “qmetro” object.

Additionally, you can try out “gpu_change,” “verbose,” and “clean” messages. Also, you can try giving different filenames and attributes to the “ml_tf-inferencer” object. A detailed explanation of every parameter and functionally is provided in the reference window.

Please go ahead copy and adapt “ml_tf-inferencer” help patch to your own machine learning inference project!

Good luck!!

### Credits

Tornike karchkhadze 
UC San Diego
03.17.2022






