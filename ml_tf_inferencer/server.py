"""
If executing this script returns an 'Address already in use' error
make sure there are no processes running on the ports already.
To do that run 'sudo lsof -i:9997' 'sudo lsof -i:9998'
(9997 and 9998 are the default ports used here, so adjust accordingly
if using different ports) This commands brings up list of processes using these ports,
and gives their PID. For each process type, 'kill XXXX' where XXXX is PID.
"""


import argparse
import numpy as np
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client

#import multiprocessing

import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib
from tensorflow.python.ops.lookup_ops import _as_string
from tensorflow.python.util.nest import flatten
import time



x=[]
mat=[]
input_size = []
input_size_array=np.zeros(1)
client_port=[]
model = None
#model=tf.keras.Model()


def gpu_switch(*args):
    if args[1]< 0:
        devices=device_lib.list_local_devices()

        print("\n\nThere are", len(devices), "devices on this computer:")
        for i in range(len(devices)):
          if devices[i].name[8] == "C":
            print("{:d}.".format  (i+1), devices[i].name, "with {:.2f}".format(devices[i].memory_limit/10000000), "GB memory limit(",devices[i].physical_device_desc,")")
          if devices[i].name[8] == "G":
            print("{:d}.".format  (i+1), devices[i].name, "with {:.2f}".format(devices[i].memory_limit/1000000000), "GB memory limit(",devices[i].physical_device_desc,")")
        print("You can decide which to use by adjasting 'gpu_switch' value")
        
    elif args[1]== 0:
        #try:
        #    # Disable all GPUS
        #    tf.config.set_visible_devices([], 'GPU')
        #    visible_devices = tf.config.get_visible_devices()
        #    for device in visible_devices:
        #        assert device.device_type != 'GPU'
        #except:
        #    # Invalid device or cannot modify virtual devices once initialized.
        #    pass
        #print(tf.config.list_physical_devices())
        
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # here we stop GPU from working with tensorflow #

        devices=device_lib.list_local_devices()

        print("\n\nThere are", len(devices), "devices on this computer:")
        for i in range(len(devices)):
          if devices[i].name[8] == "C":
            print("{:d}.".format  (i+1), devices[i].name, "with {:.2f}".format(devices[i].memory_limit/10000000), "GB memory limit(",devices[i].physical_device_desc,")")
          if devices[i].name[8] == "G":
            print("{:d}.".format  (i+1), devices[i].name, "with {:.2f}".format(devices[i].memory_limit/1000000000), "GB memory limit(",devices[i].physical_device_desc,")")
        
        
    elif args[1]> 0:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
          # Restrict TensorFlow to only allocate part of memory on the first GPU
          try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=args[1])])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
          except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
        
        devices=device_lib.list_local_devices()
        
        print("\n\nThere are", len(devices), "devices on this computer:")
        for i in range(len(devices)):
          if devices[i].name[8] == "C":
            print("{:d}.".format  (i+1), devices[i].name, "with {:.2f}".format(devices[i].memory_limit/10000000), "GB memory limit(",devices[i].physical_device_desc,")")
          if devices[i].name[8] == "G":
            print("{:d}.".format  (i+1), devices[i].name, "with {:.2f}".format(devices[i].memory_limit/1000000000), "GB memory limit(",devices[i].physical_device_desc,")")
    #client.send_message("/gpu_status_ready", True)     



def data_append(*args):
    x.append(args[1:])
    client.send_message("/appended", True)


def input_size_append(*args):
    global input_size
    global input_size_array
    x.clear()
    input_size.clear()
    input_size.append(args[1:])
    input_size_array=np.array(input_size)
    print("\nInput size received from Max:", *input_size, "\n")


def predict(*args):
    if input_size_array.any()>0:  ###check if input size is given
        input_array=np.array(x)
        print("shape of appended array ", input_array.shape, "size given", input_size_array[0])

        if input_array.size==np.prod(input_size_array[0], axis=0): #### check if input size and coming data are inline
            reshaped_array = np.reshape(input_array,input_size_array[0])
            print("shape of input:",reshaped_array.shape)
            try:                                                   #### try prediction 
                prediction=model.predict(reshaped_array)
                flatten_prediction=prediction.flatten()
                for i in range(0,len(flatten_prediction),256):
                    prediction_for_sending=str(flatten_prediction[i:min(i+256,len(flatten_prediction))])
                    client.send_message("/prediction",prediction_for_sending[1:-1])
                    time.sleep(0.0000001)

        
                print (">>>" )
            
            except:
                if model != None:
                    print("\nCan't predict! Please, load correct h5 model to the server first!\n")
                else:
                    print("\nCan't predict! Please, load h5 model to the server first!\n")
        else:
            print("Given input data size and real input data size does not match! Check your settings!")
    else:
        print("Input matrix size is not given! Check your settings!")
    x.clear()
    input_array=np.empty(0)
    

def load_network(*args):
    global model
    model = load_model(args[1:][0])
    print("\n", args[1:][0], "Model loaded! Here is its summary:")
    print(model.summary())





def clear_x(*args):
    global x
    x=[]











######################### SERVER #############################

if __name__ == "__main__":
    import warnings
    #multiprocessing.freeze_support()   
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip",
        default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("--serverport",
        type=int, default=9997, help="The port for server listen on")
    parser.add_argument("--clientport",
        type=int, default=9996, help="The client port")
 
    args = parser.parse_args()
    
    

    dispatcher = dispatcher.Dispatcher()


    dispatcher.map("/data", data_append)
    dispatcher.map("/input_size", input_size_append)
    dispatcher.map("/predict", predict)
    dispatcher.map("/filename", load_network)
    dispatcher.map("/gpu_switch", gpu_switch)

    dispatcher.map("/clear", clear_x)

    

    ### server
    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.serverport), dispatcher)

    ### client
    client_port=str(args.clientport)
    client = udp_client.SimpleUDPClient(args.ip, args.clientport)


    client.send_message("/ready", True) 


    print("Serving on {}".format(server.server_address))
    #print("Ctrl+C to quit")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        while True:
            pass
        #except KeyboardInterrupt:
        #    pass
        #sys.exit()








