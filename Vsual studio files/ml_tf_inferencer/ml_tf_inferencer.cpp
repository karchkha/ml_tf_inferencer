/**
	@file
	ml_tf_inferencer
	
	ml_tf_inferencer is a MAX/MSP extension object that can load and run TensorFlow/Keras neural network by running the Python server. 
	The the python server will run Keras and Tensorflow libraries and thus it will be acquisitively fast and will use the GPU if available. 
	For the object to function, you need Python and Tensorflow installed on your machine. Ecxact list of nessesary librieries will be provided
	with the object in README file. ml_tf_inferencer takes the network's name, input matrix sizes and input data as arguments. 
	The object can load a neural network with message "read" saved in HDF5 format. The message with the message "data" followed by a list 
	of numbers and ending with mnessage "end" will trigger the prediction process. In response, the object will output a list of numbers 
	with indexies of the corresponding size as a prediction from the network. When deleted from the patch, the object must deactivate 
	the python server and free all the memory taken.

	Tornike Karchkhadze, tkarchkhadze@ucsd.edu
*/



#include <stdio.h>
#include "ext.h"
#include "ext_obex.h"

#include <time.h>


#include "shellapi.h"
#include <windows.h>


#include <iostream>
#include "osc/OscOutboundPacketStream.h"
#include "ip/UdpSocket.h"
#include "osc/OscReceivedElements.h"
#include "osc/OscPacketListener.h"

#include <mutex>
//#include <shared_mutex>
//#include <condition_variable>
//#include <stdatomic.h>
//#include <thread>
//#include <functional>

#include<stdio.h>		/* for dumpping memroy to file */

//#include <map>
//#include <string>
//#include <string_view>

//using namespace std::placeholders;


#define ADDRESS "127.0.0.1" /* local host ip */
#define OUTPUT_BUFFER_SIZE 8192 /* maximum buffer size for osc send */

/* This following class is for controling threading and data flow to and from the python server */
/* we have this class because this is much more convinient to call its instances and call its functions this way */

class thread_control
{
	std::mutex mutex;
	std::condition_variable condVar;

public:
	thread_control()
	{}
	void notify() /* this will be used to notify threads that server have done processing */
	{
		condVar.notify_one();
	}
	void waitforit() /* this will be used to stop threads and wait for responces from server */
	{
		std::unique_lock<std::mutex> mlock(mutex);
		condVar.wait(mlock);
	}
	void lock() {
		mutex.lock();
	}
	void unlock() {
		mutex.unlock();
	}
};



// Data Structures
typedef struct _ml_tf_inferencer {
	t_object	ob;

	float * data_buffer;	 /* input data buffer for stroring incoming data that will be sent to server */
	int input_size[10];		/* input data matrix shape, asssuming that it will not have more dimentions than 10 */
	int inner_size;			/* size of matrixes inner vector, used for buffering and cheking lenght */
	int full_size;			/* full length of the data, used for buffering and cheking lenght */
	long filling_index;		/* filling buffer index */

	int output_index;		/* indexing output data */
	float output_float;		/* output data */

	char filename[MAX_PATH_CHARS]; /* Keras Network name saved as .h5 file */

	long gpu_switch; /* variable for gpu switching on/off and/or limiting */

	int verbose_flag; /* defines if extention will or won't output execution times and other information */

	SHELLEXECUTEINFO lpExecInfo; /* Open .h5 file handler */

	/* paralel thereads for osc listening and outputing */
	t_systhread		listener_thread;
	t_systhread		output_thread;

	/* comunication ports with pythin server */
	int PORT_SENDER; 
	int PORT_LISTENER; 

	/* comunication with OSC listener */
	const char** prediction;	/* this is pointer to pointer that will extract data from OSC listener */
	bool* server_predicted;		/* flag if server has predicted */
	bool* server_ready;			/* flag if server is running and ready to receive data */
	bool* python_import_done;	/* flag if server iported chunk of data and is ready to receive next chunk */

	/* thread controls */
	thread_control* out_tread_control;		/* lock, unlock and notify routine for output thread */
	thread_control* server_control;			/* lock, unlock and notify routine for server start up thread */
	thread_control* python_import_control;	/* lock, unlock and notify routine for server's data importing thread */

	/* outlets */
	void* index_outlet;
	void* float_outlet;

	long test;
	
} t_ml_tf_inferencer;


// Prototypes
t_ml_tf_inferencer* ml_tf_inferencer_new(t_symbol* s, long argc, t_atom* argv);

void		ml_tf_inferencer_assist(t_ml_tf_inferencer* x, void* b, long m, long a, char* s);
void		ml_tf_inferencer_clear(t_ml_tf_inferencer* x);
void		ml_tf_inferencer_free(t_ml_tf_inferencer* x);
void		ml_tf_inferencer_data(t_ml_tf_inferencer* x, t_symbol* s, long argc, t_atom* argv);
void		ml_tf_inferencer_end(t_ml_tf_inferencer* x, t_symbol* s, long argc, t_atom* argv);
void		ml_tf_inferencer_input_size(t_ml_tf_inferencer* x, t_symbol* s, long argc, t_atom* argv);
void		ml_tf_inferencer_read(t_ml_tf_inferencer* x, t_symbol* s);
void		ml_tf_inferencer_doread(t_ml_tf_inferencer* x, t_symbol* s, long argc, t_atom* argv);
void		ml_tf_inferencer_server(t_ml_tf_inferencer* x, long command);
void		ml_tf_inferencer_verbose(t_ml_tf_inferencer* x, long command);

void		ml_tf_inferencer_OSC_data_sender(t_ml_tf_inferencer* x, int argc, char* argv[]);
void		ml_tf_inferencer_OSC_input_size_sender(t_ml_tf_inferencer* x);
void		ml_tf_inferencer_OSC_filename_sender(t_ml_tf_inferencer* x, char *filename);
void		ml_tf_inferencer_OSC_time_to_predict_sender(t_ml_tf_inferencer* x);
void		ml_tf_inferencer_OSC_send_gpu_switch_change(t_ml_tf_inferencer* x, long argv);
void		ml_tf_inferencer_OSC_send_gpu_switch(t_ml_tf_inferencer* x);

void		* ml_tf_inferencer_OSC_listener(t_ml_tf_inferencer* x, int argc, char* argv[]);
void		ml_tf_inferencer_OSC_listen_thread(t_ml_tf_inferencer* x);

void		ml_tf_inferencer_start_output_thread(t_ml_tf_inferencer* x);
void		ml_tf_inferencer_output_thread_stop(t_ml_tf_inferencer* x);
void		ml_tf_inferencer_output(t_ml_tf_inferencer* x);

//void		ml_tf_inferencer_file(t_ml_tf_inferencer* x);
//void		ml_tf_inferencer_post(t_ml_tf_inferencer* x);
void		timestamp();



// Globals and Statics
static t_class* s_ml_tf_inferencer_class = NULL;

///**********************************************************************/

// Class Definition and Life Cycle

void ext_main(void* r)
{
	t_class* c;

	c = class_new("ml_tf_inferencer", (method)ml_tf_inferencer_new, (method)ml_tf_inferencer_free, sizeof(t_ml_tf_inferencer), (method)NULL, A_GIMME, 0L);

	class_addmethod(c, (method)ml_tf_inferencer_assist, "assist", A_CANT, 0);
	class_addmethod(c, (method)ml_tf_inferencer_data, "data", A_GIMME, 0);
	class_addmethod(c, (method)ml_tf_inferencer_end, "end", A_GIMME, 0);
	class_addmethod(c, (method)ml_tf_inferencer_input_size, "input_size", A_GIMME, 0);
	class_addmethod(c, (method)ml_tf_inferencer_read, "read", A_DEFSYM, 0);
	class_addmethod(c, (method)ml_tf_inferencer_server, "server", A_LONG, 0);
	class_addmethod(c, (method)ml_tf_inferencer_OSC_send_gpu_switch_change, "gpu_change", A_LONG, 0);
	class_addmethod(c, (method)ml_tf_inferencer_clear, "clear", 0);
	class_addmethod(c, (method)ml_tf_inferencer_verbose, "verbose", A_LONG, 0);
	

	//class_addmethod(c, (method)ml_tf_inferencer_post, "post", 0);		/* this is for posting what's in buffer. used for debugging. this will desapiar */
	//class_addmethod(c, (method)ml_tf_inferencer_file, "save", 0);		/* this is for saving data buffer as file for debuging purposes. this will desapiar too */

	/* attributes */
	CLASS_ATTR_LONG(c, "gpu", 0, t_ml_tf_inferencer, gpu_switch);  
	CLASS_ATTR_LONG(c, "verb", 0, t_ml_tf_inferencer, verbose_flag);


	class_register(CLASS_BOX, c);
	s_ml_tf_inferencer_class = c;
}


/***********************************************************************/
/***********************************************************************/
/********************** Initialisation *********************************/
/***********************************************************************/
/***********************************************************************/

t_ml_tf_inferencer* ml_tf_inferencer_new(t_symbol* s, long argc, t_atom* argv)
{
	t_ml_tf_inferencer* x = (t_ml_tf_inferencer*)object_alloc(s_ml_tf_inferencer_class);

	if (x) {

		/* initialising variables */

		/* generate 2 random number that will beconme port numbers */
		srand(time(NULL));		 
		x->PORT_SENDER = rand() % 100000;	
		x->PORT_LISTENER = rand() % 100000;
		
		memset(x->filename, 0, MAX_PATH_CHARS); /* emptying network name variable for the begining */

		x->verbose_flag = 0;
		x->filling_index = 0;
		x->output_index = 0;
		for (int i = 0; i < 10; i++) { x->input_size[i] = 0; };
		x->inner_size = 1;
		x->full_size = 1;
		x->gpu_switch = -1; /* -1 means we don't do anything with gpu and just use whatever is available */
					
		//x->output_float = 0.0;		/* we ended up not using this */
		//x->lpExecInfo = NULL;			/* no need to be initialised */

		/* Starting up OSC listener server and output thread */
		ml_tf_inferencer_OSC_listen_thread(x);   /* this function starts OSC listener inside sub-thread */
		ml_tf_inferencer_start_output_thread(x); /* this starts up output thread that runs independently and outputs whenever server gives answer */
		
		x->out_tread_control = new thread_control;
		x->server_control = new thread_control;
		x->python_import_control = new thread_control;

		//*x->server_ready = false;

		x->data_buffer = (float *)sysmem_newptr(sizeof(float) * x->full_size);
				
		/* here we check if argument are given */

		long offset = attr_args_offset((short)argc, argv); /* this is number of arguments before attributes start with @-sign */

		if (offset && offset > 0) {

			/* first argument is filename */
			ml_tf_inferencer_read(x, atom_getsym(argv));

			if (offset > 1) { /* following numbers (up to 10 or up to offset) will be read as input shape */

				t_atom* slice = (t_atom*)sysmem_newptr(sizeof(t_atom) * (offset - 1)); /* we create new array of t-atoms*/
				memcpy(slice, argv + 1, sizeof(t_atom) * (offset - 1));  /* we copy argumnets to this new array */

				ml_tf_inferencer_input_size(x, gensym("input_size"), offset - 1, slice); /* set input size */

			}
		}

		

		/* outlets */
		x->float_outlet = outlet_new((t_ml_tf_inferencer*)x, NULL);
		x->index_outlet = outlet_new((t_ml_tf_inferencer*)x, NULL);

		attr_args_process(x, argc, argv); /* this is attribute reader */
		
	}
	return x;

}


void ml_tf_inferencer_free(t_ml_tf_inferencer* x)
{

	sysmem_freeptr(x->data_buffer);

	/************** Close pythoon server ****************/
	TerminateProcess(x->lpExecInfo.hProcess, 0);
	CloseHandle(x->lpExecInfo.hProcess);

}



/**********************************************************************/
/**********************************************************************/
/*************************** Methods ********************************/
/**********************************************************************/
/**********************************************************************/


/* this function shows info when user brings mouse to inlets and outlets */
void  ml_tf_inferencer_assist(t_ml_tf_inferencer* x, void* b, long m, long a, char* s)
{
	if (m == ASSIST_INLET) { // inlet
		sprintf(s, "data input for prediction, input_size for data size");
	}
	else {	// outlet
		switch (a) {
		case 0: sprintf(s, "Output index"); break;
		case 1: sprintf(s, "Output predictions"); break;
		}
	}
}

/************************ This function clears all the settings in object and server ******/
void ml_tf_inferencer_clear(t_ml_tf_inferencer* x) {

	x->filling_index = 0;
	x->output_index = 0;
	memset(x->filename, 0, MAX_PATH_CHARS); /* emptying array for the begining array */
	for (int i = 0; i < 10; i++) { x->input_size[i] = 0; };
	x->inner_size = 1;
	x->full_size = 1;
	x->gpu_switch = -1;
	//*x->python_import_done = false;
	x->verbose_flag = 0;
	
	if (*x->server_ready) { /* if server is running it is restarted */
		ml_tf_inferencer_server(x, 0); /* close server */
		ml_tf_inferencer_server(x, 1); /* start server over */
	}
}

/* function that sets verbose flag. hwne werbose is 1 server outputs log data */
void ml_tf_inferencer_verbose(t_ml_tf_inferencer* x, long command) {

	x->verbose_flag = command;

	if (command == 1) {

		post("Input_size:");
		for (int i = 0; i < 10; i++) { if (x->input_size[i] > 0) { post(" % d", x->input_size[i]); } }
		post("Network file: %s", x->filename);
		post("GPU status: %d", x->gpu_switch);
		post("Listening on port: %d", x->PORT_LISTENER);
		post("Sending to port: %d", x->PORT_SENDER);
		post("Server running: %d", *x->server_ready);
	}
	else if (command == 0) {}

}



/****** getting and gathering float number in buffer for prediction. number are prepanded with word "data" ********/

void ml_tf_inferencer_data(t_ml_tf_inferencer* x, t_symbol* s, long argc, t_atom* argv)
{
	if (*x->server_ready) { /* this only happens when server is running */
		int i = 0;

		if (x->filling_index == 0 && x->verbose_flag==1) {
			
			post(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>data import started"); timestamp(); 
		}

		// With each incoming agument of message "data" we fill an array.
		for (i = 0; i < argc; i++) {
			if (x->filling_index + i > x->full_size) {
				post("Filling index exceeded buffer"); //timestamp();
			}
			else {
				*(x->data_buffer + x->filling_index + i) = (float)atom_getfloat(argv);
			}
		}

		x->filling_index = x->filling_index + i; /* filling index is incremented */
	}
	else {
		
	}

}


/********* symbol "end" that comes after data  ******************/
/********* checking if buffer was filled correctly and sending data to server *********************/
void ml_tf_inferencer_end(t_ml_tf_inferencer* x, t_symbol* s, long argc, t_atom* argv)
{
	if (*x->server_ready) { /* this only happens when server is running */

		/* afret importing we need to check if buffer succsided and then we start exporting */

		if (x->verbose_flag == 1) {
			post("'End' message on filling index %d", x->filling_index); timestamp();
		}

		if (x->full_size > x->filling_index + 1) {
			post("Data did'n fill the buffer!!! Please, check your input_size and data size. Import unsuccessful!"); //timestamp();

			/*restarting fillin index as prediction failed */
			x->filling_index = 0;
		}
		else {

			/* send data to server */
			ml_tf_inferencer_OSC_data_sender(x, NULL, NULL);

			/* if data was too long just letting know the user */
			if (x->full_size < x->filling_index) {
				int diff = x->filling_index - x->full_size;
				post("Data was longer then buffer and it was truncated by: %d", diff); //timestamp();

			}

			/*restarting fillin index for next data flow */
			x->filling_index = 0;

			/*********Sending signal that it is time to predict ******/

			if (x->verbose_flag == 1) { post("Data appended to the servet. Now it's time to predict!"); timestamp(); }
			ml_tf_inferencer_OSC_time_to_predict_sender(x);

			
		}

		x->output_index = 0; /* output index in zeroed out in any case here */

	}
	else { post("Server is not running!"); }
}

/* creating output subthread */
void ml_tf_inferencer_start_output_thread(t_ml_tf_inferencer* x)
{
	if (x->output_thread == NULL) {

		//post("initialising new thread for output!"); timestamp();
		systhread_create((method)ml_tf_inferencer_output, x, 0, 0, 0, &x->output_thread);
	}
}



/******* output function that runs in independent thread and outputs data *******/
/********************** whenever it gets prediction from the server *************/
void ml_tf_inferencer_output(t_ml_tf_inferencer* x)
{	
	char* p;
	double f;
	
	while (true) {
		
		/* this calls threadcontroll class and stops this thread as soon as it starts. It waits for notification from output osc listener */
		x->out_tread_control->waitforit(); 

		if (x->server_predicted !=nullptr && *x->server_predicted) {

			p = strtok((char*)*(x->prediction), " "); /* reading srting seperated by " " */

			//if (atoi(p) == x->PORT_LISTENER) {
			//p = strtok(NULL, " ");

			while (p != NULL)
			{
				f = atof(p);

				outlet_int(x->index_outlet, (t_atom_long)x->output_index);
				outlet_float(x->float_outlet, f);

				p = strtok(NULL, " ");

				x->output_index = x->output_index + 1;

			}
						
		*x->server_predicted = false;	/* reseting back server status for next prediction */

		if (x->verbose_flag == 1) { post("Output_done: %d", x->output_index); timestamp(); }

		}
	}
}

/* stoping output thread */
/* we ended up not useing this function but let it be here */
void ml_tf_inferencer_output_thread_stop(t_ml_tf_inferencer* x)
{
	unsigned int ret;

	if (x->output_thread) {
		post("stopping output thread, that must never happen :))");
		systhread_join(x->output_thread, &ret);					// wait for the thread to stop
		x->output_thread = NULL;
	}
}




// read message input_size and notify python server about the data shape to be expected. 
// this is used to set the size of the array internal buffer and make sure data will be send 
// correctly to python server 

void ml_tf_inferencer_input_size(t_ml_tf_inferencer* x, t_symbol* s, long argc, t_atom* argv)
{
	int i;
	
	for (int i = 0; i < 10; i++) { x->input_size[i] = 0; }; /* setting input size to 0s */
	x->full_size = 1;
	x->filling_index = 0; /* zeroing this out just in case */
	
	if (argc > 10) {
		object_error((t_object*)x, "Input size can't be more than 10 dimentional!");
	}
	else {

		// for each argument after the the message 'input_size' we multiply numbers on each other to calculate full size
		for (i = 0; i < argc; i++) {
			x->input_size[i] = atom_getlong(argv + i);
			x->full_size *= x->input_size[i];
		}

		/* we save inner size of the matrix. It will be used for buffering the matrix data */
		x->inner_size = atom_getlong(argv + i-1);

		/* allocate memory for matrix */// *the memory will be freed when the item is removed.*/
		x->data_buffer = (float*)sysmem_resizeptrclear(x->data_buffer, sizeof(float) * x->full_size);

		/* send intput size to python server so it orginises data on while receiving corespondingly */
		ml_tf_inferencer_OSC_input_size_sender(x);

	}
}



/********************** read Tensorflow/keras network *******************/
void ml_tf_inferencer_read(t_ml_tf_inferencer* x, t_symbol* s)
{
	defer((t_object*)x, (method)ml_tf_inferencer_doread, s, 0, NULL);
}


void ml_tf_inferencer_doread(t_ml_tf_inferencer* x, t_symbol* s, long argc, t_atom* argv)
{
	short path, err;
	t_fourcc type; //= FOUR_CHAR_CODE('h5');
	char file[MAX_PATH_CHARS]; /*name of the file will be used to extract full path */

	if (s == gensym("")) { /* open diealog when no filename is given */
		if (open_dialog(file, &path, &type, &type, 1))
			return;
	}
	else { /* locating file when name is given */
		strcpy(file, s->s_name);
		if (locatefile_extended(file, &path, &type, NULL, 0)) {
			object_error((t_object*)x, "can't find file %s", file);
			return;
		}
	}
	err = path_toabsolutesystempath(path, file, x->filename); /* ablolute path */
	if (err) {
		object_error((t_object*)x, "%s: error %d opening file", file , err);
		return;
	}

	//post("%s", file);
	post("Network imported: %s", x->filename);

	/**********************************************************/
		/*  Osc send the network file name to server */
	/**********************************************************/
	ml_tf_inferencer_OSC_filename_sender(x, x->filename);

}



/****************************************************/
/******************* Python Server ******************/
/****************************************************/


static void ml_tf_inferencer_server(t_ml_tf_inferencer* x, long command)
{
	char text[512];

	/* this text will be pass to shell executer as argunents to set server port numbers  */
	snprintf(text, 512, "--serverport %d --clientport %d", x->PORT_SENDER, x->PORT_LISTENER);

	/******locating server.py file in max directories *********/

	short path, err;
	t_fourcc type;
	char file[MAX_PATH_CHARS] = "server.py";
	char fullpath[MAX_PATH_CHARS];

	if (locatefile_extended(file, &path, &type, NULL, 0)) {
		object_error((t_object*)x, "can't find file %s", file);
		return;
	}
	err = path_toabsolutesystempath(path, file, fullpath);

	/**********************************************************/

	x->lpExecInfo.cbSize = sizeof(SHELLEXECUTEINFO);
	x->lpExecInfo.lpFile = fullpath; //"server.py";
	//x->lpExecInfo.lpFile = "AAAserver.py";
	x->lpExecInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
	x->lpExecInfo.hwnd = NULL;
	x->lpExecInfo.lpVerb = NULL;
	x->lpExecInfo.lpParameters = text; //"--serverport 1000 --clientport 2000"; 

	//x->lpExecInfo.lpDirectory = "..\\..\\..\\..\\..\\..\\ml_tf_inferencer";
	x->lpExecInfo.lpDirectory = NULL; //"..\\..\\..\\..\\Max 8\\Packages\\max-sdk-main\\externals";   //Max 8\\Packages
	x->lpExecInfo.nShow = SW_SHOWNORMAL;
	
	
	if (command == 1) /* Opening the server */
	{
		ShellExecuteEx(&x->lpExecInfo);

		*x->server_ready = false;

		/* wait before python server ready flag becomes true */
		x->server_control->waitforit();

		/* pass input size and gpu status information to server if we know already */
		ml_tf_inferencer_OSC_input_size_sender(x);
		ml_tf_inferencer_OSC_send_gpu_switch(x);
		
		/* if network was loaded we realod network too */
		if (x->filename[0]!=0) { ml_tf_inferencer_OSC_filename_sender(x, x->filename); }; 
	}
	else if (command == 0) {/* Closing the server */
		if (x->lpExecInfo.hProcess)
		{
			TerminateProcess(x->lpExecInfo.hProcess, 0);
			CloseHandle(x->lpExecInfo.hProcess);
			*x->server_ready = false;
		}
		
	}
}


/* gpu usage control */
void ml_tf_inferencer_OSC_send_gpu_switch_change(t_ml_tf_inferencer* x, long argv)
{

	x->gpu_switch = argv;

	if (x->server_ready != nullptr && *x->server_ready) {

		ml_tf_inferencer_server(x, 0); /* close server */
		ml_tf_inferencer_server(x, 1); /* start server over */
	}
}






/****************************************** Some help and debuging functions **************************/

//void ml_tf_inferencer_post(t_ml_tf_inferencer* x)
//{
//	for (int i = 0; i < x->full_size; i++)
//	{
//		post("the variable is %f", *(x->data_buffer + i));
//		//post(ADDRESS);
//	}
//}

///* expoerting memoruy to text file for testing */
//
//void ml_tf_inferencer_file(t_ml_tf_inferencer* x)
//{
//	FILE* f = fopen("AAAmemory.txt", "w");
//	for (unsigned i = 0; i < x->full_size; i++) {
//		fprintf(f, "%f	", *(x->data_buffer + i));
//		if (i > 2 && ((i+1) % x->inner_size == 0)) 
//		{
//			fprintf(f, "\n");
//		}
//	}
//
//	fclose(f);
//}

void timestamp()     /*********this is used to track timing of data flow *****/
{

	SYSTEMTIME t;
	GetSystemTime(&t); // or GetLocalTime(&t)
	post("%02d:%02d.%4d\n",
		t.wMinute, t.wSecond, t.wMilliseconds);

}
/***********************************************************************************************/



/**************************** OSC SERVER *****************************/



/*********************************************************************/
/*************************** Data Send *******************************/
/*********************************************************************/

/* function for data sending */
void ml_tf_inferencer_OSC_data_sender(t_ml_tf_inferencer* x, int argc, char* argv[])
{
	(void)argc; // suppress unused parameter warnings
	(void)argv; // suppress unused parameter warnings


	UdpTransmitSocket transmitSocket(IpEndpointName(ADDRESS, x->PORT_SENDER));

	char buffer[OUTPUT_BUFFER_SIZE];

	osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE);

	/* here will be another for loop for (full size/inner_size) outer size */

	for (long send_index = 0; send_index < x->full_size;)
	{
		p.Clear();
		p << osc::BeginMessage("/data");

		for (int i = send_index; i < x->inner_size + send_index; i++) {
			p << *(x->data_buffer + i);
		}

		p << osc::EndMessage;

		transmitSocket.Send(p.Data(), p.Size());

		send_index = send_index +x->inner_size;

		/* set python import flag to false */
		*x->python_import_done = false;

	
		/* wait before server responds and python import flag becomes true */
		x->python_import_control->waitforit();
		
		//if (send_index == x->full_size) {}
	}
}

/* function to send input size */
void ml_tf_inferencer_OSC_input_size_sender(t_ml_tf_inferencer* x)
{

	UdpTransmitSocket transmitSocket(IpEndpointName(ADDRESS, x->PORT_SENDER));

	char buffer[512];

	osc::OutboundPacketStream p(buffer, 512);

	p << osc::BeginMessage("/input_size");

	for (int i = 0; i < 10; i++) {
		if (x->input_size[i] > 0) {
			p << x->input_size[i];
		}
	}

	p << osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());

}

/* function to send network file name */
void ml_tf_inferencer_OSC_filename_sender(t_ml_tf_inferencer* x, char* filename)
{

	UdpTransmitSocket transmitSocket(IpEndpointName(ADDRESS, x->PORT_SENDER));

	char buffer[4096];

	osc::OutboundPacketStream p(buffer, 4096);

	p << osc::BeginMessage("/filename") << filename << osc::EndMessage;


	transmitSocket.Send(p.Data(), p.Size());

}

/* send signal that server needs to predict now */
void ml_tf_inferencer_OSC_time_to_predict_sender(t_ml_tf_inferencer* x)
{
	int one = 1;

	UdpTransmitSocket transmitSocket(IpEndpointName(ADDRESS, x->PORT_SENDER));
	
	char buffer[32];

	osc::OutboundPacketStream p(buffer, 32);

	p << osc::BeginMessage("/predict") << one << osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());

}

/* gpu status send to server */
void ml_tf_inferencer_OSC_send_gpu_switch(t_ml_tf_inferencer* x) {

	UdpTransmitSocket transmitSocket(IpEndpointName(ADDRESS, x->PORT_SENDER));

	char buffer[512];

	osc::OutboundPacketStream p(buffer, 512);

	p << osc::BeginMessage("/gpu_switch") << x->gpu_switch << osc::EndMessage;

	transmitSocket.Send(p.Data(), p.Size());
}


/************************************************************/
/************************* Listening ************************/
/************************************************************/


class ml_tf_inferencer_packetListener : public osc::OscPacketListener {


public:

	/* these are thread controls that are defined here and will be pluged to look at thread controls from the object */
	thread_control* out_tread_control;
	thread_control* server_control;
	thread_control* python_import_control;

	/* allocating buffer for output data */
	const char* prediction= (char*)sysmem_newptr(sizeof(float) * 257);  // just enought memory for the chunk of output data //sizeof(char) * 8192

	/* these are local variables that will be aslo read form object variables */
	bool server_predicted=false;
	bool server_ready = false;
	bool python_import_done = false;


	virtual void ProcessMessage(const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint)
	{
		(void)remoteEndpoint; // suppress unused parameter warning

		try {
			/* here comes signal that sevrer did append part of the data and needs next part, while sending data to server */
			if (std::strcmp(m.AddressPattern(), "/appended") == 0) {
				osc::ReceivedMessageArgumentStream args = m.ArgumentStream();

				/**** Here we read osc message with local variable. this is read from the main stuct also by pointer ***/
				python_import_control->lock();
				args >> python_import_done >> osc::EndMessage;
				python_import_control->unlock();

				/***** notify data sending function that is waiting for this ****/
				python_import_control->notify();
			}

			/* Here comes signal that server is loaded and running */
			if (std::strcmp(m.AddressPattern(), "/ready") == 0) {
				osc::ReceivedMessageArgumentStream args = m.ArgumentStream();

				/**** Here we read osc message with local variable. this is read from the main stuct also by pointer ***/

				server_control->lock();
				args >> server_ready >> osc::EndMessage;
				server_control->unlock();

				/***** notify main thread server starting function that is waiting for this ****/
				server_control->notify();
			}

			/* here comes prediction data in chunks of 256 */
			if (std::strcmp(m.AddressPattern(), "/prediction") == 0) {
				//osc::ReceivedMessageArgumentStream args = m.ArgumentStream();
				osc::ReceivedMessage::const_iterator arg = m.ArgumentsBegin();

				
				
				out_tread_control->lock();

				/**** Here we read osc message with local variable, that is read from main stuct by pointer ***/
				prediction = (arg++)->AsString(); 
				server_predicted= true;

				out_tread_control->unlock();
								
				///***** notify output thread that is waiting for this ****/
				out_tread_control->notify();

			}

		}
		catch (osc::Exception& e) {

			post("error while parsing message from server!");

		}
	}

};

/* starting listerenr server thread */
void ml_tf_inferencer_OSC_listen_thread(t_ml_tf_inferencer* x)
{
	// create new thread + begin execution
	if (x->listener_thread == NULL) {

		systhread_create((method)ml_tf_inferencer_OSC_listener, x, 0, 0, 0, &x->listener_thread);
	}
}

/* this is actiual listerer server thread */
void *ml_tf_inferencer_OSC_listener(t_ml_tf_inferencer* x, int argc, char* argv[])
{
	(void)argc; // suppress unused parameter warnings
	(void)argv; // suppress unused parameter warnings

	ml_tf_inferencer_packetListener listener;

	UdpListeningReceiveSocket s(
		IpEndpointName(IpEndpointName::ANY_ADDRESS, x->PORT_LISTENER),
		&listener);

	post("Listening to port: %d", x->PORT_LISTENER);

	/* here we point struct variables to listener's local variables */
	x->prediction = &listener.prediction;
	x->server_predicted = &listener.server_predicted;
	x->server_ready = &listener.server_ready;
	x->python_import_done = &listener.python_import_done;

	/* here same thing but other way around - listener's local class (and functions) point at struct class */
	listener.out_tread_control = x->out_tread_control;
	listener.server_control = x->server_control;
	listener.python_import_control = x->python_import_control;

	s.Run();	/* listerenr server will run forever! */

	return NULL;
}