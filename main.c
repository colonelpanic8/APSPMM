#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>

/*--------------------------------------------------------------------------------*/

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
    #include <mach/mach_time.h>
    typedef uint64_t                    time_delta_t;
    typedef mach_timebase_info_data_t   frequency_t;
#else
    #include <CL/cl.h>
    typedef struct timeval              time_delta_t;
    typedef double                      frequency_t;
#endif

/*--------------------------------------------------------------------------------*/

#define PRINT
#define BLOCK_SIZE 16
#define DEFAULT_MATRIX_SIZE (16*1)
#define DEFAULT_KERNEL_FILENAME ("kernel.cl")
#define problem(...) fprintf(stderr, __VA_ARGS__)
#define BAR "--------------------------------------------------------------------------------\n"

/*--------------------------------------------------------------------------------*/

static const char*
GetErrorString(cl_int error) {
    switch(error)
    {
    case(CL_SUCCESS):                           return "Success";
    case(CL_DEVICE_NOT_FOUND):                  return "Device not found!";
    case(CL_DEVICE_NOT_AVAILABLE):              return "Device not available!";
    case(CL_MEM_OBJECT_ALLOCATION_FAILURE):     return "Memory object allocation failure!";
    case(CL_OUT_OF_RESOURCES):                  return "Out of resources!";
    case(CL_OUT_OF_HOST_MEMORY):                return "Out of host memory!";
    case(CL_PROFILING_INFO_NOT_AVAILABLE):      return "Profiling information not available!";
    case(CL_MEM_COPY_OVERLAP):                  return "Overlap detected in memory copy operation!";
    case(CL_IMAGE_FORMAT_MISMATCH):             return "Image format mismatch detected!";
    case(CL_IMAGE_FORMAT_NOT_SUPPORTED):        return "Image format not supported!";
    case(CL_INVALID_VALUE):                     return "Invalid value!";
    case(CL_INVALID_DEVICE_TYPE):               return "Invalid device type!";
    case(CL_INVALID_DEVICE):                    return "Invalid device!";
    case(CL_INVALID_CONTEXT):                   return "Invalid context!";
    case(CL_INVALID_QUEUE_PROPERTIES):          return "Invalid queue properties!";
    case(CL_INVALID_COMMAND_QUEUE):             return "Invalid command queue!";
    case(CL_INVALID_HOST_PTR):                  return "Invalid host pointer address!";
    case(CL_INVALID_MEM_OBJECT):                return "Invalid memory object!";
    case(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR):   return "Invalid image format descriptor!";
    case(CL_INVALID_IMAGE_SIZE):                return "Invalid image size!";
    case(CL_INVALID_SAMPLER):                   return "Invalid sampler!";
    case(CL_INVALID_BINARY):                    return "Invalid binary!";
    case(CL_INVALID_BUILD_OPTIONS):             return "Invalid build options!";
    case(CL_INVALID_PROGRAM):                   return "Invalid program object!";
    case(CL_INVALID_PROGRAM_EXECUTABLE):        return "Invalid program executable!";
    case(CL_INVALID_KERNEL_NAME):               return "Invalid kernel name!";
    case(CL_INVALID_KERNEL):                    return "Invalid kernel object!";
    case(CL_INVALID_ARG_INDEX):                 return "Invalid index for kernel argument!";
    case(CL_INVALID_ARG_VALUE):                 return "Invalid value for kernel argument!";
    case(CL_INVALID_ARG_SIZE):                  return "Invalid size for kernel argument!";
    case(CL_INVALID_KERNEL_ARGS):               return "Invalid kernel arguments!";
    case(CL_INVALID_WORK_DIMENSION):            return "Invalid work dimension!";
    case(CL_INVALID_WORK_GROUP_SIZE):           return "Invalid work group size!";
    case(CL_INVALID_GLOBAL_OFFSET):             return "Invalid global offset!";
    case(CL_INVALID_EVENT_WAIT_LIST):           return "Invalid event wait list!";
    case(CL_INVALID_EVENT):                     return "Invalid event!";
    case(CL_INVALID_OPERATION):                 return "Invalid operation!";
    case(CL_INVALID_GL_OBJECT):                 return "Invalid OpenGL object!";
    case(CL_INVALID_BUFFER_SIZE):               return "Invalid buffer size!";
    default:                                    return "Unknown error!";
    };
    return "Unknown error";
}

/*--------------------------------------------------------------------------------*/

static unsigned long ReadFromTextFile(FILE *fh, char* buffer, size_t buffer_size) {
    unsigned long count = (unsigned long)fread(buffer, buffer_size, 1, fh);
    buffer[buffer_size] = '\0';
    return count;
}

static char *LoadTextFromFile(const char *filename, unsigned long *size /* returned file size in bytes */) {
  FILE* fh;
  struct stat statbuf;
  
  //Open File.
  fh = fopen(filename, "r");
  if (!fh)
    problem("File did not open successfully\n");
  
  //Get file size.
  stat(filename, &statbuf);
  unsigned long bytes = (*size);
  
  if(size)
    (*size) = (unsigned long)statbuf.st_size;
  bytes = *size;

  //To be returned.
  char *text = (char*)malloc(*size + 1);
  if(!text)
    return 0;
  
  ReadFromTextFile(fh, text, bytes);
  fclose(fh);
  
  return text;
}

/*--------------------------------------------------------------------------------*/

void check_failure(cl_int err) {
  if (err != CL_SUCCESS) {
    problem("%s", GetErrorString(err));
    exit(err);
  }
}

/*--------------------------------------------------------------------------------*/

struct timeval tv_delta(struct timeval start, struct timeval end){
  struct timeval delta = end;
  delta.tv_sec -= start.tv_sec;
  delta.tv_usec -= start.tv_usec;
  if (delta.tv_usec < 0) {
    delta.tv_usec += 1000000;
    delta.tv_sec--;
  }
  return delta;
}

/*--------------------------------------------------------------------------------*/

cl_int *getMatrixFromFile(char *filename, cl_int *size) {
  FILE *fh;
  cl_int *output;
  int i, matrix_size;

  fh = fopen(filename, "r");
  if(!fh)
    problem("file failed to open\n");
  
  fscanf(fh, "%d", &matrix_size);
  *size = matrix_size;
  output = (cl_int *)malloc(sizeof(cl_int)*matrix_size*matrix_size);

  if(!output)
    exit(-1);
  for(i = 0; i < matrix_size*matrix_size; i++) {
      fscanf(fh, "%d", &(output[i]));
  }
  return output;
}

cl_float *randomMatrix(int size) {
  int num = size*size;
  cl_float *output = (cl_float *)malloc(sizeof(cl_float)*num);
  int i, j;
  for(i = 0; i < size; i++) {
    output[i] = INFINITY;
  }
  for(i = 0; i < size; i++) {
    memmove(output + size*i, output, size*sizeof(cl_float));
  }
  srand(time(NULL));
  for(i = 0; i < size; i++) {
    for(j = 0; j < size/4; j++) {
      int index = rand() % size;
      output[size*i + index] = rand() % 400;
    }
    output[size*i + i] = 0;
  }
  return output;
}

cl_uint *initPreds(int size) {
  cl_uint *output = (cl_uint *)malloc(sizeof(cl_uint)*size*size);
  int i;
  for(i = 0; i < size*size; i++) {
    output[i] = i/size + 1;
  }
  return output;
}

void printMatrix(cl_float *matrix, cl_int rows, cl_int cols) {
  int i;
  printf("%4.0f ", (float)-1); 
  for(i = 0; i < cols; i++)
    printf("%4.0d ",i+1); 
  printf("\n");
  for(i=0; i<cols; i++)
    printf("_____");
  printf("____");
    
  
  for(i = 0; i < rows*cols; i++) {
    if(i % cols == 0) {
      printf("\n");
      printf("%4d|", (i/cols + 1));
    }
    if(matrix[i] > 4000000)
      matrix[i] = INFINITY;
    printf("%4.0f ", matrix[i]);
  }
  printf("\n");
}

void printPreds(cl_uint *matrix, cl_int size) {
  int i;
  printf("%4.0f ", (float)-1); 
  for(i = 0; i < size; i++)
    printf("%4.0d ", i+1); 
  printf("\n");
  for(i=0; i<size; i++)
    printf("_____");
  printf("____");
  for(i = 0; i < size*size; i++) {
    if(i % size == 0) {
      printf("\n");
      printf("%4d|", (i/size + 1));
    }
    printf("%4d ", matrix[i]);
  }
  printf("\n");
}

void multiplyMatrix(cl_float *left, cl_float *right, cl_int l_col_length, cl_int size, cl_int r_row_length, cl_float **result) {
  int i, j, k, temp;
  *result = (cl_float *)malloc(l_col_length*r_row_length*sizeof(cl_float));
  for(i = 0; i < l_col_length; i++) {
    for(j = 0; j < r_row_length; j++) {
      temp = 0;
      for(k = 0; k < size; k++) {
	temp += left[i*size + k]*right[r_row_length*k + j];
      }
      (*result)[r_row_length*i + j] = temp;
    }
  }   
}

int areEqual(cl_float *a, cl_float *b, size_t size) {
  size_t i;
  for(i = 0; i < size; i++) {
    if(a[i] != b[i]) {
      return 0;
    }
  }
  return 1;
}

/*--------------------------------------------------------------------------------*/

int main(int argc, char **argv) {
  cl_int err;

  //Get device id.
  cl_device_id device_id;
  cl_platform_id platform = 0;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU , 1, &device_id, NULL);
  check_failure(err);
  
  //Output the name of our device.
  cl_char vendor_name[1024];
  cl_char device_name[1024];
  err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
  err|= clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
  check_failure(err);
  printf("Using %s %s. \n", vendor_name, device_name);
  printf(BAR);
  
  //Create a context.
  cl_context context;
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  check_failure(err);

  //Create a command queue.
  cl_command_queue commands;
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  check_failure(err);

  //Load kernel from file into a string.
  char *source;
  unsigned long source_length = 0;
  
  source = LoadTextFromFile(DEFAULT_KERNEL_FILENAME, &source_length);
  
  //Create our kernel.
  cl_program program;
  cl_kernel kernel;
  program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  check_failure(err);
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];
    
    problem("ERROR: Failed to build program executable! %s\n", GetErrorString(err));
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    problem("%s\n", buffer);
    return EXIT_FAILURE;
  }
  kernel = clCreateKernel(program, "matrix_product", &err);
  check_failure(err);

  printf("Reading matrices from disk.\n");
  printf(BAR);
  //Read matrices from file
  cl_int m_size;
  cl_float *matrix;
  cl_uint *preds_init;
  cl_mem input;
  cl_mem input2;
  cl_mem preds;
  
  if (argc < 2) {
    matrix = randomMatrix(DEFAULT_MATRIX_SIZE);
    preds_init = initPreds(DEFAULT_MATRIX_SIZE);
    m_size = DEFAULT_MATRIX_SIZE;
  } else {
    matrix = randomMatrix(atoi(argv[1]));
    preds_init = initPreds(atoi(argv[1]));
    m_size = atoi(argv[1]);
  }
  
#ifdef PRINT
  if(m_size < 100) {
    printMatrix(matrix, m_size, m_size);
    printf(BAR);
    printPreds(preds_init, m_size);
  }
#endif
  
  printf("Creating data buffers.\n");
  printf(BAR);
  //Create data buffers on the device.
  input = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(cl_float)*m_size*m_size, NULL, NULL);

  input2 = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(cl_float)*m_size*m_size, NULL, NULL);
  
  preds = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(cl_uint)*m_size*m_size, NULL, NULL);
  
  if(!input || !input2 || !preds ) {
    problem("Failed to allocate device memory.\n");
    exit(-1);
  }
  
  err = 0;
  printf("Putting data into device memory.\n");
  printf(BAR);
  //Put data into device Memory.
  err  =  clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, 
  		       sizeof(cl_float)*m_size*m_size, matrix, 0, NULL, NULL);
  err  =  clEnqueueWriteBuffer(commands, preds, CL_TRUE, 0, 
  		       sizeof(cl_uint)*m_size*m_size, preds_init, 0, NULL, NULL);
  check_failure(err);
  
  printf("Setting Kernel Arguments.\n");
  printf(BAR);
  //Set arguments.
  int i = 0;
  err  =  clSetKernelArg(kernel, i++, sizeof(cl_mem), &input);
  err |=  clSetKernelArg(kernel, i++, sizeof(cl_mem), &input2);
  err |=  clSetKernelArg(kernel, i++, sizeof(cl_mem), &preds);
  err |=  clSetKernelArg(kernel, i++, sizeof(cl_int), &m_size);
  check_failure(err);

  //Determine work group size.
  size_t global_size[] = {m_size, m_size};
  size_t local_size[] = {BLOCK_SIZE, BLOCK_SIZE};
  
  printf("Running.\n");
  printf(BAR);
  clFinish(commands);
  //Run our program.
  struct timeval start, end, delta;
  gettimeofday(&start, NULL);
  int exp = 1;
  while(exp < m_size) {
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    check_failure(err);
    clFinish(commands);
    exp = exp*2;
    i = 0;
    clSetKernelArg(kernel, i++, sizeof(cl_mem), &input2);
    clSetKernelArg(kernel, i, sizeof(cl_mem), &input);
    clFinish(commands);
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    clFinish(commands);
    exp = exp*2;
    i = 0;
    clSetKernelArg(kernel, i++, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, i, sizeof(cl_mem), &input2);
    printf("%d\n", exp);
    clFinish(commands);
  }
  gettimeofday(&end, NULL);
  delta = tv_delta(start, end);
  printf("GPU Time: %ld.%06ld\n", 
	 (long int)delta.tv_sec, 
	 (long int)delta.tv_usec);
  printf(BAR);
  
  printf("Getting data.\n");
  printf(BAR);
  //Retrieve and print output.
  cl_float *result;
  cl_uint *preds_result;
  result = (cl_float *)malloc(sizeof(cl_float)*m_size*m_size);
  preds_result = (cl_uint *)malloc(sizeof(cl_int)*m_size*m_size);
  err = clEnqueueReadBuffer(commands, input2, CL_TRUE, 0, sizeof(cl_float)*m_size*m_size,
			    result, 0, NULL, NULL );
  err = clEnqueueReadBuffer(commands, preds, CL_TRUE, 0, sizeof(cl_uint)*m_size*m_size,
			    preds_result, 0, NULL, NULL );
  check_failure(err);
  
  /*
  printMatrix(matrix, m_size, m_size);
  printf("*\n");
  printMatrix(right_matrix, m_size, m_size);
  printf("=\n");
  printMatrix(result, m_size, m_size);
  */

  //Do the computation on the CPU to verify.
  
  printf("Running computation on the CPU.\n");
  printf(BAR);
  gettimeofday(&start, NULL);
  //multiplyMatrix(matrix, right_matrix, m_size, m_size, m_size, &comp_result);
  gettimeofday(&end, NULL);
  delta = tv_delta(start, end);
#ifdef PRINT
  if(m_size < 100) {
    printMatrix(result, m_size, m_size);
    printf(BAR);
    printPreds(preds_result, m_size);
  }
#endif


  printf(BAR);
  printf(BAR);
  printf("Cleanup.\n");
  //Device Cleanup.
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
  clReleaseMemObject(input);
  clReleaseMemObject(input2);

  //Memory Cleanup.
  free(result);
  free(source);
  free(matrix);
  
  return 0;
}
