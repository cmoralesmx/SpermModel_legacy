<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:template match="/">
// includes
#include &lt;stdlib.h&gt;
#include &lt;stdio.h&gt;
#include &lt;string.h&gt;
#include &lt;math.h&gt;
//#include &lt;cutil.h&gt;
//#include &lt;cutil_math.h&gt;
#include &lt;helper_cuda.h&gt;
//#include &lt;helper_functions.h&gt;
#include &lt;helper_math.h&gt;
#include &lt;cudpp.h&gt;

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"

/* SM padding and offset variables */
int SM_START;
int PADDING;

//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */
int h_current_iteration_no = 0;
/* END FLAME GPU EXTENSIONS */
//#endif

/* Agent Memory */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
/* <xsl:value-of select="xmml:name"/> Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s;      /**&lt; Pointer to agent list (population) on the device*/
xmachine_memory_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_swap; /**&lt; Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_new;  /**&lt; Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count;   /**&lt; Agent population size counter */ <xsl:if test="gpu:type='discrete'">
int h_xmachine_memory_<xsl:value-of select="xmml:name"/>_block_width;   /**&lt; Agent population size counter */
int h_xmachine_memory_<xsl:value-of select="xmml:name"/>_grid_width;   /**&lt; Agent population size counter */</xsl:if>
uint * d_xmachine_memory_<xsl:value-of select="xmml:name"/>_keys;	  /**&lt; Agent sort identifiers keys*/
uint * d_xmachine_memory_<xsl:value-of select="xmml:name"/>_values;  /**&lt; Agent sort identifiers value */
    <xsl:for-each select="xmml:states/gpu:state">
/* <xsl:value-of select="../../xmml:name"/> state variables */
xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;      /**&lt; Pointer to agent list (population) on host*/
xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;      /**&lt; Pointer to agent list (population) on the device*/
int h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count;   /**&lt; Agent population size counter */ 
</xsl:for-each>
</xsl:for-each>

/* Message Memory */
<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
/* <xsl:value-of select="xmml:name"/> Message variables */
xmachine_message_<xsl:value-of select="xmml:name"/>_list* h_<xsl:value-of select="xmml:name"/>s;         /**&lt; Pointer to message list on host*/
xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s;         /**&lt; Pointer to message list on device*/
xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_swap;    /**&lt; Pointer to message swap list on device (used for holding optional messages)*/
<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">/* Non partitioned and spatial partitioned message variables  */
int h_message_<xsl:value-of select="xmml:name"/>_count;         /**&lt; message list counter*/
int h_message_<xsl:value-of select="xmml:name"/>_output_type;   /**&lt; message output type (single or optional)*/
</xsl:if><xsl:if test="gpu:partitioningSpatial">/* Spatial Partitioning Variables*/
uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys;	  /**&lt; message sort identifier keys*/
uint * d_xmachine_message_<xsl:value-of select="xmml:name"/>_values;  /**&lt; message sort identifier values */
xmachine_message_<xsl:value-of select="xmml:name"/>_PBM * d_<xsl:value-of select="xmml:name"/>_partition_matrix;  /**&lt; Pointer to PCB matrix */
float3 h_message_<xsl:value-of select="xmml:name"/>_min_bounds;           /**&lt; min bounds (x,y,z) of partitioning environment */
float3 h_message_<xsl:value-of select="xmml:name"/>_max_bounds;           /**&lt; max bounds (x,y,z) of partitioning environment */
int3 h_message_<xsl:value-of select="xmml:name"/>_partitionDim;           /**&lt; partition dimensions (x,y,z) of partitioning environment */
float h_message_<xsl:value-of select="xmml:name"/>_radius;                 /**&lt; partition radius (used to determin the size of the partitions) */
</xsl:if><xsl:if test="gpu:partitioningDiscrete">/* Discrete Partitioning Variables*/
int h_message_<xsl:value-of select="xmml:name"/>_range;     /**&lt; range of the discrete message*/
int h_message_<xsl:value-of select="xmml:name"/>_width;     /**&lt; with of the message grid*/
</xsl:if><xsl:if test="gpu:partitioningDiscrete or gpu:partitioningSpatial">/* Texture offset values for host */<xsl:for-each select="xmml:variables/gpu:variable">
int h_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset;</xsl:for-each>
<xsl:if test="gpu:partitioningSpatial">
int h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset;
int h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_offset;
</xsl:if></xsl:if>
</xsl:for-each>

/*Global condition counts*/<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/gpu:globalCondition">
int h_<xsl:value-of select="../xmml:name"/>_condition_count;
</xsl:for-each>

//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */

/*Recursive condition counts*/<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/gpu:recursiveCondition">
    int h_<xsl:value-of select="../xmml:name"/>_condition_count;
  </xsl:for-each>
/* END FLAME GPU EXTENSIONS */
//#endif

/* RNG rand48 */
RNG_rand48* h_rand48;    /**&lt; Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**&lt; Pointer to RNG_rand48 seed list on device*/

/* CUDA Parallel Primatives variables */
CUDPPHandle cudpp_scanplan;   /**&lt; CUDPPHandle*/
CUDPPHandle cudpp_sortplan;   /**&lt; CUDPPHandle*/
int cudpp_last_sum;           /**&lt; Indicates if the position (in message list) of last message*/
int cudpp_last_included;      /**&lt; Indicates if last sum value is included in the total sum count*/
int radix_keybits = 32;

/* Agent function prototypes */
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function">
/** <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>
 * Agent function prototype for <xsl:value-of select="xmml:name"/> function of <xsl:value-of select="../../xmml:name"/> agent
 */
void <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>();
</xsl:for-each>
  
CUDPPHandle* getCUDPPSortPlan(){
    return &amp;cudpp_sortplan;
}


void setPaddingAndOffset()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&amp;deviceProp, 0);
    int x64_sys = 0;

	// This function call returns 9999 for both major &amp; minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 &amp;&amp; deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(0);
	}
    
    //check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
    printf("Simulation requires full precision double values\n");
    if ((deviceProp.major &lt; 2)&amp;&amp;(deviceProp.minor &lt; 3)){
        printf("Error: Hardware does not support full precision double values!\n");
        exit(0);
    }
    
#endif

    //check 32 or 64bit
    x64_sys = (sizeof(void*)==8);
    if (x64_sys)
    {
        printf("64Bit System Detected\n");
    }
    else
    {
        printf("32Bit System Detected\n");
    }

    //check for FERMI
	if ((deviceProp.major >= 2)){
		printf("FERMI Card or better detected (compute >= 2.0)\n");
        if (x64_sys){
            SM_START = 8;
            PADDING = 0;
        }else
        {
            SM_START = 4;
            PADDING = 0;
        }
	}	
    //not fermi
    else{
  	    printf("Pre FERMI Card detected (less than compute 2.0)\n");
        if (x64_sys){
            SM_START = 0;
            PADDING = 4;
        }else
        {
            SM_START = 0;
            PADDING = 4;
        }
    }
  
    //copy padding and offset to GPU
    checkCudaErrors(cudaMemcpyToSymbol( d_SM_START, &amp;SM_START, sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol( d_PADDING, &amp;PADDING, sizeof(int)));

        
}


void initialise(char * inputfile){

    //set the padding and offset values depending on architecture and OS
    setPaddingAndOffset();
  

	printf("Allocating Host and Device memeory\n");
  
	/* Agent memory allocation (CPU) */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	int xmachine_<xsl:value-of select="xmml:name"/>_SoA_size = sizeof(xmachine_memory_<xsl:value-of select="xmml:name"/>_list);<xsl:for-each select="xmml:states/gpu:state">
	h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/> = (xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list*)malloc(xmachine_<xsl:value-of select="../../xmml:name"/>_SoA_size);</xsl:for-each></xsl:for-each>

	/* Message memory allocation (CPU) */<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
	int message_<xsl:value-of select="xmml:name"/>_SoA_size = sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_list);
	h_<xsl:value-of select="xmml:name"/>s = (xmachine_message_<xsl:value-of select="xmml:name"/>_list*)malloc(message_<xsl:value-of select="xmml:name"/>_SoA_size);</xsl:for-each>

    //Exit if agent or message buffer sizes are to small for function outpus<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/xmml:xagentOutputs/gpu:xagentOutput">
    <xsl:variable name="xagent_output" select="xmml:xagentName"/><xsl:variable name="xagent_buffer" select="../../../../gpu:bufferSize"/><xsl:if test="../../../../../gpu:xagent[xmml:name=$xagent_output]/gpu:bufferSize&lt;$xagent_buffer">
    printf("ERROR: <xsl:value-of select="$xagent_output"/> agent buffer is too small to be used for output by <xsl:value-of select="../../../../xmml:name"/> agent in <xsl:value-of select="../../xmml:name"/> function!\n");
    exit(0);
    </xsl:if>    
    </xsl:for-each>
    
	<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message"><xsl:if test="gpu:partitioningDiscrete">
	
	/* Set discrete <xsl:value-of select="xmml:name"/> message variables (range, width)*/
	h_message_<xsl:value-of select="xmml:name"/>_range = <xsl:value-of select="gpu:partitioningDiscrete/gpu:radius"/>; //from xml
	h_message_<xsl:value-of select="xmml:name"/>_width = (int)floor(sqrt((float)xmachine_message_<xsl:value-of select="xmml:name"/>_MAX));
	//check the width
	if (h_message_<xsl:value-of select="xmml:name"/>_width*h_message_<xsl:value-of select="xmml:name"/>_width != xmachine_message_<xsl:value-of select="xmml:name"/>_MAX){
		printf("ERROR: sqrt of <xsl:value-of select="xmml:name"/> message max must be a whole number for a 2D discrete message grid!\n");
		exit(0);
	}
  checkCudaErrors(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_range, &amp;h_message_<xsl:value-of select="xmml:name"/>_range, sizeof(int)));	
	checkCudaErrors(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_width, &amp;h_message_<xsl:value-of select="xmml:name"/>_width, sizeof(int)));
	</xsl:if><xsl:if test="gpu:partitioningSpatial">
			
	/* Set spatial partitioning <xsl:value-of select="xmml:name"/> message variables (min_bounds, max_bounds)*/
	h_message_<xsl:value-of select="xmml:name"/>_radius = (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:radius"/>;
	checkCudaErrors(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_radius, &amp;h_message_<xsl:value-of select="xmml:name"/>_radius, sizeof(float)));	
	    h_message_<xsl:value-of select="xmml:name"/>_min_bounds = make_float3((float)<xsl:value-of select="gpu:partitioningSpatial/gpu:xmin"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:ymin"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:zmin"/>);
	checkCudaErrors(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_min_bounds, &amp;h_message_<xsl:value-of select="xmml:name"/>_min_bounds, sizeof(float3)));	
	h_message_<xsl:value-of select="xmml:name"/>_max_bounds = make_float3((float)<xsl:value-of select="gpu:partitioningSpatial/gpu:xmax"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:ymax"/>, (float)<xsl:value-of select="gpu:partitioningSpatial/gpu:zmax"/>);
	checkCudaErrors(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_max_bounds, &amp;h_message_<xsl:value-of select="xmml:name"/>_max_bounds, sizeof(float3)));	
	h_message_<xsl:value-of select="xmml:name"/>_partitionDim.x = (int)ceil((h_message_<xsl:value-of select="xmml:name"/>_max_bounds.x - h_message_<xsl:value-of select="xmml:name"/>_min_bounds.x)/h_message_<xsl:value-of select="xmml:name"/>_radius);
	h_message_<xsl:value-of select="xmml:name"/>_partitionDim.y = (int)ceil((h_message_<xsl:value-of select="xmml:name"/>_max_bounds.y - h_message_<xsl:value-of select="xmml:name"/>_min_bounds.y)/h_message_<xsl:value-of select="xmml:name"/>_radius);
	h_message_<xsl:value-of select="xmml:name"/>_partitionDim.z = (int)ceil((h_message_<xsl:value-of select="xmml:name"/>_max_bounds.z - h_message_<xsl:value-of select="xmml:name"/>_min_bounds.z)/h_message_<xsl:value-of select="xmml:name"/>_radius);
	checkCudaErrors(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_partitionDim, &amp;h_message_<xsl:value-of select="xmml:name"/>_partitionDim, sizeof(int3)));	
	</xsl:if></xsl:for-each>
	
	
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:if test="gpu:type='discrete'">
	
	/* Set discrete <xsl:value-of select="xmml:name"/> agent variables (on a fixed grid size these can be precalulated unlike with continous agents)*/
	h_xmachine_memory_<xsl:value-of select="xmml:name"/>_block_width = (int)floor(sqrt((float)THREADS_PER_TILE));
	//check block size
	if (h_xmachine_memory_<xsl:value-of select="xmml:name"/>_block_width*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_block_width != THREADS_PER_TILE){
		printf("ERROR: sqrt of THREADS_PER_TILE must be a whole number for 2D discrete agents!\n");
		exit(0);
	}
    //check the agent count
	if ((xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX % THREADS_PER_TILE) != 0){
		printf("ERROR: <xsl:value-of select="xmml:name"/>s agent count must be exactly divisible by THREADS_PER_TILE for 2D discrete agents!\n");
		exit(0);
	}
	int <xsl:value-of select="xmml:name"/>s_total_blocks = xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX/THREADS_PER_TILE;
	h_xmachine_memory_<xsl:value-of select="xmml:name"/>_grid_width = (int)floor(sqrt((float)<xsl:value-of select="xmml:name"/>s_total_blocks));
	//check the grid size
	if (h_xmachine_memory_<xsl:value-of select="xmml:name"/>_grid_width*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_grid_width != <xsl:value-of select="xmml:name"/>s_total_blocks){
		printf("ERROR: sqrt of (<xsl:value-of select="xmml:name"/>s agent count / THREADS_PER_TILE) must be a whole number for 2D discrete agents!\n");
		exit(0);
	}
    </xsl:if></xsl:for-each>


	//read initial states
	readInitialStates(inputfile, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">h_<xsl:value-of select="xmml:name"/>s_<xsl:value-of select="xmml:states/xmml:initialState"/>, &amp;h_xmachine_memory_<xsl:value-of select="xmml:name"/>_<xsl:value-of select="xmml:states/xmml:initialState"/>_count<xsl:if test="position()!=last()">, </xsl:if></xsl:for-each>);
	
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	/* <xsl:value-of select="xmml:name"/> Agent memory allocation (GPU) */
	checkCudaErrors( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s, xmachine_<xsl:value-of select="xmml:name"/>_SoA_size));
	checkCudaErrors( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s_swap, xmachine_<xsl:value-of select="xmml:name"/>_SoA_size));
	checkCudaErrors( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s_new, xmachine_<xsl:value-of select="xmml:name"/>_SoA_size));
    <xsl:if test="gpu:type='continuous'">//continuous agent sort identifiers
    checkCudaErrors( cudaMalloc( (void**) &amp;d_xmachine_memory_<xsl:value-of select="xmml:name"/>_keys, xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
	checkCudaErrors( cudaMalloc( (void**) &amp;d_xmachine_memory_<xsl:value-of select="xmml:name"/>_values, xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));</xsl:if>
    <xsl:for-each select="xmml:states/gpu:state">
	/* <xsl:value-of select="xmml:name"/> memory allocation (GPU) */
	checkCudaErrors( cudaMalloc( (void**) &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_<xsl:value-of select="../../xmml:name"/>_SoA_size));
	checkCudaErrors( cudaMemcpy( d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_<xsl:value-of select="../../xmml:name"/>_SoA_size, cudaMemcpyHostToDevice));
    </xsl:for-each>
	</xsl:for-each>

	<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
	/* <xsl:value-of select="xmml:name"/> Message memory allocation (GPU) */
	checkCudaErrors( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s, message_<xsl:value-of select="xmml:name"/>_SoA_size));
	checkCudaErrors( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>s_swap, message_<xsl:value-of select="xmml:name"/>_SoA_size));
	checkCudaErrors( cudaMemcpy( d_<xsl:value-of select="xmml:name"/>s, h_<xsl:value-of select="xmml:name"/>s, message_<xsl:value-of select="xmml:name"/>_SoA_size, cudaMemcpyHostToDevice));<xsl:if test="gpu:partitioningSpatial">
	checkCudaErrors( cudaMalloc( (void**) &amp;d_<xsl:value-of select="xmml:name"/>_partition_matrix, sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>_PBM)));
	checkCudaErrors( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));
	checkCudaErrors( cudaMalloc( (void**) &amp;d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, xmachine_message_<xsl:value-of select="xmml:name"/>_MAX* sizeof(uint)));</xsl:if><xsl:text>
	</xsl:text></xsl:for-each>	

	/*Set global condition counts*/<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function/gpu:condition">
	h_<xsl:value-of select="../xmml:name"/>_condition_false_count = 0;
	</xsl:for-each>

  // Added on CUDPP 2.3
  CUDPPHandle theCudpp;
  cudppCreate(&amp;theCudpp);
  
  /* CUDPP Init */
  CUDPPConfiguration cudpp_config;
  cudpp_config.op = CUDPP_ADD;
  cudpp_config.datatype = CUDPP_INT;
  cudpp_config.algorithm = CUDPP_SCAN;
  cudpp_config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
  cudpp_scanplan = 0;
  CUDPPResult result = cudppPlan(theCudpp, &amp;cudpp_scanplan, cudpp_config, buffer_size_MAX, 1, 0);
  if (CUDPP_SUCCESS != result)
  {
  printf("Error creating CUDPPPlan\n");
  exit(-1);
  }

  /* Radix sort */
  CUDPPConfiguration cudpp_sort_config;
  cudpp_sort_config.algorithm = CUDPP_SORT_RADIX;
  cudpp_sort_config.datatype = CUDPP_UINT;
  cudpp_sort_config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
  cudpp_sortplan = 0;
  CUDPPResult sort_result = cudppPlan(theCudpp, &amp;cudpp_sortplan, cudpp_sort_config, buffer_size_MAX, 1, 0);  
	if (CUDPP_SUCCESS != result)
	{
		printf("Error creating CUDPPPlan for radix sort\n");
		exit(-1);
	}

	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	checkCudaErrors( cudaMalloc( (void**) &amp;d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
  srand ( (unsigned int)time(NULL) );
	int seed = rand();
	//int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i &lt; buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A &amp; 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) &amp; 0xFFFFFFLL;
	h_rand48->C.x = C &amp; 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) &amp; 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) &lt;&lt; 16) | 0x330E;
	for (unsigned int i = 0; i &lt; buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x &amp; 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) &amp; 0xFFFFFFLL;
	}
	//copy to device
	checkCudaErrors( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

	/* Call all init functions */
	<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:initFunctions/gpu:initFunction">
	<xsl:value-of select="gpu:name"/>();<xsl:text>
	</xsl:text></xsl:for-each>
} 

//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */

void reinit_RNG(int seed) {
	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	//h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	//checkCudaErrors( cudaMalloc( (void**) &amp;d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
  //srand ( time(NULL) );
	//int seed = rand();
	//int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i &lt; buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
h_rand48->A.x = A &amp; 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) &amp; 0xFFFFFFLL;
	h_rand48->C.x = C &amp; 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) &amp; 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) &lt;&lt; 16) | 0x330E;
	for (unsigned int i = 0; i &lt; buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x &amp; 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) &amp; 0xFFFFFFLL;
	}
	//copy to device
	checkCudaErrors( cudaMemcpy( d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));
}

/* END FLAME GPU EXTENSIONS */
//#endif
    

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:if test="gpu:type='continuous'"> <xsl:for-each select="xmml:states/gpu:state">
void sort_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* agents))
{
	dim3 grid;
	dim3 threads;
	int tile_size = (int)ceil((float)h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count/THREADS_PER_TILE);
	grid.x = tile_size;
	threads.x = THREADS_PER_TILE;

	//generate sort keys
	generate_key_value_pairs&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_keys, d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_values, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>);
  //CUT_CHECK_ERROR("Kernel execution failed");

  //sort
  cudppRadixSort(cudpp_sortplan, d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_keys, d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_values, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count);
	//CUT_CHECK_ERROR("Kernel execution failed");

	//reorder agents
	reorder_<xsl:value-of select="../../xmml:name"/>_agents&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_values, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s_swap);
	//CUT_CHECK_ERROR("Kernel execution failed");

	//swap
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_temp = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/> = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = d_<xsl:value-of select="../../xmml:name"/>s_temp;	
}
</xsl:for-each></xsl:if></xsl:for-each>

void cleanup(){

	/* Agent data free*/
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	/* <xsl:value-of select="xmml:name"/> Agent variables */
	checkCudaErrors(cudaFree(d_<xsl:value-of select="xmml:name"/>s));
	checkCudaErrors(cudaFree(d_<xsl:value-of select="xmml:name"/>s_swap));
	checkCudaErrors(cudaFree(d_<xsl:value-of select="xmml:name"/>s_new));
	<xsl:for-each select="xmml:states/gpu:state">
	free( h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>);
	checkCudaErrors(cudaFree(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>));
	</xsl:for-each>
	</xsl:for-each>

	/* Message data free */
	<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message">
	/* <xsl:value-of select="xmml:name"/> Message variables */
	free( h_<xsl:value-of select="xmml:name"/>s);
	checkCudaErrors(cudaFree(d_<xsl:value-of select="xmml:name"/>s));
	checkCudaErrors(cudaFree(d_<xsl:value-of select="xmml:name"/>s_swap));<xsl:if test="gpu:partitioningSpatial">
	checkCudaErrors(cudaFree(d_<xsl:value-of select="xmml:name"/>_partition_matrix));
	checkCudaErrors(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys));
	checkCudaErrors(cudaFree(d_xmachine_message_<xsl:value-of select="xmml:name"/>_values));</xsl:if><xsl:text>
	</xsl:text></xsl:for-each>
  
  cudaDeviceReset();
}

void singleIteration(){

//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */
checkCudaErrors(cudaMemcpyToSymbol( d_current_iteration_no, &amp;h_current_iteration_no, sizeof(int)));

h_current_iteration_no++;
/* END FLAME GPU EXTENSIONS */
//#endif


	/* set all non partitioned and spatial partitionded message counts to 0*/<xsl:for-each select="gpu:xmodel/xmml:messages/gpu:message"><xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">
	h_message_<xsl:value-of select="xmml:name"/>_count = 0;
	//upload to device constant
	checkCudaErrors(cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_count, &amp;h_message_<xsl:value-of select="xmml:name"/>_count, sizeof(int)));
	</xsl:if></xsl:for-each>

	/* Call agent functions in order itterating through the layer functions */
	<xsl:for-each select="gpu:xmodel/xmml:layers/xmml:layer">
	/* Layer <xsl:value-of select="position()"/>*/
	<xsl:for-each select="gpu:layerFunction">
	<xsl:variable name="function" select="xmml:name"/><xsl:for-each select="../../../xmml:xagents/gpu:xagent/xmml:functions/gpu:function[xmml:name=$function]">
	<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>();
	</xsl:for-each></xsl:for-each></xsl:for-each>

			
	//Syncronise thread blocks (and relax)
	cudaThreadSynchronize();
}

/* Environment functions */

<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
void set_<xsl:value-of select="xmml:name"/>(<xsl:value-of select="xmml:type"/>* h_<xsl:value-of select="xmml:name"/>){
	checkCudaErrors(cudaMemcpyToSymbol(<xsl:value-of select="xmml:name"/>, h_<xsl:value-of select="xmml:name"/>, sizeof(<xsl:value-of select="xmml:type"/>)<xsl:if test="xmml:arrayLength">*<xsl:value-of select="xmml:arrayLength"/></xsl:if>));
}
</xsl:for-each>

/* Agent data access functions*/
<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
    
int get_agent_<xsl:value-of select="xmml:name"/>_MAX_count(){
    return xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX;
}

<xsl:for-each select="xmml:states/gpu:state">
int get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count(){
	<xsl:if test="../../gpu:type='continuous'">//continuous agent
	return h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count;
	</xsl:if><xsl:if test="../../gpu:type='discrete'">//discrete agent 
	return xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX;</xsl:if>
}

xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(){
	return d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;
}

xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(){
	return h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>;
}
</xsl:for-each>
<xsl:if test="gpu:type='discrete'">
int get_<xsl:value-of select="xmml:name"/>_population_width(){
  return h_xmachine_memory_<xsl:value-of select="xmml:name"/>_block_width*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_grid_width;
}
</xsl:if>

</xsl:for-each>


/* Agent functions */

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:functions/gpu:function">
/** <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>
 * Agent function prototype for <xsl:value-of select="xmml:name"/> function of <xsl:value-of select="../../xmml:name"/> agent
 */
void <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>(){
	dim3 grid;
	dim3 threads;
	int sm_size;
	
	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0
	<xsl:if test="../../gpu:type='continuous'">
	if (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count == 0)
	{
		return;
	}
	</xsl:if>
	
	//SET GRID AND BLOCK SIZES
	//set tile size depending on agent count, set a 1d grid and block
	int tile_size = (int)ceil((float)h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count/THREADS_PER_TILE);
	grid.x = tile_size;
	threads.x = THREADS_PER_TILE;
	sm_size = SM_START;

	<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:for-each select="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentName"/><xsl:if test="../../../../../gpu:xagent[xmml:name=$xagent_output]/gpu:type='continuous'">
    //FOR <xsl:value-of select="xmml:xagentName"/> AGENT OUTPUT, RESET THE AGENT NEW LIST SCAN INPUT
	reset_<xsl:value-of select="xmml:xagentName"/>_scan_input&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="xmml:xagentName"/>s_new);
	//CUT_CHECK_ERROR("Kernel execution failed");
	</xsl:if></xsl:for-each></xsl:if>

	//******************************** AGENT FUNCTION CONDITION *********************
	<xsl:choose>
	<xsl:when test="xmml:condition"><xsl:if test="../../gpu:type='continuous'">//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION
  	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;
    checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>);
	//CUT_CHECK_ERROR("Kernel execution failed");
	//reset scan input for working lists
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s);
	//CUT_CHECK_ERROR("Kernel execution failed");

	//APPLY FUNCTION FILTER
	<xsl:value-of select="xmml:name"/>_function_filter&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>, d_<xsl:value-of select="../../xmml:name"/>s);
	//CUT_CHECK_ERROR("Kernel execution failed");
		
	//COMPACT CURRENT STATE LIST
	cudppScan(cudpp_scanplan, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
    //reset agent count
    checkCudaErrors( cudaMemcpy( &amp;cudpp_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors( cudaMemcpy( &amp;cudpp_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (cudpp_last_included == 1)
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = cudpp_last_sum+1;
	else		
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = cudpp_last_sum;
	//Scatter into swap
	scatter_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_swap, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>, 0, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	//CUT_CHECK_ERROR("Kernel execution failed");
	//use a temp pointer change working swap list with current state list
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
    //update the device count
    checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));	
		
	//COMPACT WORKING STATE LIST
	cudppScan(cudpp_scanplan, d_<xsl:value-of select="../../xmml:name"/>s->_position, d_<xsl:value-of select="../../xmml:name"/>s->_scan_input, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
    //reset agent count
    checkCudaErrors( cudaMemcpy( &amp;cudpp_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors( cudaMemcpy( &amp;cudpp_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_swap, d_<xsl:value-of select="../../xmml:name"/>s, 0, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
    //CUT_CHECK_ERROR("Kernel execution failed");
	//update working agent count after the scatter
    if (cudpp_last_included == 1)
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = cudpp_last_sum+1;
	else		
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = cudpp_last_sum;
    //use a temp pointer change working swap list with current state list
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = <xsl:value-of select="../../xmml:name"/>s_temp;
	//update the device count
	checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count == 0)
	{
		return;
	}
	
	<xsl:if test="../../gpu:type='continuous'">//Update the grid and block size for the working list size of continuous agent
	tile_size = (int)ceil((float)h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count/THREADS_PER_TILE);
	grid.x = tile_size;
	threads.x = THREADS_PER_TILE;
	</xsl:if>
			
	</xsl:if></xsl:when><xsl:when test="gpu:globalCondition">//THERE IS A GLOBAL CONDITION
	
	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;
	checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	//RESET SCAN INPUTS
	//reset scan input for currentState
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>);
	//CUT_CHECK_ERROR("Kernel execution failed");
	
	//APPLY FUNCTION FILTER
	<xsl:value-of select="xmml:name"/>_function_filter&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>);
	//CUT_CHECK_ERROR("Kernel execution failed");
	
	//GET CONDTIONS TRUE COUNT FROM CURRENT STATE LIST
	cudppScan(cudpp_scanplan, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	//reset agent count
	checkCudaErrors( cudaMemcpy( &amp;cudpp_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors( cudaMemcpy( &amp;cudpp_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	int global_conditions_true = 0;
	if (cudpp_last_included == 1)
		global_conditions_true = cudpp_last_sum+1;
	else		
		global_conditions_true = cudpp_last_sum;
	//check if condition is true for all agents or if max condition count is reached
	if ((global_conditions_true <xsl:choose><xsl:when test="gpu:globalCondition/gpu:mustEvaluateTo='true'">!</xsl:when><xsl:otherwise>=</xsl:otherwise></xsl:choose>= h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count)&amp;&amp;(h_<xsl:value-of select="xmml:name"/>_condition_count &lt; <xsl:value-of select="gpu:globalCondition/gpu:maxItterations"/>))
	{
		h_<xsl:value-of select="xmml:name"/>_condition_count ++;
		return;
	}
	if ((h_<xsl:value-of select="xmml:name"/>_condition_count == <xsl:value-of select="gpu:globalCondition/gpu:maxItterations"/>))
	{
		printf("Global agent condition for <xsl:value-of select="xmml:name"/> funtion reached the maximum number of <xsl:value-of select="gpu:globalCondition/gpu:maxItterations"/> conditions\n");
	}
	
	//RESET THE CONDITION COUNT
	h_<xsl:value-of select="xmml:name"/>_condition_count = 0;
	
	//MAP CURRENT STATE TO WORKING LIST
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
	//set current state count to 0
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = 0;
	checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	
	
	</xsl:when><xsl:otherwise>//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
	//set working count to current state count
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count;
	checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	//set current state count to 0
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = 0;
	checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));	
	</xsl:otherwise>
	</xsl:choose>
  
  <xsl:if test="../../gpu:type='discrete'">//SET 2D BLOCK SIZE FOR DISCRETE AGENTS (sizes are pre calculated)
	grid.x = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_grid_width;
	grid.y = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_grid_width;
	threads.x = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_block_width;
	threads.y = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_block_width;
	sm_size = SM_START;
	</xsl:if>

	//******************************** AGENT FUNCTION *******************************

	<xsl:if test="xmml:outputs/gpu:output"><xsl:if test="../../gpu:type='continuous'">
	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	if (h_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_count + h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count > xmachine_message_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>_MAX){
		printf("Error: Buffer size of <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/> message will be exceeded in function <xsl:value-of select="xmml:name"/>\n");
		exit(0);
	}
	</xsl:if></xsl:if>


	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	//UPDATE SHARED MEMEORY SIZE FOR EACH FUNCTION INPUT
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone">//Continuous agent and message input has no partitioning
	sm_size += (threads.x * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	</xsl:if><xsl:if test="gpu:partitioningDiscrete">//Continuous agent and message input has discrete partitioning
	//Will be reading using texture lookups so sm size can stay the same but need to hold range and width
	sm_size += (threads.x * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	</xsl:if><xsl:if test="gpu:partitioningSpatial">//Continuous agent and message input is spatially partitioned
	sm_size += (threads.x * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
	</xsl:if>
    //all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (threads.x * PADDING);
	</xsl:for-each>
	</xsl:if><xsl:if test="../../gpu:type='discrete'">
    <xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone">//Discrete agent and message input has no partitioning
	sm_size += (threads.x * sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>));
    //all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (threads.x * PADDING);
	</xsl:if><xsl:if test="gpu:partitioningDiscrete">//Discrete agent and message input has discrete partitioning
	int sm_grid_size = (int)pow((float)threads.x+(h_message_<xsl:value-of select="xmml:name"/>_range*2), 2);
	sm_size += (sm_grid_size *sizeof(xmachine_message_<xsl:value-of select="xmml:name"/>)); //update sm size
    sm_size += (sm_grid_size * PADDING);  //offset for avoiding conflicts
	//check that the range is not greater than the square of the block size. If so then there will be too many uncoalesded reads
	if (h_message_<xsl:value-of select="xmml:name"/>_range > (int)threads.x){
		printf("ERROR: Message range is greater than the thread block size. Increase thread block size or reduce the range!");
		exit(0);
	}
	</xsl:if>
    </xsl:for-each>
  
    </xsl:if>
	</xsl:if>
	
	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningDiscrete or gpu:partitioningSpatial">//continuous agent with discrete or partitioned message input uses texture caching
	<xsl:for-each select="xmml:variables/gpu:variable">size_t tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_byte_offset;    
    checkCudaErrors( cudaBindTexture(&amp;tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_byte_offset, tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>, sizeof(int)*xmachine_message_<xsl:value-of select="../../xmml:name"/>_MAX));
	h_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset = (int)tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_byte_offset / sizeof(<xsl:value-of select="xmml:type"/>);
    checkCudaErrors(cudaMemcpyToSymbol( d_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset, &amp;h_tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_offset, sizeof(int)));
    </xsl:for-each><xsl:if test="gpu:partitioningSpatial">//bind pbm start and end indices to textures
    size_t tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_byte_offset;
    size_t tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_byte_offset;
    checkCudaErrors( cudaBindTexture(&amp;tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_byte_offset, tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start, d_<xsl:value-of select="xmml:name"/>_partition_matrix->start, sizeof(int)*xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size));
    h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset = (int)tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_byte_offset / sizeof(int);
    checkCudaErrors(cudaMemcpyToSymbol( d_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset, &amp;h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start_offset, sizeof(int)));
    checkCudaErrors( cudaBindTexture(&amp;tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_byte_offset, tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end, d_<xsl:value-of select="xmml:name"/>_partition_matrix->end, sizeof(int)*xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size));
    h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_offset = (int)tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_byte_offset / sizeof(int);
    checkCudaErrors(cudaMemcpyToSymbol( d_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_offset, &amp;h_tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end_offset, sizeof(int)));
    </xsl:if></xsl:if>
	</xsl:for-each></xsl:if></xsl:if>
	
	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/><xsl:variable name="outputType" select="xmml:outputs/gpu:output/gpu:type"/>
	//SET THE OUTPUT MESSAGE TYPE
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_<xsl:value-of select="xmml:name"/>_output_type = <xsl:value-of select="$outputType"/>;
	checkCudaErrors( cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_output_type, &amp;h_message_<xsl:value-of select="xmml:name"/>_output_type, sizeof(int)));
	<xsl:if test="$outputType='optional_message'">//message is optional so reset the swap
	reset_<xsl:value-of select="xmml:name"/>_swaps&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="xmml:name"/>s_swap);
	//CUT_CHECK_ERROR("Kernel execution failed");
	</xsl:if></xsl:if></xsl:for-each>
	</xsl:if></xsl:if>
	
	
	<xsl:if test="../../gpu:type='continuous'"><xsl:if test="gpu:reallocate='true'">
	//IF CONTINUOUS AGENT CAN REALLOCATE (process dead agents) THEN RESET AGENT SWAPS	
	reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s);
	//CUT_CHECK_ERROR("Kernel execution failed");
	</xsl:if></xsl:if>
	
	//MAIN XMACHINE FUNCTION CALL (<xsl:value-of select="xmml:name"/>)
	//Reallocate   : <xsl:choose><xsl:when test="gpu:reallocate='true'">true</xsl:when><xsl:otherwise>false</xsl:otherwise></xsl:choose>
	//Input        : <xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>
	//Output       : <xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>
	//Agent Output : <xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>
	GPUFLAME_<xsl:value-of select="xmml:name"/>&lt;&lt;&lt;grid, threads, sm_size&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">, d_<xsl:value-of select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/>s_new</xsl:if>
		<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messagename" select="xmml:inputs/gpu:input/xmml:messageName"/>, d_<xsl:value-of select="xmml:inputs/gpu:input/xmml:messageName"/>s<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messagename]"><xsl:if test="gpu:partitioningSpatial">, d_<xsl:value-of select="xmml:name"/>_partition_matrix</xsl:if></xsl:for-each></xsl:if>
		<xsl:if test="xmml:outputs/gpu:output">, d_<xsl:value-of select="xmml:outputs/gpu:output/xmml:messageName"/>s</xsl:if>
		<xsl:if test="gpu:RNG='true'">, d_rand48</xsl:if>);
	//CUT_CHECK_ERROR("Kernel execution failed");
    
    <xsl:if test="../../gpu:type='discrete'">
    //FOR DISCRETE AGENTS RESET GRID AND BLOCK SIZES (1d block required for new agent scattering)
	grid.x = tile_size;
    grid.y = 1;
	threads.x = THREADS_PER_TILE;
    threads.y = 1;
	sm_size = SM_START;
    </xsl:if>
	
	<xsl:if test="xmml:inputs/gpu:input"><xsl:variable name="messageName" select="xmml:inputs/gpu:input/xmml:messageName"/>
	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningDiscrete or gpu:partitioningSpatial">//continuous agent with discrete or partitioned message input uses texture caching
	<xsl:for-each select="xmml:variables/gpu:variable">checkCudaErrors( cudaUnbindTexture(tex_xmachine_message_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>));
	</xsl:for-each><xsl:if test="gpu:partitioningSpatial">//unbind pbm indices
    checkCudaErrors( cudaUnbindTexture(tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_start));
    checkCudaErrors( cudaUnbindTexture(tex_xmachine_message_<xsl:value-of select="xmml:name"/>_pbm_end));
    </xsl:if></xsl:if>
	</xsl:for-each></xsl:if></xsl:if>

	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/><xsl:variable name="outputType" select="xmml:outputs/gpu:output/gpu:type"/><xsl:variable name="xagentName" select="../../xmml:name"/>
	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES
	<xsl:if test="../../gpu:type='continuous'"><xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">
	<xsl:if test="$outputType='optional_message'">//<xsl:value-of select="xmml:name"/> Message Type Prefix Sum
	cudppScan(cudpp_scanplan, d_<xsl:value-of select="xmml:name"/>s_swap->_position, d_<xsl:value-of select="xmml:name"/>s_swap->_scan_input, h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count);
	//Scatter
	scatter_optional_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="xmml:name"/>s, d_<xsl:value-of select="xmml:name"/>s_swap);
	//CUT_CHECK_ERROR("Kernel execution failed");
	</xsl:if></xsl:if>
	</xsl:for-each></xsl:if>
	</xsl:if>
	
	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/><xsl:variable name="outputType" select="xmml:outputs/gpu:output/gpu:type"/><xsl:variable name="xagentName" select="../../xmml:name"/>
	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT <xsl:if test="../../gpu:type='continuous'">
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningNone or gpu:partitioningSpatial">
	<xsl:if test="$outputType='optional_message'">
	checkCudaErrors( cudaMemcpy( &amp;cudpp_last_sum, &amp;d_<xsl:value-of select="xmml:name"/>s_swap->_position[h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors( cudaMemcpy( &amp;cudpp_last_included, &amp;d_<xsl:value-of select="xmml:name"/>s_swap->_scan_input[h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	//If last item in prefix sum was 1 then increase its index to get the count
	if (cudpp_last_included == 1)
		h_message_<xsl:value-of select="xmml:name"/>_count += cudpp_last_sum+1;
	else
		h_message_<xsl:value-of select="xmml:name"/>_count += cudpp_last_sum;
	</xsl:if><xsl:if test="$outputType='single_message'">
	h_message_<xsl:value-of select="xmml:name"/>_count += h_xmachine_memory_<xsl:value-of select="$xagentName"/>_count;	
	</xsl:if>//Copy count to device
	checkCudaErrors( cudaMemcpyToSymbol( d_message_<xsl:value-of select="xmml:name"/>_count, &amp;h_message_<xsl:value-of select="xmml:name"/>_count, sizeof(int)));	
	</xsl:if>
	</xsl:for-each>
	</xsl:if>
	</xsl:if>
	
	<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentOutputs/gpu:xagentOutput/xmml:xagentName"/><xsl:if test="../../../gpu:xagent[xmml:name=$xagent_output]/gpu:type='continuous'">
    //COPY ANY AGENT COUNT BEFORE <xsl:value-of select="../../xmml:name"/> AGENTS ARE KILLED (needed for scatter)
	int <xsl:value-of select="../../xmml:name"/>s_pre_death_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	</xsl:if>
	</xsl:if>
	
	<xsl:if test="../../gpu:type='continuous'"><xsl:if test="gpu:reallocate='true'">
	//FOR CONTINUOUS AGENTS WITH REALLOCATION REMOVE POSSIBLE DEAD AGENTS	
	cudppScan(cudpp_scanplan, d_<xsl:value-of select="../../xmml:name"/>s->_position, d_<xsl:value-of select="../../xmml:name"/>s->_scan_input, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	//Scatter into swap
	scatter_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_swap, d_<xsl:value-of select="../../xmml:name"/>s, 0, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	//CUT_CHECK_ERROR("Kernel execution failed");
	//use a temp pointer to make swap default
	xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* <xsl:value-of select="xmml:name"/>_<xsl:value-of select="../../xmml:name"/>s_temp = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = d_<xsl:value-of select="../../xmml:name"/>s_swap;
	d_<xsl:value-of select="../../xmml:name"/>s_swap = <xsl:value-of select="xmml:name"/>_<xsl:value-of select="../../xmml:name"/>s_temp;
	//reset agent count
	checkCudaErrors( cudaMemcpy( &amp;cudpp_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s_swap->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors( cudaMemcpy( &amp;cudpp_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s_swap->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (cudpp_last_included == 1)
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = cudpp_last_sum+1;
	else
		h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count = cudpp_last_sum;
	//Copy count to device
	checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count, sizeof(int)));	
	</xsl:if></xsl:if>

	<xsl:if test="xmml:xagentOutputs/gpu:xagentOutput"><xsl:for-each select="xmml:xagentOutputs/gpu:xagentOutput">
	<xsl:variable name="xagent_output" select="xmml:xagentName"/><xsl:if test="../../../../../gpu:xagent[xmml:name=$xagent_output]/gpu:type='continuous'">
    //FOR <xsl:value-of select="xmml:xagentName"/> AGENT OUTPUT SCATTER AGENTS 
	cudppScan(cudpp_scanplan, d_<xsl:value-of select="xmml:xagentName"/>s_new->_position, d_<xsl:value-of select="xmml:xagentName"/>s_new->_scan_input, <xsl:value-of select="../../../../xmml:name"/>s_pre_death_count);
	//reset agent count
	int <xsl:value-of select="xmml:xagentName"/>_after_birth_count;
	checkCudaErrors( cudaMemcpy( &amp;cudpp_last_sum, &amp;d_<xsl:value-of select="xmml:xagentName"/>s_new->_position[<xsl:value-of select="../../../../xmml:name"/>s_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors( cudaMemcpy( &amp;cudpp_last_included, &amp;d_<xsl:value-of select="xmml:xagentName"/>s_new->_scan_input[<xsl:value-of select="../../../../xmml:name"/>s_pre_death_count-1], sizeof(int), cudaMemcpyDeviceToHost));
	if (cudpp_last_included == 1)
		<xsl:value-of select="xmml:xagentName"/>_after_birth_count = h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count + cudpp_last_sum+1;
	else
		<xsl:value-of select="xmml:xagentName"/>_after_birth_count = h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count + cudpp_last_sum;
	//check buffer is not exceeded
	if (<xsl:value-of select="xmml:xagentName"/>_after_birth_count > xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_MAX){
		printf("Error: Buffer size of <xsl:value-of select="xmml:xagentName"/> agents in state <xsl:value-of select="xmml:state"/> will be exceeded writing new agents in function <xsl:value-of select="../../xmml:name"/>\n");
		exit(0);
	}
	//Scatter into swap
	scatter_<xsl:value-of select="xmml:xagentName"/>_Agents&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="xmml:xagentName"/>s_<xsl:value-of select="xmml:state"/>, d_<xsl:value-of select="xmml:xagentName"/>s_new, h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count, <xsl:value-of select="../../../../xmml:name"/>s_pre_death_count);
	//CUT_CHECK_ERROR("Kernel execution failed");
	//Copy count to device
	h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count = <xsl:value-of select="xmml:xagentName"/>_after_birth_count;
	checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="xmml:xagentName"/>_<xsl:value-of select="xmml:state"/>_count, sizeof(int)));	
	</xsl:if></xsl:for-each>
	</xsl:if>
	
	<xsl:if test="xmml:outputs/gpu:output"><xsl:variable name="messageName" select="xmml:outputs/gpu:output/xmml:messageName"/>
	<xsl:for-each select="../../../../xmml:messages/gpu:message[xmml:name=$messageName]">
	<xsl:if test="gpu:partitioningSpatial">
	//HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
	//Get message hash values for sorting
	hash_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, d_<xsl:value-of select="xmml:name"/>s);
    checkCudaErrors("Kernel execution failed");
    //Sort
    cudppRadixSort(cudpp_sortplan, d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, h_message_<xsl:value-of select="xmml:name"/>_count);
    //CUT_CHECK_ERROR("Kernel execution failed");
	//reorder and build pcb
	checkCudaErrors(cudaMemset(d_<xsl:value-of select="xmml:name"/>_partition_matrix->start, 0xffffffff, xmachine_message_<xsl:value-of select="xmml:name"/>_grid_size* sizeof(int)));
	int reorder_sm_size = sizeof(unsigned int)*(THREADS_PER_TILE+1);
	reorder_<xsl:value-of select="xmml:name"/>_messages&lt;&lt;&lt;grid, threads, reorder_sm_size&gt;&gt;&gt;(d_xmachine_message_<xsl:value-of select="xmml:name"/>_keys, d_xmachine_message_<xsl:value-of select="xmml:name"/>_values, d_<xsl:value-of select="xmml:name"/>_partition_matrix, d_<xsl:value-of select="xmml:name"/>s, d_<xsl:value-of select="xmml:name"/>s_swap);
	//CUT_CHECK_ERROR("Kernel execution failed");
	//swap ordered list
	xmachine_message_<xsl:value-of select="xmml:name"/>_list* d_<xsl:value-of select="xmml:name"/>s_temp = d_<xsl:value-of select="xmml:name"/>s;
	d_<xsl:value-of select="xmml:name"/>s = d_<xsl:value-of select="xmml:name"/>s_swap;
	d_<xsl:value-of select="xmml:name"/>s_swap = d_<xsl:value-of select="xmml:name"/>s_temp;
	</xsl:if>
	</xsl:for-each>
	</xsl:if>




  <xsl:if test="gpu:recursiveCondition">
    //#ifdef FLAME_GPU_EXTENSIONS
    /* FLAME GPU EXTENSIONS SECTION */

    //THERE IS A RECURSIVE CONDITION

    //RESET SCAN INPUTS
    //reset scan input for currentState
    reset_<xsl:value-of select="../../xmml:name"/>_scan_input&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s);
    //CUT_CHECK_ERROR("Kernel execution failed");

    //APPLY FUNCTION FILTER
    <xsl:value-of select="xmml:name"/>_function_filter&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s);
    //CUT_CHECK_ERROR("Kernel execution failed");

    //GET CONDTIONS TRUE COUNT FROM CURRENT STATE LIST
    cudppScan(cudpp_scanplan, d_<xsl:value-of select="../../xmml:name"/>s->_position, d_<xsl:value-of select="../../xmml:name"/>s->_scan_input, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
  //reset agent count
  checkCudaErrors( cudaMemcpy( &amp;cudpp_last_sum, &amp;d_<xsl:value-of select="../../xmml:name"/>s->_position[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors( cudaMemcpy( &amp;cudpp_last_included, &amp;d_<xsl:value-of select="../../xmml:name"/>s->_scan_input[h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count-1], sizeof(int), cudaMemcpyDeviceToHost));
  int recursive_conditions_true = 0;
  if (cudpp_last_included == 1)
  recursive_conditions_true = cudpp_last_sum+1;
  else
  recursive_conditions_true = cudpp_last_sum;
  //check if condition is true for all agents or if max condition count is reached
  if ((recursive_conditions_true <xsl:choose>
    <xsl:when test="gpu:recursiveCondition/gpu:mustEvaluateTo='true'">!</xsl:when>
    <xsl:otherwise>=</xsl:otherwise>
  </xsl:choose>= h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count)&amp;&amp;(h_<xsl:value-of select="xmml:name"/>_condition_count &lt; <xsl:value-of select="gpu:recursiveCondition/gpu:maxItterations"/>))
  {
  h_<xsl:value-of select="xmml:name"/>_condition_count ++;

    <xsl:choose>
      <xsl:when test="../../gpu:type='continuous'">
        //check the working agents wont exceed the buffer size in the new state list
        if (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count+h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count > xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX){
        printf("Error: Buffer size of <xsl:value-of select="xmml:name"/> agents in state <xsl:value-of select="xmml:currentState"/> will be exceeded moving working agents to next state in function <xsl:value-of select="xmml:name"/>\n");
        exit(0);
        }
        //append agents to next state list
        append_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>, d_<xsl:value-of select="../../xmml:name"/>s, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
        //CUT_CHECK_ERROR("Kernel execution failed");
        //update new state agent size
        h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count += h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
        checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));
      </xsl:when>
      <xsl:when test="../../gpu:type='discrete'">
        //currentState maps to working list
        <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
        d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = d_<xsl:value-of select="../../xmml:name"/>s;
        d_<xsl:value-of select="../../xmml:name"/>s = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
        //set current state count
        h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
        checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));
      </xsl:when>
    </xsl:choose>
    
  
  return <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>();
  }
  if ((h_<xsl:value-of select="xmml:name"/>_condition_count == <xsl:value-of select="gpu:recursiveCondition/gpu:maxItterations"/>))
  {
  printf("Recursive agent condition for <xsl:value-of select="xmml:name"/> funtion reached the maximum number of <xsl:value-of select="gpu:recursiveCondition/gpu:maxItterations"/> conditions\n");
  }

  //RESET THE CONDITION COUNT
  h_<xsl:value-of select="xmml:name"/>_condition_count = 0;
   
/* END FLAME GPU EXTENSIONS */
//#endif
  </xsl:if>



  //************************ MOVE AGENTS TO NEXT STATE ****************************
  <xsl:choose>
    <xsl:when test="../../gpu:type='continuous'">
	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count+h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count > xmachine_memory_<xsl:value-of select="../../xmml:name"/>_MAX){
		printf("Error: Buffer size of <xsl:value-of select="xmml:name"/> agents in state <xsl:value-of select="xmml:nextState"/> will be exceeded moving working agents to next state in function <xsl:value-of select="xmml:name"/>\n");
		exit(0);
	}
	//append agents to next state list
	append_<xsl:value-of select="../../xmml:name"/>_Agents&lt;&lt;&lt;grid, threads&gt;&gt;&gt;(d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:nextState"/>, d_<xsl:value-of select="../../xmml:name"/>s, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count, h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count);
	//CUT_CHECK_ERROR("Kernel execution failed");
	//update new state agent size
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count += h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:nextState"/>_count, sizeof(int)));	
	</xsl:when>
    <xsl:when test="../../gpu:type='discrete'">
    //currentState maps to working list
	<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp = d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>;
	d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/> = d_<xsl:value-of select="../../xmml:name"/>s;
	d_<xsl:value-of select="../../xmml:name"/>s = <xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:currentState"/>_temp;
    //set current state count
	h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count = h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count;
	checkCudaErrors( cudaMemcpyToSymbol( d_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, &amp;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:currentState"/>_count, sizeof(int)));	
	</xsl:when>
  </xsl:choose>
	
}


</xsl:for-each>

</xsl:template>
</xsl:stylesheet>