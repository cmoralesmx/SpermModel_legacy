<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:template match="/">
#include &lt;header.h&gt;
#include &lt;stdio.h&gt;
//#include &lt;helper_functions.h&gt;
#include &lt;helper_timer.h&gt;
#ifdef VISUALISATION
#include &lt;GL/glew.h&gt;
#include &lt;GL/glut.h&gt;
#endif

/* IO Variables*/
char inputfile[1000];          /**&lt; Input path char buffer*/
char outputpath[1000];         /**&lt; Output path char buffer*/
int CUDA_argc;				  /**&lt; number of CUDA arguments*/
char** CUDA_argv;			  /**&lt; CUDA arguments*/

#define OUTPUT_TO_XML 0


/** checkUsage
 * Function to check the correct number of arguments
 * @param arc	main argument count
 * @param argv	main argument values
 * @return true if usage is correct, otherwise false
 */
int checkUsage( int argc, char** argv){

//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */
	printf("FLAMEGPU EXTENSIONS mode\n");
	if(argc &lt; 4)
	{
		printf("Usage: main [XML model data] [Iterations] [Write Out Every n Iterations] [Optional CUDA arguments]\n");
		return false;
	}

	 /* END FLAME GPU EXTENSIONS */
  //#else
    
	//Check usage
#ifdef VISUALISATION
	printf("FLAMEGPU Visualisation mode\n");
	if(argc &lt; 2)
	{
		printf("Usage: main [XML model data] [Optional CUDA arguments]\n");
		return false;
	}
//#else
//	printf("FLAMEGPU Console mode\n");
//	if(argc &lt; 3)
//	{
//		printf("Usage: main [XML model data] [Itterations] [Optional CUDA arguments]\n");
//		return false;
//	}
//#endif
#endif
	return true;
}


/** setFilePaths
 * Function to set global variables for the input XML file and its directory location
 *@param input input path of model xml file
 */
void setFilePaths(char* input){
	//Copy input file
	strcpy(inputfile, input);
	printf("Initial states: %s\n", inputfile);

	//Calculate the output path from the path of the input file
	int i = 0;
	int lastd = -1;
	while(inputfile[i] != '\0')
	{
		/* For windows directories */
		if(inputfile[i] == '\\') lastd=i;
		/* For unix directories */
		if(inputfile[i] == '/') lastd=i;
		i++;
	}
	strcpy(outputpath, inputfile);
	outputpath[lastd+1] = '\0';
	printf("Ouput dir: %s\n", outputpath);
}


void initCUDA(int argc, char** argv){
	//start position of CUDA arguments in arg v
	int CUDA_start_args;
  
//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */
  CUDA_argc = argc-3;
	CUDA_start_args = 4;
/* END FLAME GPU EXTENSIONS */
//  #else
#ifdef VISUALISATION
	//less model file argument
	CUDA_argc = argc-1;
	CUDA_start_args = 2;
//#else
	//less model file and itterations arguments
//	CUDA_argc = argc-2;
//	CUDA_start_args = 3;
//#endif
#endif
	//copy first argument
	CUDA_argv = new char*[CUDA_argc];
	size_t dst_size = strlen(argv[0])+1; //+/0
	CUDA_argv[0] = new char[dst_size];
	strcpy_s(CUDA_argv[0], dst_size, argv[0]);
	
	//all args after FLAME GPU specific are passed to CUDA
	int j = 1;
	for (int i=CUDA_start_args; i&lt;argc; i++){
  dst_size = strlen(argv[i])+1; //+/0
  CUDA_argv[j] = new char[dst_size];
  strcpy_s(CUDA_argv[j], dst_size, argv[i]);
  j++;
  }

  //CUT_DEVICE_INIT(CUDA_argc, CUDA_argv);
  }


  /**
  * Program main (Handles arguments)
  */
  int main( int argc, char** argv)
  {
  //check usage mode
  if (!checkUsage(argc, argv))
  exit(0);

  //get the directory paths
  setFilePaths(argv[1]);

  //initialise CUDA
  initCUDA(argc, argv);

  #ifdef VISUALISATION
  //Init visualisation must be done before simulation init
  initVisualisation();
  #endif

  //initialise the simulation
  initialise(inputfile);


  #ifdef VISUALISATION
  runVisualisation();
  exit(0);
  #else
  //Get the number of itterations
  int itterations = atoi(argv[2]);
  if (itterations == 0)
  {
  printf("Second argument must be an integer (Number of Itterations)\n");
  exit(0);
  }

  //#ifdef FLAME_GPU_EXTENSIONS
  /* FLAME GPU EXTENSIONS SECTION */

  /*int repetitionNo = atoi(argv[3]);

  if (repetitionNo == 0)
  {
  printf("Third argument must be an integer (Repetition No)\n");
  exit(0);
  }*/

  int repetitionNo = 0;

  int writeout_interval = atoi(argv[3]);

  if (writeout_interval == 0)
  {
  printf("Fourth argument must be an integer (writeout interval)\n");
  exit(0);
  }


  runCustomSimulationType(itterations, repetitionNo, writeout_interval);

  /* END FLAME GPU EXTENSIONS */
  <!--#else

  //Benchmark simulation
  StopWatchInterface *timer=NULL;
  sdkCreateTimer(&amp;timer);
  sdkResetTimer(&amp;timer);
  sdkStartTimer(&amp;timer);

  for (int i=0; i&lt; itterations; i++)
	{
		printf("Processing Simulation Step %i", i+1);

		//single simulation itteration
		singleIteration();

		if (OUTPUT_TO_XML)
		{
			saveIterationData(outputpath, i+1, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
				//<xsl:value-of select="xmml:name"/> state <xsl:value-of select="../../xmml:name"/> agents
				get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()<xsl:choose><xsl:when test="position()=last()">);</xsl:when><xsl:otherwise>,</xsl:otherwise></xsl:choose>
				</xsl:for-each>

  printf(": Saved to XML:");
  }

  printf(": Done\n");
  }

  //CUDA stop timing
  cudaThreadSynchronize();
  sdkStopTimer(&amp;timer);
  printf( "Total Processing time: %f (ms)\n", sdkGetTimerValue(&amp;timer));
  sdkDeleteTimer(&amp;timer);
  #endif-->

  #endif

  cleanup();
  //CUT_EXIT(CUDA_argc, CUDA_argv);
  cudaDeviceReset();
  return 0;
  }

  //#ifdef FLAME_GPU_EXTENSIONS
  /* FLAME GPU EXTENSIONS SECTION */

  #include &lt;stdlib.h&gt;
#include &lt;string.h&gt;

char simulationDescription[1000];

#define OUTPUT_TO_FLAME_BINARY 1
#define OUTPUT_TO_BINARY 0
#define OUTPUT_TO_COPY_ONLY 0

extern "C" void setSimulationDescription(char * desc) {
  sprintf(simulationDescription, desc);
}

#pragma region Simulation Execution and Output Functions


/*
 * Runs the simulation and outputs to uncompressed Flame Binary (FLB) format
 */
void runSimulationOutputToFlameBinary(int noOfIterations, int repetitionNo, int writeout_interval) {
  printf("\r\nOutput to Flame Binary (FLB) format\r\n");
  printf("Repetition %i - %i iterations to process\r\n", repetitionNo, noOfIterations);
  printf("Progress Interval: 5 Seconds\r\n\r\n");
  int noOfRecords = noOfIterations + 1;
  createFlameBinaryOutputFile(outputpath, noOfRecords, simulationDescription, repetitionNo);

  //Write initial states to binary file
  saveIterationDataToFlameBinary(0, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
    //<xsl:value-of select="xmml:name"/> state <xsl:value-of select="../../xmml:name"/> agents
    get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()<xsl:choose>
      <xsl:when test="position()=last()">);</xsl:when>
      <xsl:otherwise>,</xsl:otherwise>
    </xsl:choose>
  </xsl:for-each>

  StopWatchInterface *timer=NULL;
  sdkCreateTimer(&amp;timer);
  sdkResetTimer(&amp;timer);
  sdkStartTimer(&amp;timer);

  clock_t begin = clock();
  clock_t current;

  printf("Processing Simulation...\n");

  for (int i=1; i&lt;= noOfIterations; i++) {
  
  

    //single simulation iteration
    singleIteration();

if ((i)%writeout_interval == 0 || (i) == noOfIterations) {
    saveIterationDataToFlameBinary(i, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
      //<xsl:value-of select="xmml:name"/> state <xsl:value-of select="../../xmml:name"/> agents
      get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()<xsl:choose>
        <xsl:when test="position()=last()">);</xsl:when>
        <xsl:otherwise>,</xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>
  }
  current = clock();
  if (((current - begin)/(float)CLOCKS_PER_SEC) >= 5) {
  begin = current;
  printf("Completed Simulation Step %i\n", i);
  }

  }

  //CUDA stop timing
  cudaThreadSynchronize();
  sdkStopTimer(&amp;timer);
  float accurateRunTime = sdkGetTimerValue(&amp;timer);
  __int64 simulationRunTime = (__int64)lroundf(accurateRunTime);

  int totalSeconds = (int)floor(accurateRunTime / (float)1000);
  int totalMinutes =  (int)floor(totalSeconds / (float)60);
  int totalHours =  (int)floor(totalMinutes / (float)60);
  float remainderms = ((accurateRunTime/1000.0f) - ((((totalHours * 60) + totalMinutes) * 60)));

  printf( "Total Processing time: %f (ms), [%02i:%02i:%07.4f] (hh:mm:ss.mmmm)\n", accurateRunTime, totalHours, totalMinutes, remainderms);
  sdkDeleteTimer(&amp;timer);

  closeFlameBinaryOutputFile(noOfRecords, simulationRunTime);
  }


  /*
  * Runs the simulation and outputs to intermediary binary format for conversion to FLB
  */
  void runSimulationOutputToBinary(int noOfIterations) {
  printf("\r\nOutput to Multi-File Binary (tmp) format\r\n");
  printf("%i iterations to process\r\n\r\n", noOfIterations);
  createBinaryOutputFile(outputpath);

  //Write initial states to binary file
  saveIterationDataToBinary(outputpath, 0, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
    //<xsl:value-of select="xmml:name"/> state <xsl:value-of select="../../xmml:name"/> agents
    get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()<xsl:choose>
      <xsl:when test="position()=last()">);</xsl:when>
      <xsl:otherwise>,</xsl:otherwise>
    </xsl:choose>
  </xsl:for-each>

  StopWatchInterface *timer=NULL;
  sdkCreateTimer(&amp;timer);
  sdkResetTimer(&amp;timer);
  sdkStartTimer(&amp;timer);

  for (int i=1; i&lt;= noOfIterations; i++) {
    
    //Only output every 10 processing step messages - small increase in performance
    if ((i)%10 == 0) {
      printf("Processing Simulation Step %i\n", i);
    }

    //single simulation iteration
    singleIteration();

    saveIterationDataToBinary(outputpath, i, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
      //<xsl:value-of select="xmml:name"/> state <xsl:value-of select="../../xmml:name"/> agents
      get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()<xsl:choose>
        <xsl:when test="position()=last()">);</xsl:when>
       <xsl:otherwise>,</xsl:otherwise>
     </xsl:choose>
    </xsl:for-each>
  }

  //CUDA stop timing
  cudaThreadSynchronize();
  sdkStopTimer(&amp;timer);
  float accurateRunTime = sdkGetTimerValue(&amp;timer);
  printf( "Total Processing time: %f (ms)\n", accurateRunTime);
  sdkDeleteTimer(&amp;timer);

  closeBinaryOutputFile();
  }


  /*
  * Runs the simulation and outputs to traditional Flame XML format
  */
  void runSimulationOutputToXML(int noOfIterations) {
  printf("\r\nOutput to Multi-File XML (xml) format\r\n");
  printf("%i iterations to process\r\n\r\n", noOfIterations);
  
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&amp;timer);
  sdkResetTimer(&amp;timer);
  sdkStartTimer(&amp;timer);
  
  for (int i=1; i&lt;= noOfIterations; i++) {
    //Only output every 10 processing step messages - small increase in performance
    if ((i)%10 == 0) {
      printf("Processing Simulation Step %i\n", i);
    }

    //single simulation itteration
    singleIteration();

    saveIterationData(outputpath, i, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
      //<xsl:value-of select="xmml:name"/> state <xsl:value-of select="../../xmml:name"/> agents
      get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()<xsl:choose>
        <xsl:when test="position()=last()">);</xsl:when>
        <xsl:otherwise>,</xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>
  }
  // printf(": Saved to XML:");
  //CUDA stop timing
  cudaThreadSynchronize();
  sdkStopTimer(&amp;timer);
  float accurateRunTime = sdkGetTimerValue(&amp;timer);
  sdkDeleteTimer( &amp;timer);
  printf( "Total Processing time: %f (ms)\n", accurateRunTime);
  }

  /*
  * Runs the simulation copies the data to the CPU every iteration for performance testing
  */
  void runSimulationCopyDataOnly(int noOfIterations) {
  printf("\r\nNo Output - Copy Data Only for Performance tests\r\n");
  printf("%i iterations to process\r\n\r\n", noOfIterations);

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&amp;timer);
  sdkResetTimer(&amp;timer);
  sdkStartTimer(&amp;timer);

  for (int i=1; i&lt;= noOfIterations; i++) {
    //Only output every 10 processing step messages - small increase in performance
	  if ((i)%10 == 0) {
	  	printf("Processing Simulation Step %i\n", i);
	  }

	  //single simulation itteration
	  singleIteration();

	  saveIterationDataToCopyOnly(<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
      //<xsl:value-of select="xmml:name"/> state <xsl:value-of select="../../xmml:name"/> agents
      get_host_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_device_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_agents(), get_agent_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count()<xsl:choose>
        <xsl:when test="position()=last()">);</xsl:when>
        <xsl:otherwise>,</xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>

  }
  // printf(": Saved to XML:");
  //CUDA stop timing
  cudaThreadSynchronize();
  sdkStopTimer(&amp;timer);
  float accurateRunTime = sdkGetTimerValue(&amp;timer);
  sdkDeleteTimer( &amp;timer);
  printf( "Total Processing time: %f (ms)\n", accurateRunTime);
  }

  /*
  * Runs the simulation and does not copy any data to the CPU - for performance testing
  */
  void runSimulationNoDataTransfer(int noOfIterations) {
  printf("\r\nNo Output\r\n");
  printf("%i iterations to process\r\n\r\n", noOfIterations);

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&amp;timer);
  sdkResetTimer(&amp;timer);
  sdkStartTimer(&amp;timer);

  for (int i=1; i&lt;= noOfIterations; i++) {
  //Only output every 10 processing step messages - small increase in performance
  if ((i)%10 == 0) {
  printf("Processing Simulation Step %i\n", i);
  }

  //single simulation itteration
  singleIteration();
  }

  //CUDA stop timing
  cudaThreadSynchronize();
  sdkStopTimer(&amp;timer);
  float accurateRunTime = sdkGetTimerValue(&amp;timer);
  sdkDeleteTimer( &amp;timer);
  printf( "Total Processing time: %f (ms)\n", accurateRunTime);
  }

  extern "C" void runCustomSimulationType(int noOfIterations, int repetitionNo, int writeout_interval) {
  if (OUTPUT_TO_FLAME_BINARY) {
  runSimulationOutputToFlameBinary(noOfIterations, repetitionNo, writeout_interval);
  }
  else if (OUTPUT_TO_BINARY) {
  runSimulationOutputToBinary(noOfIterations);
  }
  else if (OUTPUT_TO_COPY_ONLY) {
  runSimulationCopyDataOnly(noOfIterations);
  }
  else if (OUTPUT_TO_XML) {
  runSimulationOutputToXML(noOfIterations);
  }
  else {
  runSimulationNoDataTransfer(noOfIterations);
  }
  }

  #pragma endregion

  /* END FLAME GPU EXTENSIONS */
  //#endif

</xsl:template>
</xsl:stylesheet>