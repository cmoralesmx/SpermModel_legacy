<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
                xmlns:xmml="http://www.dcs.shef.ac.uk/~paul/XMML"
                xmlns:gpu="http://www.dcs.shef.ac.uk/~paul/XMMLGPU">
<xsl:output method="text" version="1.0" encoding="UTF-8" indent="yes" />
<xsl:template match="/">
#include &lt;stdlib.h&gt;
#include &lt;stdio.h&gt;
#include &lt;string.h&gt;
#include &lt;math.h&gt;
//#include &lt;cutil.h&gt;
#include &lt;helper_math.h&gt;
#include &lt;limits.h&gt;

// include header
#include &lt;header.h&gt;

  float3 agent_maximum;
  float3 agent_minimum;

  //#ifdef FLAME_GPU_EXTENSIONS
  /* FLAME GPU EXTENSIONS SECTION */

  char variantDescription[1000];
  char dataPath[1000];

  int getDataPath(char* path) {
  sprintf(path, "%s", dataPath);
  return (int)strlen(path);
  }

  /* END FLAME GPU EXTENSIONS */
  //#endif

  void saveIterationData(char* outputpath, int iteration_number, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, int h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>)
{
	//Device to host memory transfer
	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
	cudaMemcpy( h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, sizeof(xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list), cudaMemcpyDeviceToHost);</xsl:for-each>
	
	/* Pointer to file */
	FILE *file;
	char data[1000];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing itteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
	fputs("&lt;states&gt;\n&lt;itno&gt;", file);
	sprintf(data, "%i", iteration_number);
	fputs(data, file);
	fputs("&lt;/itno&gt;\n", file);
	fputs("&lt;environment&gt;\n" , file);
	fputs("&lt;/environment&gt;\n" , file);

	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state"><xsl:variable name="stateName" select="xmml:name"/>//Write each <xsl:value-of select="../../xmml:name"/> agent to xml
	for (int i=0; i&lt;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count; i++){
		fputs("&lt;xagent&gt;\n" , file);
		fputs("&lt;name&gt;<xsl:value-of select="../../xmml:name"/>&lt;/name&gt;\n", file);
		<xsl:for-each select="../../xmml:memory/gpu:variable">
		fputs("&lt;<xsl:value-of select="xmml:name"/>&gt;", file);
		sprintf(data, "%<xsl:choose><xsl:when test="xmml:type='int'">i</xsl:when><xsl:otherwise>f</xsl:otherwise></xsl:choose>", h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="$stateName"/>-><xsl:value-of select="xmml:name"/>[i]);
		fputs(data, file);
		fputs("&lt;/<xsl:value-of select="xmml:name"/>&gt;\n", file);
		</xsl:for-each>
		fputs("&lt;/xagent&gt;\n", file);
	}
	</xsl:for-each>
	

	fputs("&lt;/states&gt;\n" , file);
	
	/* Close the file */
	fclose(file);
}

void readInitialStates(char* inputpath, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">xmachine_memory_<xsl:value-of select="xmml:name"/>_list* h_<xsl:value-of select="xmml:name"/>s, int* h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>)
{

//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */
int in_environment = 0;
int in_env_variant_description = 0;
int in_env_data_path = 0;
<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
	int in_env_<xsl:value-of select="xmml:name"/>;</xsl:for-each>
  /* Variables for environment data */<xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable"><xsl:text>
	</xsl:text><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text>env_<xsl:value-of select="xmml:name"/><xsl:if test="xmml:defaultValue"> = (<xsl:value-of select="xmml:type"/>)<xsl:value-of select="xmml:defaultValue"/></xsl:if>;</xsl:for-each>
/* END FLAME GPU EXTENSIONS */
//#endif
	int temp = 0;
	int* itno = &amp;temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	char buffer[10000];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_name;<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">
	int in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;</xsl:for-each>

	/* for continuous agents: set agent count to zero */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent"><xsl:if test="gpu:type='continuous'">	
	*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count = 0;</xsl:if></xsl:for-each>
	
	/* Variables for initial state data */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable"><xsl:text>
	</xsl:text><xsl:value-of select="xmml:type"/><xsl:text> </xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;</xsl:for-each>
	
	/* Open config file to read-only */
	if((file = fopen(inputpath, "r"))==NULL)
	{
		printf("error opening initial states\n");
		exit(0);
	}
	
	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
	i = 0;
	in_tag = 0;
	in_itno = 0;
	in_name = 0;<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">
	in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = 0;</xsl:for-each>

	<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
	//set all <xsl:value-of select="xmml:name"/> values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k&lt;xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX; k++)
	{	<xsl:for-each select="xmml:memory/gpu:variable">
		h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[k] = 0;</xsl:for-each>
	}
	</xsl:for-each>

	/* Default variables for memory */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable"><xsl:text>
	</xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = <xsl:choose><xsl:when test="xmml:defaultValue"><xsl:value-of select="xmml:defaultValue"/></xsl:when><xsl:otherwise>0</xsl:otherwise></xsl:choose>;</xsl:for-each>

	/* Read file until end of xml */
	while(reading==1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);
		
		/* If the end of a tag */
		if(c == '&gt;')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;
			
			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */
      if(strcmp(buffer, "environment") == 0) in_environment = 1;
			if(strcmp(buffer, "/environment") == 0) in_environment = 0;
/* END FLAME GPU EXTENSIONS */
//#endif
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
			if(strcmp(buffer, "/xagent") == 0)
			{
				<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
				<xsl:if test="position()!=1">else </xsl:if>if(strcmp(agentname, "<xsl:value-of select="xmml:name"/>") == 0)
				{		
					if (*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count > xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent <xsl:value-of select="xmml:name"/> exceeded whilst reading data\n", xmachine_memory_<xsl:value-of select="xmml:name"/>_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    <xsl:for-each select="xmml:memory/gpu:variable">
					h_<xsl:value-of select="../../xmml:name"/>s-><xsl:value-of select="xmml:name"/>[*h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_count] = <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    <xsl:if test="xmml:name='x'">//Check maximum x value
                    if(agent_maximum.x &lt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_maximum.x = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if>
                    <xsl:if test="xmml:name='y'">//Check maximum y value
                    if(agent_maximum.y &lt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_maximum.y = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if>
                    <xsl:if test="xmml:name='z'">//Check maximum z value
                    if(agent_maximum.z &lt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_maximum.z = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if>
                    <xsl:if test="xmml:name='x'">//Check minimum x value
                    if(agent_minimum.x &gt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_minimum.x = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if>
                    <xsl:if test="xmml:name='y'">//Check minimum y value
                    if(agent_minimum.y &gt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_minimum.y = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if>
                    <xsl:if test="xmml:name='z'">//Check minimum z value
                    if(agent_minimum.z &gt; <xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>)
                        agent_minimum.z = (float)<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>;
                    </xsl:if></xsl:for-each>
					(*h_xmachine_memory_<xsl:value-of select="xmml:name"/>_count) ++;
					
					
				}
				</xsl:for-each>else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable"><xsl:text>
				</xsl:text><xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = <xsl:choose><xsl:when test="xmml:defaultValue"><xsl:value-of select="xmml:defaultValue"/></xsl:when><xsl:otherwise>0</xsl:otherwise></xsl:choose>;</xsl:for-each>
			}
			<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">if(strcmp(buffer, "<xsl:value-of select="xmml:name"/>") == 0) in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = 1;
			if(strcmp(buffer, "/<xsl:value-of select="xmml:name"/>") == 0) in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = 0;
			</xsl:for-each>
			
//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */
  
   if(strcmp(buffer, "VariantDescription") == 0) in_env_variant_description = 1;
	 if(strcmp(buffer, "/VariantDescription") == 0) in_env_variant_description = 0;
   if(strcmp(buffer, "DataPath") == 0) in_env_data_path = 1;
	 if(strcmp(buffer, "/DataPath") == 0) in_env_data_path = 0;

  <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">if(strcmp(buffer, "<xsl:value-of select="xmml:name"/>") == 0) in_env_<xsl:value-of select="xmml:name"/> = 1;
			if(strcmp(buffer, "/<xsl:value-of select="xmml:name"/>") == 0) in_env_<xsl:value-of select="xmml:name"/> = 0;
			</xsl:for-each>
/* END FLAME GPU EXTENSIONS */
//#endif
			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '&lt;')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;
			
			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */
if (in_environment) {
  if (in_env_variant_description) {sprintf(variantDescription, buffer);}
  if (in_env_data_path) {sprintf(dataPath, buffer);}
        <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable"><xsl:text>
	</xsl:text>if (in_env_<xsl:value-of select="xmml:name"/>) {env_<xsl:value-of select="xmml:name"/> = (<xsl:value-of select="xmml:type"/>) ato<xsl:choose><xsl:when test="xmml:type='int'">i</xsl:when><xsl:otherwise>f</xsl:otherwise></xsl:choose>(buffer);}</xsl:for-each>
      }
/* END FLAME GPU EXTENSIONS */
//#endif
			else
			{
				<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:memory/gpu:variable">if(in_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>){ 
					<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/> = (<xsl:value-of select="xmml:type"/>) ato<xsl:choose><xsl:when test="xmml:type='int'">i</xsl:when><xsl:otherwise>f</xsl:otherwise></xsl:choose>(buffer);
				}
				</xsl:for-each>
			}
			
			/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
			buffer[i] = c;
			i++;
		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
	/* Close the file */
	fclose(file);
//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */
  /* set GPU constants */
  <xsl:for-each select="gpu:xmodel/gpu:environment/gpu:constants/gpu:variable">
    set_<xsl:value-of select="xmml:name"/>(&amp;env_<xsl:value-of select="xmml:name"/>);</xsl:for-each>
/* END FLAME GPU EXTENSIONS */
//#endif
}

float3 getMaximumBounds(){
    return agent_maximum;
}

float3 getMinimumBounds(){
    return agent_minimum;
}

//#ifdef FLAME_GPU_EXTENSIONS
/* FLAME GPU EXTENSIONS SECTION */

//#include &lt;cutil_math.h&gt;
#include &lt;time.h&gt;
extern "C" int __cdecl _fseeki64(FILE *, __int64, int);
extern "C" __int64 __cdecl _ftelli64(FILE *);

#pragma region Variables


  /* Pointer to binary file */
  FILE *file = NULL;

  int fileNumberCounter = 0;
  const int MAX_FILE_SIZE = 2000000000;

  __int64 iterationOffsetLocation = -1;
  //__int64* iterationOffsetTable = NULL;


#pragma endregion

#pragma region File Handler Functions

  void createBinaryOutputFile(char* outputpath) {
  char data[1000];
  sprintf(data, "%sTMP_IX%03i.tmp", outputpath, fileNumberCounter);
  file = fopen(data, "wb+");
  }

  void closeBinaryOutputFile() {
  if (file != NULL) {
  fclose(file);
  file = NULL;
  }
  }

  void ToNext4ByteBoundary() {
  __int64 pt = _ftelli64(file);
  __int64 offset = pt % 4;
  if (offset > 0) {
  _fseeki64(file, 4-offset, SEEK_CUR);
  }
  }

  void ToNext8ByteBoundary() {
  __int64 pt = _ftelli64(file);
  __int64 offset = pt % 8;
  if (offset > 0) {
  _fseeki64(file, 8-offset, SEEK_CUR);
  }
  }

  void ToNext16ByteBoundary() {
  __int64 pt = _ftelli64(file);
  __int64 offset = pt % 16;
  if (offset > 0) {
  _fseeki64(file, 16-offset, SEEK_CUR);
  }
  }

  void EnsureFileSize(char* outputpath) {
  _fseeki64(file, 0, SEEK_END);
  __int64 pos = _ftelli64(file);
  if (pos > MAX_FILE_SIZE) {
  fclose(file);
  fileNumberCounter++;
  createBinaryOutputFile(outputpath);
  }
  }

  inline void writeString(const char* string, int strLen, FILE* f) {
  fwrite(&amp;strLen, sizeof(int), 1, f);
  fwrite(string, sizeof(char), strLen, f);
  }
  
  #pragma endregion


#pragma region Flame Binary Specific Functions

  void writeFlameBinaryHeader(int noOfRecords, const char* simulationDescription, const char* variantDescription, int repetitionNo) {
    char isCompressed = 0;
    int noOfAgentTypes = <xsl:value-of select="count(gpu:xmodel/xmml:xagents/gpu:xagent)"></xsl:value-of>;
    __int64 simulationDateTime = (__int64)time(NULL);
    __int64 simulationRunTime = 0;
    fwrite(&amp;isCompressed, sizeof(char), 1, file);
    fwrite(&amp;simulationDateTime, sizeof(__int64), 1, file);
    fwrite(&amp;simulationRunTime, sizeof(__int64), 1, file);
    fwrite(&amp;noOfAgentTypes, sizeof(int), 1, file);
    fwrite(&amp;noOfRecords, sizeof(int), 1, file);
    fwrite(&amp;repetitionNo, sizeof(int), 1, file);
    writeString("Flame GPU 1.0.3 Extension 0.1", 29, file);
    writeString(simulationDescription, (int)strlen(simulationDescription), file);
    writeString(variantDescription, (int)strlen(variantDescription), file);
  }

  void writeParameter(char* parameterName, int nameLength, int typeID) {
  fwrite(&amp;typeID, sizeof(int), 1, file);
  writeString(parameterName, nameLength, file);
  }

  void writeAgent(int currentAgentTypeID, char* agentName, int nameLength, int noOfParameters) {
  fwrite(&amp;currentAgentTypeID, sizeof(int), 1, file);
  writeString(agentName, nameLength, file);
  fwrite(&amp;noOfParameters, sizeof(int), 1, file);
  }

  void writeFlameBinaryAgentHeaders() {
  int currentAgentTypeID = 0;
  <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
    writeAgent(currentAgentTypeID++, "<xsl:value-of select="xmml:name"/>", <xsl:value-of select="string-length(xmml:name)"/>,  <xsl:value-of select="count(xmml:memory/gpu:variable)"></xsl:value-of>);<xsl:for-each select="xmml:memory/gpu:variable">
    writeParameter("<xsl:value-of select="xmml:name"/>", <xsl:value-of select="string-length(xmml:name)"/>, <xsl:choose>
      <xsl:when test="xmml:type='int'">0</xsl:when>
      <xsl:when test="xmml:type='long'">1</xsl:when>
      <xsl:when test="xmml:type='float'">2</xsl:when>
      <xsl:when test="xmml:type='double'">3</xsl:when>
      <xsl:otherwise>4</xsl:otherwise>
    </xsl:choose>);</xsl:for-each>
    
  </xsl:for-each>
  }
  
  void createFlameBinaryOutputFile(char* outputpath, int noOfRecords, const char* simulationDescription, int repetitionNo) {
  char data[1000];
  sprintf(data, "%soutput_%03i.flb", outputpath, repetitionNo);
  file = fopen(data, "wb+");
  writeFlameBinaryHeader(noOfRecords, simulationDescription, variantDescription, repetitionNo);
  writeFlameBinaryAgentHeaders();
  ToNext16ByteBoundary();
  iterationOffsetLocation = _ftelli64(file);
  //iterationOffsetTable = (__int64*)malloc((noOfRecords) * sizeof(__int64));

  _fseeki64(file, sizeof(__int64) * (noOfRecords), SEEK_CUR);
  }

  void closeFlameBinaryOutputFile(int noOfRecords, __int64 simulationRunTime) {
  if (file != NULL) {
  _fseeki64(file, sizeof(char) + sizeof(__int64), SEEK_SET);
  fwrite(&amp;simulationRunTime, sizeof(__int64), 1, file);
  //_fseeki64(file, iterationOffsetLocation, SEEK_SET);
  //writeIterationOffsetTable(noOfRecords);
  _fseeki64(file, 0, SEEK_END);
  fclose(file);
  file = NULL;
  }
  }
  
  #pragma endregion
  
  #pragma region Save Iteration Data Functions

  
void saveIterationDataToFlameBinary(int iteration_number, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
    xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, int h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if>
  </xsl:for-each>) {
  //Device to host memory transfer
  <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
    cudaMemcpy( h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, sizeof(xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list), cudaMemcpyDeviceToHost);</xsl:for-each>

  
  ToNext4ByteBoundary();
  //fwrite(&amp;iteration_number, sizeof(int), 1, file);
  __int64 chunkOffsetPosition = _ftelli64(file);
  
  _fseeki64(file, chunkOffsetPosition, SEEK_SET);
  //iterationOffsetTable[iteration_number] = chunkOffsetPosition;

  int chunkSize = 0;
  fwrite(&amp;chunkSize, sizeof(int), 1, file);
  int noOfAgents = 0;
  fwrite(&amp;noOfAgents, sizeof(int), 1, file);

  int currentAgentTypeID = 0;

  <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
    <xsl:for-each select="xmml:states/gpu:state">
      <xsl:variable name="stateName" select="xmml:name"/>//Write each <xsl:value-of select="../../xmml:name"/> agent to binary
      for (int i=0; i&lt;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count; i++){
      noOfAgents++;
      fwrite(&amp;currentAgentTypeID, sizeof(int), 1, file);
      <xsl:for-each select="../../xmml:memory/gpu:variable">
        fwrite(&amp;h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="$stateName"/>-><xsl:value-of select="xmml:name"/>[i], sizeof h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="$stateName"/>-><xsl:value-of select="xmml:name"/>[i], 1, file);</xsl:for-each>
      }
    </xsl:for-each>
    currentAgentTypeID++;
  </xsl:for-each>
  chunkSize = (int)(_ftelli64(file) - chunkOffsetPosition) - sizeof(int);
  _fseeki64(file, chunkOffsetPosition, SEEK_SET);
  fwrite(&amp;chunkSize, sizeof(int), 1, file);
  fwrite(&amp;noOfAgents, sizeof(int), 1, file);
  
   _fseeki64(file, iterationOffsetLocation + (sizeof(__int64) * iteration_number), SEEK_SET);
  fwrite(&amp;chunkOffsetPosition, sizeof(__int64), 1, file);
  
  _fseeki64(file, 0, SEEK_END);
  }
  
  

  void saveIterationDataToCopyOnly(<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, int h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>) {
  //Device to host memory transfer
  <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
    cudaMemcpy( h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, sizeof(xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list), cudaMemcpyDeviceToHost);</xsl:for-each>
}

void saveIterationDataToBinary(char* outputpath, int iteration_number, <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list* d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, int h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count<xsl:if test="position()!=last()">,</xsl:if></xsl:for-each>) {
  //Device to host memory transfer
  <xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent/xmml:states/gpu:state">
    cudaMemcpy( h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, d_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="xmml:name"/>, sizeof(xmachine_memory_<xsl:value-of select="../../xmml:name"/>_list), cudaMemcpyDeviceToHost);</xsl:for-each>

  _fseeki64(file, 0, SEEK_END);
  fwrite(&amp;iteration_number, sizeof(int), 1, file);

  __int64 chunkOffsetPosition = _ftelli64(file);
  int chunkSize = 0;
  fwrite(&amp;chunkSize, sizeof(int), 1, file);
  int noOfAgents = 0;
  fwrite(&amp;noOfAgents, sizeof(int), 1, file);

  int currentAgentTypeID = 0;

<xsl:for-each select="gpu:xmodel/xmml:xagents/gpu:xagent">
  <xsl:for-each select="xmml:states/gpu:state">
    <xsl:variable name="stateName" select="xmml:name"/>//Write each <xsl:value-of select="../../xmml:name"/> agent to binary
  for (int i=0; i&lt;h_xmachine_memory_<xsl:value-of select="../../xmml:name"/>_<xsl:value-of select="xmml:name"/>_count; i++){
    noOfAgents++;
    fwrite(&amp;currentAgentTypeID, sizeof(int), 1, file);
    <xsl:for-each select="../../xmml:memory/gpu:variable">
    fwrite(&amp;h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="$stateName"/>-><xsl:value-of select="xmml:name"/>[i], sizeof h_<xsl:value-of select="../../xmml:name"/>s_<xsl:value-of select="$stateName"/>-><xsl:value-of select="xmml:name"/>[i], 1, file);</xsl:for-each>
  }
  </xsl:for-each>
  currentAgentTypeID++;
  </xsl:for-each>
  chunkSize = (int)(_ftelli64(file) - chunkOffsetPosition) - sizeof(int);
  _fseeki64(file, chunkOffsetPosition, SEEK_SET);
  fwrite(&amp;chunkSize, sizeof(int), 1, file);
  fwrite(&amp;noOfAgents, sizeof(int), 1, file);

  EnsureFileSize(outputpath);
  /* Close the file */
  //fclose(file);
  }
  
  #pragma endregion

/* END FLAME GPU EXTENSIONS */
//#endif
</xsl:template>
</xsl:stylesheet>