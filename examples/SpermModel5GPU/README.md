Customisations needed on Visual Studio Project

Configuration Properties > VC++ Directories > Include Directories
C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.0\common\inc
D:\Development\GPUFlame\Common
D:\Development\GPUFlame\OviductCollisionDetectionV1
D:\Development\External Components\cudpp\include

Library Directories
C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.0\common\lib\x64
D:\Development\External Components\cudpp-2.3\lib (for CUDPP compiled as static libraries)

Configuration Properties > Linker
Input
cudpp64.lib/cudpp64d.lib (for CUDPP compiled as static libraries)

For CMD compile, unoptimized command
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin\nvcc.exe" -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\" --use-local-env --cl-version 2013 -I..\..\include -I.\src -I.\src\model -I.\src\dynamic -I.\src\visualisation -I"D:\Development\External Components\cudpp\include" -I"D:\Development\External Components\cudpp\ext\moderngpu\include" -I"D:\Development\External Components\cudpp\ext\cub" -I"D:\Development\External Components\cudpp\src\cudpp" -I"D:\Development\External Components\cudpp\src\cudpp\app" -I"D:\Development\External Components\cudpp\src\cudpp\kernel" -I"D:\Development\External Components\cudpp\src\cudpp\cta" -I"D:\Development\External Components\cudpp-2.3\bin" -I"D:\Development\External Components\cudpp-2.3\lib" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include" -I"C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.0\common\inc" -I..\..\include -I.\src -I.\src\model -I.\src\dynamic -I.\src\visualisation -I"D:\Development\External Components\cudpp\include" -I"D:\Development\External Components\cudpp\ext\moderngpu\include" -I"D:\Development\External Components\cudpp\ext\cub" -I"D:\Development\External Components\cudpp\..\..\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include" -I"D:\Development\External Components\cudpp-2.3\bin" -I"D:\Development\External Components\cudpp-2.3\lib" -I"D:\Development\GPUFlame\OviductCollisionDetectionV1" -I"D:\Development\GPUFlame\Common" -G -lineinfo  --keep-dir x64\Debug_Console -maxrregcount=0  --machine 64 --compile -cudart static  -g   -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /Od /Zi /RTC1 /MTd  " -o x64\Debug_Console\simulation.cu.obj "D:\Development\FLAMEGPU\examples\SpermModel5GPU\src\dynamic\simulation.cu"

For CMD compile, optimized command (no replicated directories)
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin\nvcc.exe" -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_50,code=\"sm_50,compute_50\" --use-local-env --cl-version 2013 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\x86_amd64"  -I..\..\include -I.\src -I.\src\model -I.\src\dynamic -I.\src\visualisation -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include" -I..\..\include -I.\src -I.\src\model -I.\src\dynamic -I.\src\visualisation -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include" --source-in-ptx  -lineinfo  --keep-dir x64\Release_Console -maxrregcount=32  --machine 64 --compile -cudart static     -DWIN32 -DWIN64 -DNDEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /O2 /Zi  /MT  " -o x64\Release_Console\simulation.cu.obj "D:\Development\FLAMEGPU\examples\SpermModel5GPU\src\dynamic\simulation.cu"

Some projects WERE using CUDA 3.1
CUDA_INC_PATH	C:\CUDA\v3.1\include
NVSDKCOMPUTE_ROOT	C:\CUDA\SDK3.1

nvcc --cubin $(ProjectDir)/Impl/CUDAVector/CU/euler_solver_2d.cu -I"$(CUDA_INC_PATH)"  -I"$(NVSDKCOMPUTE_ROOT)/C/common/inc"  -o $(ProjectDir)/Impl/CUDAVector/CUBIN/euler_solver_2d.cubin

They have been updated to CUDA 7
CUDA_INC_PATH_V7_0	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include
NVCUDASAMPLES7_0_ROOT	C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.0

nvcc --cubin $(ProjectDir)/Impl/CUDAVector/CU/euler_solver_2d.cu -I"$(CUDA_INC_PATH_V7_0)"  -I"$(NVCUDASAMPLES7_0_ROOT)/common/inc"  -o $(ProjectDir)/Impl/CUDAVector/CUBIN/euler_solver_2d.cubin

Some projects execute a previous step to generate CUDA BINARY intermediate files. To properly define the environment "vcvarsall.bat" must be called
call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x86
nvcc --cubin "$(ProjectDir)CU\euler_solver_2d.cu" -I "D:\CUDA\SDK\common\inc" -L "D:\CUDA\SDK\common\lib" -o "$(ProjectDir)CUBIN\euler_solver_2d.cubin"
CUDA COMPILING FROM COMMAND LINE:
nvcc --cubin "D:\ForCarlos\PhDCode\CudaHelperComponents\CudaHelperComponents\CU\CommonFunctions.cu" -I"%CUDA_INC_PATH_V6_5%" -I"%NVCUDASAMPLES6_5_ROOT%\common\inc" -o "D:\ForCarlos\PhDCode\CudaHelperComponents\CudaHelperComponents\CUBIN\CommonFunctions.cubin"
nvcc --cubin "D:\ForCarlos\PhDCode\CudaHelperComponents\CudaHelperComponents\CU\CommonFunctions.cu" -I"%CUDA_INC_PATH_V6_5%" -I"%NVCUDASAMPLES6_5_ROOT%\common\inc" -o "D:\ForCarlos\PhDCode\CudaHelperComponents\CudaHelperComponents\CUBIN\CommonFunctions.cubin" -arch=sm_20 -gencode=arch=compute_20,code=sm_20 --machine 32
nvcc --cubin "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDAVector\CU\euler_solver_2d.cu" -I"%CUDA_INC_PATH_V6_5%"  -I"%NVCUDASAMPLES6_5_ROOT%\common\inc"  -o "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDAVector\CUBIN\euler_solver_2d.cubin" -arch=sm_20 -gencode=arch=compute_20,code=sm_20 --machine 32
nvcc --cubin "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDAVector\CU\euler_solver_2d.cu" -I"%CUDA_INC_PATH_V6_5%"  -I"%NVCUDASAMPLES6_5_ROOT%\common\inc" -o "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDAVector\CUBIN\rk4_solver_2d.cubin" -arch=sm_20 -gencode=arch=compute_20,code=sm_20 --machine 32
nvcc --cubin "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDAVector\CU\euler_solver_3d.cu" -I"%CUDA_INC_PATH_V6_5%"  -I"%NVCUDASAMPLES6_5_ROOT%\common\inc" -o "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDAVector\CUBIN\euler_solver_3d.cubin" -arch=sm_20 -gencode=arch=compute_20,code=sm_20 --machine 32
nvcc --cubin "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDAVector\CU\rk4_solver_3d.cu" -I"%CUDA_INC_PATH_V6_5%"  -I"%NVCUDASAMPLES6_5_ROOT%\common\inc" -o "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDAVector\CUBIN\rk4_solver_3d.cubin" -arch=sm_20 -gencode=arch=compute_20,code=sm_20 --machine 32
nvcc --cubin "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDASerial\CU\euler_solver_2d.cu" -I"%CUDA_INC_PATH_V6_5%"  -I"%NVCUDASAMPLES6_5_ROOT%\common\inc" -o "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDASerial\CUBIN\euler_solver_2d.cubin" -arch=sm_20 -gencode=arch=compute_20,code=sm_20 --machine 32
nvcc --cubin "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDASerial\CU\rk4_solver_2d.cu" -I"%CUDA_INC_PATH_V6_5%"  -I"%NVCUDASAMPLES6_5_ROOT%\common\inc" -o "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDASerial\CUBIN\rk4_solver_2d.cubin" -arch=sm_20 -gencode=arch=compute_20,code=sm_20 --machine 32
nvcc --cubin "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDASerial\CU\euler_solver_3d.cu" -I"%CUDA_INC_PATH_V6_5%"  -I"%NVCUDASAMPLES6_5_ROOT%\common\inc" -o "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDASerial\CUBIN\euler_solver_3d.cubin" -arch=sm_20 -gencode=arch=compute_20,code=sm_20 --machine 32
nvcc --cubin "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDASerial\CU\rk4_solver_3d.cu" -I"%CUDA_INC_PATH_V6_5%"  -I"%NVCUDASAMPLES6_5_ROOT%\common\inc" -o "D:\ForCarlos\PhDCode\ParticlePhysicsLibrary\Impl\CUDASerial\CUBIN\rk4_solver_3d.cubin" -arch=sm_20 -gencode=arch=compute_20,code=sm_20 --machine 32

ParticlePhysicsLibrary CUBINS
call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x86
nvcc --cubin "$(ProjectDir)\Impl\CUDAVector\CU\euler_solver_2d.cu" -I"$(CUDA_INC_PATH_V8_0)"  -I"$(NVCUDASAMPLES8_0_ROOT)\common\inc"  -o "$(ProjectDir)\Impl\CUDAVector\CUBIN\euler_solver_2d.cubin" -arch=sm_61 -gencode=arch=compute_61,code=\"sm_61,compute_61\"
nvcc --cubin "$(ProjectDir)\Impl\CUDAVector\CU\euler_solver_2d.cu" -I"$(CUDA_INC_PATH_V8_0)"  -I"$(NVCUDASAMPLES8_0_ROOT)\common\inc" -o "$(ProjectDir)\Impl\CUDAVector\CUBIN\rk4_solver_2d.cubin" -arch=sm_61 -gencode=arch=compute_61,code=\"sm_61,compute_61\"
nvcc --cubin "$(ProjectDir)\Impl\CUDAVector\CU\euler_solver_3d.cu" -I"$(CUDA_INC_PATH_V8_0)"  -I"$(NVCUDASAMPLES8_0_ROOT)\common\inc" -o "$(ProjectDir)\Impl\CUDAVector\CUBIN\euler_solver_3d.cubin" -arch=sm_61 -gencode=arch=compute_61,code=\"sm_61,compute_61\"
nvcc --cubin "$(ProjectDir)\Impl\CUDAVector\CU\rk4_solver_3d.cu" -I"$(CUDA_INC_PATH_V8_0)"  -I"$(NVCUDASAMPLES8_0_ROOT)\common\inc" -o "$(ProjectDir)\Impl\CUDAVector\CUBIN\rk4_solver_3d.cubin" -arch=sm_61 -gencode=arch=compute_61,code=\"sm_61,compute_61\"
nvcc --cubin "$(ProjectDir)\Impl\CUDASerial\CU\euler_solver_2d.cu" -I"$(CUDA_INC_PATH_V8_0)"  -I"$(NVCUDASAMPLES8_0_ROOT)\common\inc" -o "$(ProjectDir)\Impl\CUDASerial\CUBIN\euler_solver_2d.cubin" -arch=sm_61 -gencode=arch=compute_61,code=\"sm_61,compute_61\"
nvcc --cubin "$(ProjectDir)\Impl\CUDASerial\CU\rk4_solver_2d.cu" -I"$(CUDA_INC_PATH_V8_0)"  -I"$(NVCUDASAMPLES8_0_ROOT)\common\inc" -o "$(ProjectDir)\Impl\CUDASerial\CUBIN\rk4_solver_2d.cubin" -arch=sm_61 -gencode=arch=compute_61,code=\"sm_61,compute_61\"
nvcc --cubin "$(ProjectDir)\Impl\CUDASerial\CU\euler_solver_3d.cu" -I"$(CUDA_INC_PATH_V8_0)"  -I"$(NVCUDASAMPLES8_0_ROOT)\common\inc" -o "$(ProjectDir)\Impl\CUDASerial\CUBIN\euler_solver_3d.cubin" -arch=sm_61 -gencode=arch=compute_61,code=\"sm_61,compute_61\"
nvcc --cubin "$(ProjectDir)\Impl\CUDASerial\CU\rk4_solver_3d.cu" -I"$(CUDA_INC_PATH_V8_0)"  -I"$(NVCUDASAMPLES8_0_ROOT)\common\inc" -o "$(ProjectDir)\Impl\CUDASerial\CUBIN\rk4_solver_3d.cubin" -arch=sm_61 -gencode=arch=compute_61,code=\"sm_61,compute_61\"

CUDAHelperComponents CUBINS
call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" x86
nvcc --cubin "$(ProjectDir)CU\CommonFunctions.cu" -I"$(CUDA_INC_PATH_V8_0)"  -I"$(NVCUDASAMPLES8_0_ROOT)\common\inc" -o "$(ProjectDir)CUBIN\CommonFunctions.cubin" -arch=sm_61 -gencode=arch=compute_61,code=\"sm_61,compute_61\"


Some project seem to have been using an older version of the MathsLibrary project. The following changes have been performed as needed.
PARTICLE_Vector_TYPE -> PARTICLE_VECTOR_TYPE
MathsLibrary.Vector_DIMENSION -> VECTOR_DIMENSION

AddScaledVec2 - AddScaledEx
CopyVec2 - CopyEx
DistanceVec2 - Distance
AddVec2 - AddEx
MulVec2 - MulEx
ScalarProductVec2 - DotProductEx
ScaleVec2 - ScaleEx
SetVec2 - Set
SubVec2 - SubEx
LeftHandNormal
LengthVec2 - LengthEx
NormaliseVec2 - Normalise

The TAOFramework 2.1 has bindings for OpenGL 2.1

For FLBAnalyser
Install-Package NHibernate -Version 3.0.0.4000


noOfVertices=388019

noOfTriangles=769698

noOfSharedVertices=2402

No of Render Units=4
renderUnit[0], type=Triangle-Strip, recordCount=654486 <- noOfUnitTriangles
Number of Strips 1200

renderUnit[1], type=Triangle-Strip, recordCount=2629 <- noOfUnitTriangles
Number of Strips 2385

renderUnit[2], type=Triangle-Strip, recordCount=7391 <- noOfUnitTriangles
Number of Strips 7127

renderUnit[3], type=Triangle-Strip, recordCount=21611 <- noOfUnitTriangles
Number of Strips 21338

Sum of unit triangles : 686117
Sum of Strips: 32050