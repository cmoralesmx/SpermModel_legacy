
/*Definition for the 3D model*/

#ifndef _MODEL_3D_CODE_H
#define _MODEL_3D_CODE_H 1

#define FULL_MODEL 

#define NO_OF_SLICES 1201
#define NO_OF_SLICES_MINUS_ONE 1200
#define NO_OF_SEGMENTS NO_OF_SLICES_MINUS_ONE
#define NO_OF_SEGMENTS_MINUS_ONE 1199


#ifdef FULL_MODEL

#define INCLUDE_DATA_FILE "D:\\_WIP_Data\\Final_Models\\Mouse\\Corrected_Scale\\Code_Models\\OviductA_Curve1\\"

//#define NO_OF_VECTICES 326000
//#define NO_OF_TRIANGLES 652086
#define NO_OF_TRIANGLES 648930


//#define NO_OF_VECTICES 338156
//#define NO_OF_TRIANGLES 676406
#else

#define INCLUDE_DATA_FILE "D:\\_WIP_Data\\Final_Models\\Mouse\\Corrected_Scale\\Code_Models\\OviductNF_Curve1\\"

//#define NO_OF_VECTICES 145149
//#define NO_OF_TRIANGLES 288294
#define NO_OF_TRIANGLES 287644

//#define NO_OF_VECTICES 142125
//#define NO_OF_TRIANGLES 282016

#endif

//mid point and radius of spheres surrounding each triangle
static float4 TRIANGLE_BOUNDING_SPHERE_H[NO_OF_TRIANGLES];

//Pre-Calculated collision detection information for each triangle - contains triangle vertices relative to plane
static float4 TRIANGLE_PLANE_H[NO_OF_TRIANGLES * 4];

//The midpoint of each segment (defined by two adjacent slices)
static float4 SEGMENT_MIDPOINT_VECTOR_H[NO_OF_SEGMENTS];

//The cumulative index of each triangle within a tri-strip relative to each slice
static int SLICE_TRISTRIP_INDEX_H[NO_OF_SLICES];

//The definition of a plane for each slice
static float4 SLICE_PLANE_H[NO_OF_SLICES];

#endif

