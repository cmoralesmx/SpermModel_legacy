/*Collision detection on CUDA - CU implementation*/

#ifndef _COLLISION_DETECTION_CUDA_H
#define _COLLISION_DETECTION_CUDA_H 1

/***
 * Performs collisions detection with the environment for each agent. Uses sphere sweeping collision detection algorithms.
 * 
 */

#include "helper_math.h"
//#include "sphere_collisionCUDA.h"
#include "sphere_collision_comprehensiveCUDA.h"
#include "ModelTextureAccessCUDA.h"


__device__ CollisionResult resolve_environment_collisions(const int currentSegment, const float3 &oldPosition, const float3 &direction, float distance_to_move, const float radius);


/*
* Identifies the nearest tri-strips and performs collision detection on all within range of old and new positions (+/- 1 either side)
* If a collision occurs, the new position is modified to be INTERSECTION_BUFFER distance behind the intersection point.
* Returns EXIT_COLLISION_DETECTION if the distance between oldPosition and newPosition becomes less than INTERSECTION_BUFFER.
* Returns COLLISION_OCCURRED if an intersection has occurred.
* Returns NO_COLLISION otherwise.
*/

#pragma region Slice Calculation Functions

/*
 * Determines the current segment of the oviduct. Sign distance test is performed against all slice planes within range.
 * If the current position is after the previous slice, but before the next slice, then the current segment and distance
 * from segment midpoint is stored. If multiple segments pass the test, the segment with a midpoint closest to agent is chosen.
 */
__device__ int determine_segment_from_position(const int2 &sliceRange, const float3 &position) {
	float plane_distance = (calculateDistanceFromPointToPlane(getSlicePlane(sliceRange.x), position));
	int nearestSegment = sliceRange.x;
	float nearsetSegmentDistance = FLT_MAX;

	for(int i=sliceRange.x;i<sliceRange.y;i++) {
		float4 nextPlane = getSlicePlane(i+1);
		float next_plane_distance = (calculateDistanceFromPointToPlane(nextPlane, position));

		if(plane_distance >= 0) {
			if (next_plane_distance <= 0) {
				float segmentDistance = distance(getSegmentMidpointVector(i), position);
				if (segmentDistance < nearsetSegmentDistance) {
					nearsetSegmentDistance = segmentDistance;
					nearestSegment = i;
				}
			}
			else if (nearsetSegmentDistance == FLT_MAX) {
				nearestSegment = i;
			}
		}

		plane_distance = next_plane_distance;
	}

	//if (nearestSegment == INT_MAX) {
	//	return -1;
	//}
	//else {
		return nearestSegment;
	//}
}


/*
 * Evaluates the slices within the specified range to determine which slice plane the current agent
 * movement path will intersect. This is called twice, once to evaluate forwards and once to evaluate backwards.
 * Code modified from PlaneSweptSphereIntersect [see sphere_collision_comprehensiveCUDA.h for reference]
 */
__device__ int calculate_slice_range_one_way(const int start, const int end, const float3 &startPosition, const float3 &direction, const float distance_to_move, const float radius) {
	int mod = start < end ? 1 : -1;

	int invMod = -mod;
	int slice = start;

	float3 currentPosition = startPosition;
	float currentDistance = distance_to_move;

	while (slice != end) {
		Plane plane = getSlicePlane(slice) * invMod;
		float3 planeNorm = make_float3(plane);
		float3 directionDelta = direction * currentDistance;

		float planeDot = dot(planeNorm, directionDelta);

		float h = calculateDistanceFromPointToPlane(plane, currentPosition);

		if (planeDot == 0) {
			if (abs(h) > radius) {
				return slice;
			}
			else {
				return slice + mod;
			}
		}
		else {
			float outT1 = (radius - h) / planeDot;
			float outT2 = (-radius - h) / planeDot;
			if (outT1 > outT2) {
				float tmp = outT1;
				outT1 = outT2;
				outT2 = tmp;
			}
			if (outT1 > 1.0f || outT2 < 0.0f) {
				return slice;
			}
			else {
				if (outT1 <= 0) {
					return slice;
				}
				else {
					float distanceMoved = outT1 * currentDistance;
					currentPosition += (distanceMoved * direction);
					currentDistance -= distanceMoved;
				}
			}
		}

		slice += mod;
	}
	return slice;
}


/*
 * Calculates the potential range of slices which the current agent movement will intersect. Initially, a 1 segment buffer is established
 * then all previous and all following slices are evaluated until the optimal range is identified.
 */
__device__ int2 calculate_slice_range(const int currentSegment, const float3 &oldPosition, const float3 &direction, const float distance_to_move, const float radius) {

	int minSlice = max(currentSegment - 1, 0);
	int maxSlice = currentSegment < (NO_OF_SEGMENTS_MINUS_ONE - 1) ? currentSegment + 2 : NO_OF_SEGMENTS_MINUS_ONE;

	minSlice = calculate_slice_range_one_way(minSlice, 0, oldPosition, direction, distance_to_move, radius);
	maxSlice = calculate_slice_range_one_way(maxSlice, NO_OF_SEGMENTS_MINUS_ONE, oldPosition, direction, distance_to_move, radius); 

	return make_int2(minSlice >= 0 ? minSlice : 0, maxSlice < NO_OF_SEGMENTS_MINUS_ONE ? maxSlice : NO_OF_SEGMENTS_MINUS_ONE);
}



#pragma endregion


#pragma region End Capping Collision Functions

/*
 * Tests for an intersection with the end cap at the end of the mesh. 
 */
__device__ IntersectionResult resolve_end_cap_collisions(const float3 &oldPosition, const float3 &direction, const float distance_to_move, const float radius, float4 &collisionPlane) {
	float4 plane = -getSlicePlane(NO_OF_SLICES_MINUS_ONE);
	IntersectionResult result = testIntersectionPlaneSphere(plane, oldPosition, direction, distance_to_move, radius);

	if (result.intersectionOccurred == INTERSECTION_OCCURRED) {
		collisionPlane = plane;
		result = testIntersectionPlaneSphere(plane, oldPosition, direction, distance_to_move, radius + RADIUS_BUFFER);
		return make_intersection_result((result.distanceToMove), INTERSECTION_OCCURRED);
	}
	else {
		return make_intersection_result(distance_to_move, NO_INTERSECTION);
	}
}

/*
 * Tests for an intersection with the end cap at the start of the mesh. 
 */
__device__ IntersectionResult resolve_start_cap_collisions(const float3 &oldPosition, const float3 &direction, const float distance_to_move, const float radius, float4 &collisionPlane) {
	float4 plane = getSlicePlane(0);
	
	IntersectionResult result = testIntersectionPlaneSphere(plane, oldPosition, direction, distance_to_move, radius);

	if (result.intersectionOccurred == INTERSECTION_OCCURRED) {
		collisionPlane = plane;
		result = testIntersectionPlaneSphere( plane, oldPosition, direction, distance_to_move, radius + RADIUS_BUFFER);
		return make_intersection_result(result.distanceToMove, INTERSECTION_OCCURRED);
	}
	else {
		return make_intersection_result(distance_to_move, NO_INTERSECTION);
	}

}

#pragma endregion


//Modified from http://www.softsurfer.com/Archive/algorithm_0102/
__device__ float CalculateDistanceFromPointToLine(const float3 &pt, float3 line_pt0, float3 line_pt1, float3 &point_on_line) {
    float3 v = line_pt1 - line_pt0;
    float3 w = pt - line_pt0;

    float c1 = dot(w,v);
	if (c1 <= 0) {
		point_on_line = (line_pt0);
		return distance(pt, point_on_line);
	}
    float c2 = dot(v,v);
	if (c2 <= c1) {
		point_on_line = (line_pt1);
		return distance(pt, point_on_line);
	}
    float b = c1 / c2;

    point_on_line = line_pt0 + (v * b);
    return distance(pt, point_on_line);
}

/*__device__ float CalculateDistanceFromPointToLine(const float3 &pt, const float3 &line_pt0, const float3 &line_pt1, float3 &point_on_line) {
    float3 v = line_pt1 - line_pt0;
    float3 w = pt - line_pt0;

    float c1 = dot(w,v);
    float c2 = dot(v,v);
    float b = c1 / c2;

    point_on_line = line_pt0 + (v * b);

	float3 pt_Pb = pt-point_on_line;
    return length(pt_Pb);
}*/



#pragma region Model Collision Functions

/*

 */
__device__ bool isTriangleCloseEnoughToIntersect(const int index, const float3 &oldPosition, const float radius, const float3 &direction, const float distance_to_move) {
	float4 sphere = getTriangleBoundingSphere(index);

	float total_radius = radius + sphere.w + RADIUS_BUFFER;
	float3 sphere_mid = make_float3(sphere);
	if (distance(oldPosition, sphere_mid) > (total_radius + distance_to_move + INTERSECTION_BUFFER)) {
		return false;
	}

  //  return true;


	float3 pointOnLine;
	return CalculateDistanceFromPointToLine(sphere_mid, oldPosition, oldPosition + (direction * (distance_to_move + INTERSECTION_BUFFER)), pointOnLine) <= total_radius;

}

/*
* Performs collision detection against the tri-strip with the specified index
* If a collision occurs, the new position is modified to be INTERSECTION_BUFFER distance behind the intersection point.
* Returns EXIT_COLLISION_DETECTION if the distance between oldPosition and newPosition becomes less than INTERSECTION_BUFFER.
* Returns COLLISION_OCCURRED if an intersection has occurred.
* Returns NO_COLLISION otherwise.
*/


__device__ IntersectionResult hit_triangle_list(const float3 &oldPosition, const float3 &direction, float distance_to_move, const int start_index, const int end_index, const float radius, float4 &collisionPlane) {
	bool intersected = NO_INTERSECTION;

	TrianglePlane tp;
	IntersectionResult result;

	int start = getSliceTristripIndex(start_index);
	int end = getSliceTristripIndex(end_index);

	float originalDistance = distance_to_move;

	//float4 collisionplane;

	for(int i=start;i<end;i++) {

		if (!isTriangleCloseEnoughToIntersect(i, oldPosition, radius + RADIUS_BUFFER, direction, distance_to_move)) {
			continue;
		}

		tp = getTrianglePlane(i);

		result = testIntersectionTriSphere(tp, oldPosition, radius, direction * distance_to_move);

		if (result.intersectionOccurred == INTERSECTION_OCCURRED && result.distanceToMove >= 0 && result.distanceToMove <= distance_to_move) {

			originalDistance = distance_to_move;
			distance_to_move = result.distanceToMove;
			intersected = INTERSECTION_OCCURRED;
			result = testIntersectionTriSphere(tp, oldPosition, radius + RADIUS_BUFFER, direction * originalDistance);
			collisionPlane = tp.plane;
			if (result.intersectionOccurred == INTERSECTION_OCCURRED && result.distanceToMove >= 0 && result.distanceToMove <= distance_to_move) {
				distance_to_move = result.distanceToMove;
			}
		}
	}

	//collisionPlaneNormal = make_float3(collisionplane);
	return make_intersection_result(distance_to_move, intersected);

}




__device__ IntersectionResult resolve_strip_collisions(const int2 &sliceRange, const float3 &oldPosition, const float3 &direction, float distance_to_move, const float radius, float4 &collisionPlane) {
	bool intersected = NO_INTERSECTION;
	IntersectionResult result;
	
	result = hit_triangle_list(oldPosition, direction, distance_to_move, sliceRange.x, sliceRange.y, radius, collisionPlane);

	if (result.intersectionOccurred == INTERSECTION_OCCURRED) {
		intersected = INTERSECTION_OCCURRED;
		distance_to_move = result.distanceToMove;
	}

	return make_intersection_result(distance_to_move, intersected);
}



#pragma endregion

/*
* Inputs the previous position and potential new position
* If a collision occurs, the new position is modified to be INTERSECTION_BUFFER distance behind the intersection point
*/



__device__ CollisionResult resolve_environment_collisions(const int currentSegment, const float3 &oldPosition, const float3 &direction, float distance_to_move, const float radius) {

	//direction = normalize(direction);
	IntersectionResult result;

	CollisionResult rslt;
	rslt.collisionOccurred = NO_COLLISION;

	float4 collisionPlane;
	
	int newSegment = currentSegment;

	float extDistance = distance_to_move + (INTERSECTION_BUFFER);

	int2 sliceRange = calculate_slice_range(currentSegment, oldPosition, direction, extDistance, radius + RADIUS_BUFFER);
//sliceRange.x = 0;
//sliceRange.y = 10;
	rslt.sliceRange = sliceRange;

	result.intersectionOccurred = NO_INTERSECTION;

	if (sliceRange.x == 0) {
		result = resolve_start_cap_collisions(oldPosition, direction, extDistance, radius, collisionPlane);
	}
	else if (sliceRange.y >= NO_OF_SEGMENTS - 3) {
		result = resolve_end_cap_collisions(oldPosition, direction, extDistance, radius, collisionPlane);
	}

	if (result.intersectionOccurred == INTERSECTION_OCCURRED) {
		extDistance = result.distanceToMove - (INTERSECTION_BUFFER);
		if (extDistance < 0) {
			extDistance = 0;
		}
		distance_to_move = extDistance;
		rslt.collisionOccurred = COLLISION_OCCURRED;
		rslt.collisionPlaneNormal = make_float3(collisionPlane);
	}

	if (distance_to_move == 0) {
		rslt.distanceToMove = 0;
		rslt.newSegmentIndex = currentSegment;
		
		return rslt;
	}

	

	extDistance = distance_to_move + INTERSECTION_BUFFER;
	result = resolve_strip_collisions(sliceRange, oldPosition, direction, extDistance, radius, collisionPlane);

	if (result.intersectionOccurred == INTERSECTION_OCCURRED  && result.distanceToMove < extDistance ) {
		extDistance = result.distanceToMove - INTERSECTION_BUFFER;
		if (extDistance < 0) {
			extDistance = 0;
		}

		if (extDistance > 0) {
			newSegment = determine_segment_from_position(sliceRange, oldPosition + (direction * extDistance));
		}

		distance_to_move = extDistance;
		rslt.collisionOccurred = COLLISION_OCCURRED;
		rslt.collisionPlaneNormal = (make_float3(collisionPlane));
	}
	else {
		newSegment = determine_segment_from_position(sliceRange, oldPosition + (direction * distance_to_move));
		//rslt.collisionOccurred = NO_COLLISION;
	}

	
	rslt.distanceToMove = distance_to_move;
	rslt.newSegmentIndex = newSegment;
	
	return rslt;
}

#endif
