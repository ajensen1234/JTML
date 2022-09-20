#ifndef DATA_STRUCTURES_6D_H
#define DATA_STRUCTURES_6D_H

/*Standard*/
#include <algorithm>
#include "render_engine.cuh"

/*Header for Data Storage Class of DIRECT algorithm (basically a linked list)*/

/*Enum Structure for Directions*/
enum Direction {
	X_DIRECTION, Y_DIRECTION, Z_DIRECTION,
	XA_DIRECTION, YA_DIRECTION, ZA_DIRECTION
};

/*Point6D to store Pose Information*/
struct Point6D
{
	Point6D(double xval, double yval, double zval, double xaval, double yaval, double zaval);
	Point6D();
	Point6D(gpu_cost_function::Pose p);

	double x; double y; double z; double xa; double ya; double za;

	double GetDistanceFrom(Point6D otherPoint);

	Direction GetLargestDirection();

	double GetDirection(Direction direction);

	void UpdateDirection(Direction direction, double updated_value);
};

/*Storage Class (Linked List of HyperMatrices/Columns) for DIRECT optimization algorithm*/
struct HyperBox6D //Stores HyperCube Info
{
	HyperBox6D(double value, Point6D center, Point6D sides);
	HyperBox6D();

	double value_;
	double size_;

	Point6D sides_;
	Point6D center_;

	void SetSides(Point6D new_sides);
	Point6D GetSides();

	bool containsPoint(Point6D point);

	Point6D GetCenter();
	void SetCenter(Point6D new_center);

	/*Divide a Side in Three*/
	void TrisectSide(Direction trisect_side);

	void PrintCenter();

};

#endif /*DATA_STRUCTURES_6D_H*/