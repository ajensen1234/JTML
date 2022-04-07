#pragma once
#include <qdialog.h>

#include "optimizer_manager.h"
#include "optimizer_settings.h"
#include "ui_sym_trap.h"


#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkProperty.h>
#include <vtkCubeAxesActor2D.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkInteractorStyleTrackball.h>
#include <vtkSimplePointsReader.h>
#include <vtkWarpScalar.h>
#include <vtkAxisActor2D.h>

#include <QVTKWidget.h>


#include "CostFunctionManager.h"
#include "data_structures_6D.h"

#include <cmath>
#include <vector>
#include <iostream>


class sym_trap :public QDialog
{
	Q_OBJECT

public:
	sym_trap(QWidget* parent = 0, Qt::WindowFlags flags = 0);
	~sym_trap();

	//Point6D pose;

	Point6D compute_mirror_pose(Point6D point);
	static void matmult4(float ans[4][4], float matrix1[4][4], float matrix2[4][4]);
	static void matmult3(float ans[3][3], const float matrix1[3][3], const float matrix2[3][3]);
	static void invert_transform(float result[4][4], const float tran[4][4]);
	static void equivalent_axis_angle_rotation(float rot[3][3], const float m[3], const float angle);
	static void cross_product(float CP[3], const float v1[3], const float v2[3]);
	static void dot_product(float& result, const float vector1[3], const float vector2[3]);
	static void rotation_matrix(float R[3][3], Point6D pose);
	static void create_312_transform(float transform[4][4], Point6D pose);
	static void getRotations312(float& xr, float& yr, float& zr, const float Rot[3][3]);

	static void copy_matrix_by_value(float(&new_matrix)[3][3], const float(&old_matrix)[3][3]);
	static void create_vector_of_poses(std::vector<Point6D>& pose_list, Point6D pose);

	template<typename T>
	std::vector<double> static linspace(T start_in, T end_in, int num_in);


	//void set_pose(Point6D desired_pose);
	Ui::symTrap ui;

public Q_SLOTS:
	double onCostFuncAtPoint(double result);
	void graphResults();
	//void saveData();
	//void loadData();

private:
	std::vector<Point6D> search_space;
	QVTKWidget* plot_widget;


signals:
	void Done();
};