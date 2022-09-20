#pragma once
#include "ui_sym_trap.h"
#include "cost_functions/CostFunctionManager.h"
#include "core/data_structures_6D.h"

#include <qdialog.h>
#include <QFile>
#include <QTextStream>
#include <QtWidgets/qfiledialog.h>
#include <QMessageBox>
//#include "optimizer_manager.h"


#include <QVTKWidget.h>

#include <vtkImageData.h>
#include <vtkTextProperty.h>
#include <vtkAxis.h>
#include <vtkAxisActor2D.h>
#include <vtkChartXY.h>
#include <vtkContextScene.h>
#include <vtkContextView.h>
#include <vtkCubeAxesActor2D.h>
#include <vtkDataSetMapper.h>
#include <vtkFloatArray.h>
#include <vtkInteractorStyleTrackball.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPlotPoints.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSimplePointsReader.h>
#include <vtkSmartPointer.h>
#include <vtkTable.h>
#include <vtkWarpScalar.h>

#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>


class sym_trap :public QDialog
{
	Q_OBJECT

public:
	sym_trap(QWidget* parent = 0, Qt::WindowFlags flags = 0);
	~sym_trap();
	//OptimizerManager* sym_trap_optimizer = new OptimizerManager();


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
	void create_vector_of_poses(std::vector<Point6D>& pose_list, Point6D pose);

	template<typename T>
	std::vector<double> static linspace(T start_in, T end_in, int num_in);

	int getIterCount();

	//void set_pose(Point6D desired_pose);
	Ui::symTrap ui;

public Q_SLOTS:
	double onCostFuncAtPoint(double result);
	void graphResults();
	void graphResults2D();
	void setIterCount(int n);
	void saveData();
	void loadData();
	void savePlot();

private:
	std::vector<Point6D> search_space;
	QVTKWidget* plot_widget;
	int iter_count;


signals:
	void Done();
};