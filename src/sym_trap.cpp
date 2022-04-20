#include "sym_trap.h"


sym_trap::sym_trap(QWidget* parent, Qt::WindowFlags flags) : QDialog(parent, flags)

{
	ui.setupUi(this);

	QObject::connect(this, SIGNAL(Done()), this, SLOT(close()));

	QFontMetrics font_metrics(this->font());

	this->setStyleSheet(this->styleSheet() += "QGroupBox { margin-top: " + QString::number(font_metrics.height() / 2) + "px; }");
	int group_box_to_top_button_y = font_metrics.height() / 2;

	//when button is clicked, call gather_dataset()
	QObject::connect(ui.Plot_3D, SIGNAL(clicked()), this, SLOT(graphResults()));
	QObject::connect(ui.Plot_2D, SIGNAL(clicked()), this, SLOT(graphResults2D()));
	QObject::connect(ui.iterBox, SIGNAL(valueChanged(int)), this, SLOT(setIterCount(int)));
	
	QObject::connect(ui.save_data, SIGNAL(clicked()), this, SLOT(saveData()));
	QObject::connect(ui.load_data, SIGNAL(clicked()), this, SLOT(loadData()));
	QObject::connect(ui.save_plot, SIGNAL(clicked()), this, SLOT(savePlot()));

	this->iter_count = ui.iterBox->value();

	this->plot_widget = nullptr;
}

sym_trap::~sym_trap()
{
	delete this->plot_widget;
}


Point6D sym_trap::compute_mirror_pose(Point6D pose) {
	// The main function of this equation is to find the mirror pose of a specific projection geometry
	// This is important when it comes to determining the different symmetry traps that might be present

	// To start, we create the transformation matrix based on the current 6D pose (3-1-2 rotation order)
	float rad2deg = 180.0 / 3.1415928;
	//float transform[4][4]; // blank matrix that will get populated
	float Rot[3][3]; // blank matrix that contains the rotation matrix of the above transformation matrix

	// create_312_transform(transform, pose);
	rotation_matrix(Rot, pose);

	// determine the location of the viewing vector, this is also the location of the center of mass

	float viewing[3];
	viewing[0] = pose.x; viewing[1] = pose.y; viewing[2] = pose.z;


	// pull out the z-axis of the pose, determined from the transformation matrix
	float z_ax[3];
	z_ax[0] = Rot[0][2]; z_ax[1] = Rot[1][2]; z_ax[2] = Rot[2][2];
	std::cout << "[" << z_ax[0] << " " << z_ax[1] << " " << z_ax[2] << "]" << std::endl;
	// normalize viewing vector, take negative to point from object -> camera
	float view_normed[3];
	float view_mag = sqrt(pow(viewing[0], 2) + pow(viewing[1], 2) + pow(viewing[2], 2));
	std::cout << "View normed : \n [";
	for (int i = 0; i < 3; ++i) {
		view_normed[i] = -viewing[i] / view_mag;
		std::cout << view_normed[i] << " ";
	}
	std::cout << "]" << std::endl;
	// Next, take the cross product of the two vectors (z-ax and normed) to get the axis of rotation.
	// call this axis M (Crane and Duffy reference)
	float M_temp[3];
	sym_trap::cross_product(M_temp, z_ax, view_normed);

	//need to normalize M
	float M_mag = sqrt(pow(M_temp[0], 2) + pow(M_temp[1], 2) + pow(M_temp[2], 2));
	float M[3];
	std::cout << "View M : \n [";
	for (int i = 0; i < 3; i++) {
		M[i] = M_temp[i] / M_mag;
		std::cout << M[i] << " ";
	}
	std::cout << "]" << std::endl;

	// Take the dot product between the two to determine the angle between them
	// we will keep this in radians for now 

	float temp_dot = 0;
	sym_trap::dot_product(temp_dot, z_ax, view_normed);

	std::cout << "temp_dot" << temp_dot << std::endl;
	float angle_between = acos(temp_dot);
	std::cout << "Angle Between: " << angle_between << std::endl;

	float desired_rotation = 2 * angle_between;

	std::cout << desired_rotation << std::endl;

	// Now we need to compute the equivalent axis-angle rotation matrix

	float pose2dual[3][3];
	equivalent_axis_angle_rotation(pose2dual, M, desired_rotation);


	// declare some varialbes that will be the final pose.



	// calculate the rotation matrix of the dual pose in space using properties of transformation matrices
	float Rot_dual[3][3];
	matmult3(Rot_dual, pose2dual, Rot);

	// write a for loop w "rotation_increment" instead of "pose2dual"
	std::cout << "X-Axis: \n";
	for (int i = 0; i < 3; i++) {
		std::cout << Rot[i][0] << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Y-Axis: \n";
	for (int i = 0; i < 3; i++) {
		std::cout << Rot[i][1] << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Z-Axis: \n";
	for (int i = 0; i < 3; i++) {
		std::cout << Rot[i][2] << std::endl;
	}
	std::cout << std::endl;

	// extract out each of the relevent values from the final rotation matrix
	float xr_dual;
	float zr_dual;
	float yr_dual;
	getRotations312(xr_dual, yr_dual, zr_dual, Rot_dual);

	// convert to degrees

	float xr_dual_deg = xr_dual * rad2deg;
	float yr_dual_deg = yr_dual * rad2deg;
	float zr_dual_deg = zr_dual * rad2deg;

	Point6D final_pose(pose.x, pose.y, pose.z, xr_dual_deg, yr_dual_deg, zr_dual_deg);
	return final_pose;

}


void sym_trap::create_312_transform(float transform[4][4], Point6D pose)
{
	float degtorad = 3.1415928 / 180.0;
	float zr_rad = pose.za * degtorad;
	float xr_rad = pose.xa * degtorad;
	float yr_rad = pose.ya * degtorad;

	float cx = cos(xr_rad);
	float cy = cos(yr_rad);
	float cz = cos(zr_rad);
	float sx = sin(xr_rad);
	float sy = sin(yr_rad);
	float sz = sin(zr_rad);

	transform[0][0] = cy * cz - sx * sy * sz;
	transform[0][1] = -cx * sz;
	transform[0][2] = cy * sx * sz + cz * sy;
	transform[0][3] = pose.x;

	transform[1][0] = cy * sz + cz * sx * sy;
	transform[1][1] = cx * cz;
	transform[1][2] = -cy * cz * sx + sy * sz;
	transform[1][3] = pose.y;

	transform[2][0] = -cx * sy;
	transform[2][1] = sx;
	transform[2][2] = cx * cy;
	transform[2][3] = pose.z;

	transform[3][0] = transform[3][1] = transform[3][2] = 0.0f;
	transform[3][3] = 1.0f;
}

void sym_trap::invert_transform(float result[4][4], const float tran[4][4])
{
	int     i, j;
	/* Upper left 3x3 of result is transpose of upper left 3x3 of tran. */
	for (i = 0; i < 3; ++i)
		for (j = 0; j < 3; ++j)
			result[i][j] = tran[j][i];
	/* Set the values for the last column of the result */
	result[3][0] = result[3][1] = result[3][2] = 0.0;
	result[3][3] = 1.0;
	/* Initialize the values of the last column of the result. */
	result[0][3] = result[1][3] = result[2][3] = 0.0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			result[i][3] -= result[i][j] * tran[j][3];
		}
	}
}

void sym_trap::matmult4(float ans[4][4], float matrix1[4][4], float matrix2[4][4])
{
	int   i, j, k;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++)
			ans[i][j] = 0.0;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 4; j++)
			for (k = 0; k < 4; k++)
				ans[i][j] += matrix1[i][k] * matrix2[k][j];
}

void sym_trap::matmult3(float ans[3][3], const float matrix1[3][3], const float matrix2[3][3]) {
	int   i, j, k;
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			ans[i][j] = 0.0;
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
			for (k = 0; k < 3; k++)
				ans[i][j] += matrix1[i][k] * matrix2[k][j];
}

void sym_trap::dot_product(float& result, const float vector1[3], const float vector2[3]) {

	for (int i = 0; i < 3; i++) {
		result += (vector1[i] * vector2[i]);
	}
}

void sym_trap::cross_product(float CP[3], const float v1[3], const float v2[3]) {
	CP[0] = v1[1] * v2[2] - v1[2] * v2[1];
	CP[1] = -(v1[0] * v2[2] - v1[2] * v2[0]);
	CP[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

void sym_trap::equivalent_axis_angle_rotation(float rot[3][3], const float m[3], const float angle) {
	float c = cos(angle);
	float s = sin(angle);
	float v = 1.0 - c;

	rot[0][0] = m[0] * m[0] * v + c;
	rot[0][1] = m[0] * m[1] * v - m[2] * s;
	rot[0][2] = m[0] * m[2] * v + m[1] * s;

	rot[1][0] = m[0] * m[1] * v + m[2] * s;
	rot[1][1] = m[1] * m[1] * v + c;
	rot[1][2] = m[1] * m[2] * v - m[0] * s;

	rot[2][0] = m[0] * m[2] * v - m[1] * s;
	rot[2][1] = m[1] * m[2] * v + m[0] * s;
	rot[2][2] = m[2] * m[2] * v + c;

}

void sym_trap::rotation_matrix(float R[3][3], Point6D pose) {

	float degtorad = 3.1415928 / 180.0;
	float zr_rad = pose.za * degtorad;
	float xr_rad = pose.xa * degtorad;
	float yr_rad = pose.ya * degtorad;

	float cx = cos(xr_rad);
	float cy = cos(yr_rad);
	float cz = cos(zr_rad);
	float sx = sin(xr_rad);
	float sy = sin(yr_rad);
	float sz = sin(zr_rad);

	R[0][0] = cy * cz - sx * sy * sz;
	R[0][1] = -cx * sz;
	R[0][2] = cy * sx * sz + cz * sy;

	R[1][0] = cy * sz + cz * sx * sy;
	R[1][1] = cx * cz;
	R[1][2] = -cy * cz * sx + sy * sz;

	R[2][0] = -cx * sy;
	R[2][1] = sx;
	R[2][2] = cx * cy;

}


void sym_trap::getRotations312(float& xr, float& yr, float& zr, const float Rot[3][3]) {

	float sx = Rot[2][1];
	xr = asin(sx);
	//std::cout << xr << std::endl;;

	if (sx == 0 || sx == 1) {
		zr = 0;
		yr = atan2(Rot[0][2], Rot[0][0]);
	}
	else {
		zr = atan2(-Rot[0][1], Rot[1][1]);
		yr = atan2(-Rot[2][0], Rot[2][2]);
	}

}

void sym_trap::copy_matrix_by_value(float(&new_matrix)[3][3], const float(&old_matrix)[3][3]) {
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			new_matrix[i][j] = old_matrix[i][j];
		}
	}
}


void sym_trap::create_vector_of_poses(std::vector<Point6D>& pose_list, Point6D pose) {
	// convert curr_pose into a Point6D
	//Point6D pose = Point6D(curr_pose.x_location_, curr_pose.y_location_, curr_pose.z_location_, curr_pose.x_angle_, curr_pose.y_angle_, curr_pose.z_angle_);

	// loop and push values into pose list, to be used in optimizer manager
	cout << "Running Gather Dataset:" << endl;
	printf("Starting Pose: x %f, y %f, z %f, xa %f, ya %f, za %f\n", pose.x, pose.y, pose.z, pose.xa, pose.ya, pose.za);

	// number of intermediary poses
	int numPoses = getIterCount(); // read this in from UI in the future
	cout << "Using " << numPoses << " intermediary poses" << endl;

	// The main function of this equation is to find the mirror pose of a specific projection geometry
	// This is important when it comes to determining the different symmetry traps that might be present

	// To start, we create the transformation matrix based on the current 6D pose (3-1-2 rotation order)
	float rad2deg = 180.0 / 3.1415928;
	//float transform[4][4]; // blank matrix that will get populated
	float Rot[3][3]; // blank matrix that contains the rotation matrix of the above transformation matrix

	// create_312_transform(transform, pose);
	rotation_matrix(Rot, pose);

	// determine the location of the viewing vector, this is also the location of the center of mass

	float viewing[3];
	viewing[0] = pose.x; viewing[1] = pose.y; viewing[2] = pose.z;


	// pull out the z-axis of the pose, determined from the transformation matrix
	float z_ax[3];
	z_ax[0] = Rot[0][2]; z_ax[1] = Rot[1][2]; z_ax[2] = Rot[2][2];
	std::cout << "[" << z_ax[0] << " " << z_ax[1] << " " << z_ax[2] << "]" << std::endl;
	// normalize viewing vector, take negative to point from object -> camera
	float view_normed[3];
	float view_mag = sqrt(pow(viewing[0], 2) + pow(viewing[1], 2) + pow(viewing[2], 2));
	std::cout << "View normed : \n [";
	for (int i = 0; i < 3; ++i) {
		view_normed[i] = -viewing[i] / view_mag;
		std::cout << view_normed[i] << " ";
	}
	std::cout << "]" << std::endl;
	// Next, take the cross product of the two vectors (z-ax and normed) to get the axis of rotation.
	// call this axis M (Crane and Duffy reference)
	float M_temp[3];
	sym_trap::cross_product(M_temp, z_ax, view_normed);

	//need to normalize M
	float M_mag = sqrt(pow(M_temp[0], 2) + pow(M_temp[1], 2) + pow(M_temp[2], 2));
	float M[3];
	std::cout << "View M : \n [";
	for (int i = 0; i < 3; i++) {
		M[i] = M_temp[i] / M_mag;
		std::cout << M[i] << " ";
	}
	std::cout << "]" << std::endl;

	// Take the dot product between the two to determine the angle between them
	// we will keep this in radians for now 

	float temp_dot = 0;
	sym_trap::dot_product(temp_dot, z_ax, view_normed);

	std::cout << "temp_dot: " << temp_dot << std::endl;
	float angle_between = acos(temp_dot);
	std::cout << "Angle Between: " << angle_between << std::endl;

	float desired_rotation = 2 * angle_between;

	std::cout << "Desired Rotation: " << desired_rotation << std::endl;

	// Prepare linspace with a 0 at numPoses #
	std::vector<double> rotToEval = linspace(-1 * desired_rotation, 0.0f, numPoses+1);
	std::vector<double> rotAfter = linspace(0.0f, desired_rotation*2, numPoses*2+1);
	rotToEval.insert(rotToEval.end(), rotAfter.begin() + 1, rotAfter.end()-1);

	double inc = desired_rotation / numPoses;

	// calculate initial cost for pos 0 and put that in our dataset


	// initially call equivalent_axis_angle_rotation for index 1 of rotToEval to get our matrix
	// loop over length of rotToEval and repeatedly multiply by matrix to get new poses
	// each time evaluate cost function

	float rotInc[3][3];
	equivalent_axis_angle_rotation(rotInc, M, inc);

	float resIter[3][3];
	float resDir[3][3];

	matmult3(resIter, rotInc, rotInc);
	equivalent_axis_angle_rotation(resDir, M, 2 * inc);

	// check that both methods are equivalent
	cout << "[Sanity Check] Poses produced by matrix multiplication and equiv axis angle rotation are equivalent: ";
	bool equiv = true;
	std::vector<std::string> failed_elements;

	for (int i = 0; i <= 2; i++) {
		for (int j = 0; j <= 2; j++) {
			//cout << resIter[i][j] << " == " << resDir[i][j] << endl;
			if (resIter[i][j] != resDir[i][j]) {
				equiv = false;
				failed_elements.push_back("i = " + std::to_string(i) + ", j = " + std::to_string(j) + " | " + std::to_string(resIter[i][j]) + " != " + std::to_string(resDir[i][j]));
			}
		}
	}

	// Non-equiv prob due to float approx (Identical up to at least 10^-6)
	cout << ((equiv) ? "TRUE" : "FALSE") << endl;
	for (std::string s : failed_elements) {
		cout << s << endl;
	}

	// initialize r_base to current pose
	float r_base[3][3]; // base pose rotation matrix
	rotation_matrix(r_base, pose);
	
	std::vector<Point6D> pose_list_half;
	pose_list_half.push_back(pose);

	// start r_old at base pose
	float r_old[3][3];
	copy_matrix_by_value(r_old, r_base);
	// for loop to compute all_inc. Index 0 = start, Index n = end (skip?)
	// Start by going forward from pose, then go in reverse
	// Need to floor numposes/3 ?
	for (int i = numPoses + 1; i < rotToEval.size(); i++) {
		float r_new[3][3];
		matmult3(r_new, rotInc, r_old);

		float x_rot, y_rot, z_rot;
		getRotations312(x_rot, y_rot, z_rot, r_new);

		Point6D n = Point6D(pose.x, pose.y, pose.z, rad2deg * x_rot, rad2deg * y_rot, rad2deg * z_rot); //xrot yrot zrot, and however we plug x y and z positions in (Do xyz pos stay the same?)
		printf("Rotation %d (%f): x %f, y %f, z %f, xa %f, ya %f, za %f\n", i, rotToEval[i] * rad2deg, n.x, n.y, n.z, n.xa, n.ya, n.za);

		pose_list_half.push_back(n);
		copy_matrix_by_value(r_old, r_new);
	}
	
	std::vector<Point6D> pose_list_rev;

	// Reset r_old to r_base
	equivalent_axis_angle_rotation(rotInc, M, inc*-1);
	copy_matrix_by_value(r_old, r_base);
	// start reversing through rotToEval
	for (int i = numPoses - 1; i >= 0; i--) {
		float r_new[3][3];
		matmult3(r_new, rotInc, r_old);

		float x_rot, y_rot, z_rot;
		getRotations312(x_rot, y_rot, z_rot, r_new);

		Point6D n = Point6D(pose.x, pose.y, pose.z, rad2deg * x_rot, rad2deg * y_rot, rad2deg * z_rot); //xrot yrot zrot, and however we plug x y and z positions in (Do xyz pos stay the same?)
		printf("Rotation %d (%f): x %f, y %f, z %f, xa %f, ya %f, za %f\n", i, rotToEval[i] * rad2deg, n.x, n.y, n.z, n.xa, n.ya, n.za);

		pose_list_rev.push_back(n);
		copy_matrix_by_value(r_old, r_new);
	}

	// push back pose_list_rev to pose_list in reverse order
	for (int i = pose_list_rev.size() - 1; i >= 0; i--) {
		pose_list.push_back(pose_list_rev[i]);
	}
	
	// push back pose_list_half to pose_list in normal order
	for (int i = 0; i < pose_list_half.size(); i++) {
		pose_list.push_back(pose_list_half[i]);
	}

	return;
}

int sym_trap::getIterCount()
{
	return this->iter_count;
}

void sym_trap::setIterCount(int n)
{
	this->iter_count = n;
}

void sym_trap::saveData()
{
	// save a file in Qt selected by user
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"), "", tr("CSV Files (*.csv)"));
	if (fileName == "") { return; }
	QFile::remove(fileName);
	QFile::copy("Results.csv", fileName);
}

void sym_trap::loadData()
{
	// Load a file from user selected directory
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), "", tr("CSV Files (*.csv)"));
	cout << "Filename: '" << fileName.toStdString() << "'" << endl;
	if (fileName == "") { return; }

	QFile loadedFile(fileName);

	// Copy file to Results.csv

	QFile vtkResults("Results.xyz");
	QFile vtkResults2D("Results2D.xy");

	// Copy file to Results.xyz and Results2D.xyz
	if (loadedFile.open(QIODevice::ReadOnly) && vtkResults.open(QIODevice::WriteOnly) && vtkResults2D.open(QIODevice::WriteOnly)) {
		QTextStream input(&loadedFile);
		QTextStream output(&vtkResults);
		QTextStream output2D(&vtkResults2D);

		QString line;
		
		int count = 0;
		// Get line count
		while (!input.atEnd()) {
			line = input.readLine();
			count++;
		}
		cout << "count: " << count << endl;
		input.seek(0); // reset stream to beginning
		
		// Loop through lines in input
		int i = 0;
		while (!input.atEnd()) {
			line = input.readLine();
			QStringList splitLine = line.split(",");

			// Write to Results.xyz
			output << splitLine[0] << " " << splitLine[1] << " " << splitLine[3] << endl;
			output2D << i++ - count/3 << " " << splitLine[3] << endl;
		}

		loadedFile.close();
		vtkResults.close();
		vtkResults2D.close();
	}
	// Remove existing results.csv before copy
	QFile::remove("Results.csv");
	QFile::copy(fileName, "Results.csv");
	
}

void sym_trap::savePlot()
{
	// Take a screenshot of the plot and save it to a file
	if (plot_widget) {
		QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"), "", tr("PNG Files (*.png)"));
		if (fileName == "") { return; }
		//QPixmap pixmap = QPixmap::grabWidget(plot_widget);
		QFile::remove(fileName);
		
		/*pixmap.save(fileName);*/
		
		vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
		writer->SetFileName(fileName.toStdString().c_str());
		writer->SetInputData(plot_widget->cachedImage());
		writer->Write();
	}
	else {
		QMessageBox::information(this, "Error", "No plot to save");
	}
}

double sym_trap::onCostFuncAtPoint(double result) {
	std::cout << "grabbed result" << std::endl;
	return result;
}

void sym_trap::graphResults() {
	//runs when you click gatherdataset button
	//read Results.csv and graph them

	if (!QFile::exists("Results.xyz")) {
		// Error window
		QMessageBox::information(this, "Error", "No 3D results to graph");
		return;
	}
	
	fstream fname;
	fname.open("Results.csv", ios::in);
	if (!fname.is_open()) { std::cout << "error"; }

	std::vector<std::string> row;
	std::string line, data;

	if (fname.is_open())
	{
		while (std::getline(fname, line))
		{

			std::stringstream str_s(line);

			while (std::getline(str_s, data, ','))
				row.push_back(data);
		}
	}
	fname.close();

	std::vector<double> xrot;
	std::vector<double> yrot;
	std::vector<double> zrot;
	std::vector<double> costs;

	for (int i = 0; i < row.size(); i+=4) {
		xrot.push_back(stod(row.at(i)));
		yrot.push_back(stod(row.at(i+1)));
		zrot.push_back(stod(row.at(i+2)));
		costs.push_back(stod(row.at(i+3)));
	}

	// Create QVTK widget and add it to layout box
	if (!plot_widget) {
		this->plot_widget = new QVTKWidget(this);
		this->ui.verticalLayout->insertWidget(0,plot_widget); // insert widget at first index of layout box
		this->ui.verticalLayout->update();
	} else {
		QVTKWidget* temp_widget = new QVTKWidget(this);
		this->ui.verticalLayout->replaceWidget(plot_widget, temp_widget);
		delete plot_widget;
		this->plot_widget = temp_widget;
		this->ui.verticalLayout->update();
	}
	
	// Read the file
	vtkSmartPointer<vtkSimplePointsReader> reader = vtkSmartPointer<vtkSimplePointsReader>::New();
	reader->SetFileName("Results.xyz");
	reader->Update();

	vtkSmartPointer<vtkPolyData> inputPolyData = vtkSmartPointer<vtkPolyData>::New();
	inputPolyData->CopyStructure(reader->GetOutput());


	// warp plane
	vtkSmartPointer<vtkWarpScalar> warp = vtkSmartPointer<vtkWarpScalar>::New();
	warp->SetInputData(inputPolyData);
	warp->SetScaleFactor(0.0);
	
	// Visualize
	vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
	mapper->SetInputConnection(warp->GetOutputPort());



	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->GetProperty()->SetPointSize(10);
	actor->SetMapper(mapper);

	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
	plot_widget->GetRenderWindow()->AddRenderer(renderer);
	
	
	//vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	//renderWindow->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(plot_widget->GetRenderWindow());


	renderer->AddActor(actor);
	renderer->SetBackground(0, 0, 0);

	plot_widget->GetRenderWindow()->Render();

	vtkSmartPointer<vtkInteractorStyleTrackballCamera> style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
	renderWindowInteractor->SetInteractorStyle(style);

	// add & render CubeAxes
	vtkSmartPointer<vtkCubeAxesActor2D> axes = vtkSmartPointer<vtkCubeAxesActor2D>::New();
	axes->SetInputData(warp->GetOutput());
	axes->SetFontFactor(1);
	axes->SetFlyModeToNone();
	axes->SetCamera(renderer->GetActiveCamera());

	vtkSmartPointer<vtkAxisActor2D> xAxis = axes->GetXAxisActor2D();
	xAxis->SetAdjustLabels(1);


	renderer->AddViewProp(axes);
	renderWindowInteractor->Start();
	
}

void sym_trap::graphResults2D() {
	if (!QFile::exists("Results2D.xy")) {
		// Error window
		QMessageBox::information(this, "Error", "No 2D results to graph");
		return;
	}
	
	// Create QVTK widget and add it to layout 
	if (!plot_widget) {
		this->plot_widget = new QVTKWidget(this);
		this->ui.verticalLayout->insertWidget(0, plot_widget); // insert widget at first index of layout box
		this->ui.verticalLayout->update();
	} else {
		QVTKWidget* temp_widget = new QVTKWidget(this);
		this->ui.verticalLayout->replaceWidget(plot_widget, temp_widget);
		delete plot_widget;
		this->plot_widget = temp_widget;
		this->ui.verticalLayout->update();
	}
	
	// Initialize view
	vtkSmartPointer<vtkNamedColors> colors = vtkSmartPointer<vtkNamedColors>::New();
	vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
	view->GetRenderer()->SetBackground(colors->GetColor3d("SlateGray").GetData());

	// Setup chart
	vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
	view->GetScene()->AddItem(chart);
	//chart->SetShowLegend(true);

	// Set axes labels
	vtkAxis* y = chart->GetAxis(vtkAxis::LEFT);
	y->SetTitle("Cost");
	y->GetTitleProperties()->ItalicOn();

	vtkAxis* x = chart->GetAxis(vtkAxis::BOTTOM);
	x->SetTitle("Pose Deviation Index from Origin Pose");
	x->GetTitleProperties()->ItalicOn();

	chart->SetTitle("Cost Analysis");
	chart->GetTitleProperties()->BoldOn();
	
	//Create table
	vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();
	
	vtkSmartPointer<vtkFloatArray> xAxis = vtkSmartPointer<vtkFloatArray>::New();
	xAxis->SetName("Index");
	table->AddColumn(xAxis);
	
	vtkSmartPointer<vtkFloatArray> yAxis = vtkSmartPointer<vtkFloatArray>::New();
	yAxis->SetName("Cost");
	table->AddColumn(yAxis);
	
	// Add data to table
	QFile file("Results2D.xy");
	if (file.open(QIODevice::ReadOnly)) {
		QTextStream in(&file);
		QString line;
		// Count number of lines in file to set table size
		int totalRows = 0;
		while (!in.atEnd()) {
			line = in.readLine();
			totalRows++;
		}
		table->SetNumberOfRows(totalRows);
	
		// Read file again and fill table
		in.seek(0);
		int row = 0;
		while (!in.atEnd()) {
			line = in.readLine();
			QStringList list = line.split(" ");
			table->SetValue(row, 0, list.at(0).toDouble());
			table->SetValue(row, 1, list.at(1).toDouble());
			row++;
		}
	}
	file.close();

	//Add plot and set properties
	chart->SetInteractive(true);
	chart->ForceAxesToBoundsOn();
	chart->ZoomWithMouseWheelOn();
	
	vtkSmartPointer<vtkPlot> line = chart->AddPlot(vtkChart::LINE);
	vtkSmartPointer<vtkPlot> plot = chart->AddPlot(vtkChart::POINTS);
	
	line->SetInputData(table, 0, 1);
	line->SetColor(200, 200, 200, 175);
	line->SetWidth(1.5);
	
	plot->SetInputData(table, 0, 1);
	plot->SetColor(0, 0, 0, 255);
	plot->SetWidth(1.5);

	// Render
	view->SetInteractor(plot_widget->GetInteractor());
	plot_widget->SetRenderWindow(view->GetRenderWindow());
	plot_widget->GetRenderWindow()->SetMultiSamples(0);
	plot_widget->GetRenderWindow()->Render();
}

template<typename T>
std::vector<double> sym_trap::linspace(T start_in, T end_in, int num_in)
{

	std::vector<double> linspaced;

	double start = static_cast<double>(start_in);
	double end = static_cast<double>(end_in);
	double num = static_cast<double>(num_in);

	if (num == 0) { return linspaced; }
	if (num == 1)
	{
		linspaced.push_back(start);
		return linspaced;
	}

	double delta = (end - start) / (num - 1);

	for (int i = 0; i < num - 1; ++i)
	{
		linspaced.push_back(start + delta * i);
	}
	linspaced.push_back(end); // I want to ensure that start and end
							  // are exactly the same as the input
	return linspaced;
}