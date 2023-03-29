#pragma once
//////////////////////////////////////////////////////////////////////
///////													       ///////
///////	   ARCHIVE OF COMMENTED OUT FUNCTIONS FROM MAINSCREEN  ///////
///////		NOT NEEDED FOR NOW, SAVING FOR JUST IN CASE (TM)   ///////
///////														   ///////
//////////////////////////////////////////////////////////////////////

	/* std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNETSeg_BS6_LIMA1024Actual_070519_2_TORCH_SCRIPT.pt"; */

	/*Removed the code below to allow for the user to pick and choose which network they want to load instead of using hard-coded paths*/
	/* std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNET_BS8_LIMA1024TibActual_070619_BS8_LIMA1024TibActual_070619_TORCH_SCRIPT.pt";*/

	// std::shared_ptr<torch::jit::script::Module> model(torch::jit::load(pt_model_location, torch::kCUDA));
	// Commented out above part to resolve error E0289+
	//   std::shared_ptr<torch::jit::Module> model(torch::jit::load(pt_model_location, torch::kCUDA));

	/*object shared_ptr cannot be converted from _T to torch::jit::Module */
	/* typecasting might be a solution */

	/* try
	{
		model = torch::jit::load(pt_model_location, torch::kCUDA);
	} */
	/* catch (const c10::Error& e) {
		QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
	}*/

//torch::Tensor gpu_byte_placeholder(torch::zeros({ 1, 1, input_height, input_width }, torch::device(torch::kCUDA).dtype(torch::kByte)));

//cv::Mat correct_inversion = (255 * black_sil_used) + ((1 - 2 * black_sil_used) * loaded_frames[i].GetOriginalImage());
		//cv::Mat padded;
		//if (correct_inversion.cols > correct_inversion.rows)
		//	padded.create(correct_inversion.cols, correct_inversion.cols, correct_inversion.type());
		//else
		//	padded.create(correct_inversion.rows, correct_inversion.rows, correct_inversion.type());
		//unsigned int padded_width = padded.cols;
		//unsigned int padded_height = padded.rows;
		//padded.setTo(cv::Scalar::all(0));
		//correct_inversion.copyTo(padded(cv::Rect(0, 0, correct_inversion.cols, correct_inversion.rows)));
		//cv::resize(padded, padded, cv::Size(input_width, input_height));
		//cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
		//	input_width * input_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
		//std::vector<torch::jit::IValue> inputs;
		//inputs.push_back(gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({ 2 })); // Must flip first
		//cudaMemcpy(padded.data, (255 * (model->forward(inputs).toTensor() > 0)).to(torch::dtype(torch::kByte)).flip({ 2 }).data_ptr(),
		//	input_width * input_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		//cv::resize(padded, padded, cv::Size(padded_width, padded_height));
		//cv::Mat unpadded = padded(cv::Rect(0, 0, correct_inversion.cols, correct_inversion.rows));

//void MainScreen::on_actionEstimate_Femoral_Implant_s_Algorithm_2_triggered() {
////Must be in Single Selection Mode to Load Pose
//if (ui.multiple_model_radio_button->isChecked()) {
//QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Estimate Kinematics!", QMessageBox::Ok);
//return;
//}

////Must load a model
//if (loaded_models.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load a model!", QMessageBox::Ok);
//return;
//}

////Must have loaded image
//if (loaded_frames.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load images!", QMessageBox::Ok);
//return;
//}

///*Pose Estimate Progress and Label Visible*/
//ui.pose_progress->setValue(5);
//ui.pose_progress->setVisible(1);
//ui.pose_label->setText("Initializing high resolution segmentation...");
//ui.pose_label->setVisible(1);
//ui.qvtk_widget->update();
//qApp->processEvents();


///*Segment*/
//this->on_actionSegment_FemHR_triggered();
//unsigned int input_height = 1024;
//unsigned int input_width = 1024;
//unsigned int orig_height = loaded_frames[0].GetInvertedImage().rows;
//unsigned int orig_width = loaded_frames[0].GetInvertedImage().cols;
//unsigned char* host_image = (unsigned char*)malloc(input_width * input_height * sizeof(unsigned char));
//ui.pose_label->setText("Initializing STL model on GPU...");
//ui.qvtk_widget->update();

///*STL Information*/
//vector<vector<float>> triangle_information;
//QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
//stl_reader_BIG::readAnySTL(QString::fromStdString(loaded_models[selected[0].row()].file_location_), triangle_information);

///*GPU Models for the current Model*/
//gpu_cost_function::GPUModel* gpu_mod = new gpu_cost_function::GPUModel("model", true, orig_height, orig_width, 0, false, // switched cols and rows because the stored image is inverted?
//&(triangle_information[0])[0], &(triangle_information[1])[0], triangle_information[0].size() / 9, calibration_file_.camera_A_principal_); // BACKFACE CULLING APPEARS TO BE GIVING ERRORS

//ui.pose_progress->setValue(55);
//ui.pose_label->setText("Initializing femoral implant pose estimation...");
//ui.qvtk_widget->update();

///*Load JIT Model*/
//std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataLima1024_07192019_HRProcessed_Fem_07232019_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module module(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model = &module;
//if (model == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Load JIT Z Model*/
//std::string pt_model_z_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataLima1024_07192019_HRProcessed_Fem_08012019_Z_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model_z(torch::jit::load(pt_model_z_location, torch::kCUDA));
//torch::jit::Module module_z(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model_z = &module_z;
//if (model_z == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Send Each Segmented Image to GPU Tensor, Predict Orientation, Then Z (From Area), then X,Y.
//After this, convert to non (0,0) centered orientation.
//Finally, update */
//ui.pose_progress->setValue(65);
//ui.pose_label->setText("Estimating femoral implant poses...");
//ui.qvtk_widget->update();
//float* orientation = new float[3];
//float* z_norm = new float[1];
//torch::Tensor gpu_byte_placeholder(torch::zeros({ 1, 1, input_height, input_width }, torch::device(torch::kCUDA).dtype(torch::kByte)));
//for (int i = 0; i < ui.image_list_widget->count(); i++) {

//cv::Mat orig_inverted = loaded_frames[i].GetInvertedImage();
//cv::Mat padded;
//if (orig_inverted.cols > orig_inverted.rows)
//padded.create(orig_inverted.cols, orig_inverted.cols, orig_inverted.type());
//else
//padded.create(orig_inverted.rows, orig_inverted.rows, orig_inverted.type());
//unsigned int padded_width = padded.cols;
//unsigned int padded_height = padded.rows;
//padded.setTo(cv::Scalar::all(0));
//orig_inverted.copyTo(padded(cv::Rect(0, 0, orig_inverted.cols, orig_inverted.rows)));
//cv::resize(padded, padded, cv::Size(input_width, input_height));

//cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
//input_width * input_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
//std::vector<torch::jit::IValue> inputs;
//inputs.push_back(gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({ 2 })); // Must flip first
//cudaMemcpy(orientation, model->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//3 * sizeof(float), cudaMemcpyDeviceToHost);
//cudaMemcpy(z_norm, model_z->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//sizeof(float), cudaMemcpyDeviceToHost);

///*Flip Segment*/
//cv::Mat output_mat_seg = cv::Mat(orig_inverted.rows, orig_inverted.cols, CV_8UC1);
//cv::flip(orig_inverted, output_mat_seg, 0);

///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, -calibration_file_.camera_A_principal_.principal_distance_, orientation[1], orientation[2], orientation[0]));

///*Copy To Mat*/
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

///*OpenCV Image Container/Write Function*/
//cv::Mat projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image); /*Reverse before flip*/
//cv::Mat output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get Scale*/
//double z = -calibration_file_.camera_A_principal_.principal_distance_ * z_norm[0];

///*Reproject*/
///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, z, orientation[1], orientation[2], orientation[0]));
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image);
//output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get X and Y*/
//cv::Mat proj64;
//output_mat.convertTo(proj64, CV_64FC1);
//cv::Mat seg64;
//output_mat_seg.convertTo(seg64, CV_64FC1);
//cv::Point2d x_y_point = cv::phaseCorrelate(proj64, seg64) * (calibration_file_.camera_A_principal_.pixel_pitch_ * z * -1) /
//calibration_file_.camera_A_principal_.principal_distance_;
//double x = x_y_point.x;
//double y = -1 * x_y_point.y;


///*Convert from (0,0) Centered*/
//float za_rad = orientation[0] * pi / 180.0;
//float xa_rad = orientation[1] * pi / 180.0;
//float ya_rad = orientation[2] * pi / 180.0;
//float cz = cos(za_rad);
//float sz = sin(za_rad);
//float cx = cos(xa_rad);
//float sx = sin(xa_rad);
//float cy = cos(ya_rad);
//float sy = sin(ya_rad);
//Matrix_3_3 R_g(
//cz * cy - sz * sx * sy, -1.0 * sz * cx, cz * sy + sz * cy * sx,
//sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * cy * sx,
//-1.0 * cx * sy, sx, cx * cy);
//float theta_x = std::atan(-1.0 * y / z);
//float theta_y = std::asin(-1.0 * x / std::sqrt(x * x + y * y + z * z));
//Matrix_3_3 R_x(
//1, 0, 0,
//0, cos(theta_x), -sin(theta_x),
//0, sin(theta_x), cos(theta_x));
//Matrix_3_3 R_y(
//cos(theta_y), 0, sin(theta_y),
//0, 1, 0,
//-sin(theta_y), 0, cos(theta_y));
//Matrix_3_3 R_orig = calibration_file_.multiplication_mat_mat(R_y, calibration_file_.multiplication_mat_mat(R_x, R_g));
///*Rot Mat To Eul ZXY*/
///*Algorithm To Recover Z - X - Y Euler Angles*/
//float xa, ya, za;
//if (R_orig.A_32_ < 1) {
//if (R_orig.A_32_ > -1) {
//xa = asin(R_orig.A_32_);
//za = atan2(-1 * R_orig.A_12_, R_orig.A_22_);
//ya = atan2(-1 * R_orig.A_31_, R_orig.A_33_);

//}
//else {
//xa = -pi / 2.0;
//za = -1 * atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}
//}
//else {
//xa = pi / 2.0;
//za = atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}

//xa = xa * 180.0 / pi;
//ya = ya * 180.0 / pi;
//za = za * 180.0 / pi;

///*Update Model Pose*/
//model_locations_.SavePose(i, ui.model_list_widget->currentRow(), Point6D(x, y, z, xa, ya, za));

//ui.pose_progress->setValue(65 + 30 * (double)(i + 1) / (double)ui.image_list_widget->count());
//ui.qvtk_widget->update();
//qApp->processEvents();
//}

//ui.pose_progress->setValue(98);
//ui.pose_label->setText("Deleting old models...");
//ui.qvtk_widget->update();

///*Delete GPU Model*/
//delete gpu_mod;

///*Free Array*/
//free(host_image);

///*Free Values*/
//delete orientation;
//delete z_norm;

///*Update Model*/
//Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());
//model_actor_list[selected[0].row()]->SetPosition(loaded_pose.x, loaded_pose.y, loaded_pose.z);
//model_actor_list[selected[0].row()]->SetOrientation(loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
//ui.qvtk_widget->update();

//ui.pose_progress->setValue(100);
//ui.pose_label->setText("Finished!");
//ui.qvtk_widget->update();

///*Pose Estimate Progress and Label Not Visible*/
//ui.pose_progress->setVisible(0);
//ui.pose_label->setVisible(0);

//}

//void MainScreen::on_actionEstimate_Tibial_Implant_s_Alternative_Algorithm_triggered() {
////Must be in Single Selection Mode to Load Pose
//if (ui.multiple_model_radio_button->isChecked()) {
//QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Estimate Kinematics!", QMessageBox::Ok);
//return;
//}

////Must load a model
//if (loaded_models.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load a model!", QMessageBox::Ok);
//return;
//}

////Must have loaded image
//if (loaded_frames.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load images!", QMessageBox::Ok);
//return;
//}

///*Pose Estimate Progress and Label Visible*/
//ui.pose_progress->setValue(5);
//ui.pose_progress->setVisible(1);
//ui.pose_label->setText("Initializing high resolution segmentation...");
//ui.pose_label->setVisible(1);
//ui.qvtk_widget->update();
//qApp->processEvents();


///*Segment*/
//this->on_actionSegment_TibHR_triggered();
//unsigned int input_height = 1024;
//unsigned int input_width = 1024;
//unsigned int orig_height = loaded_frames[0].GetInvertedImage().rows;
//unsigned int orig_width = loaded_frames[0].GetInvertedImage().cols;
//unsigned char* host_image = (unsigned char*)malloc(input_width * input_height * sizeof(unsigned char));
//ui.pose_label->setText("Initializing STL model on GPU...");
//ui.qvtk_widget->update();

///*STL Information*/
//vector<vector<float>> triangle_information;
//QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
//stl_reader_BIG::readAnySTL(QString::fromStdString(loaded_models[selected[0].row()].file_location_), triangle_information);

///*GPU Models for the current Model*/
//gpu_cost_function::GPUModel* gpu_mod = new gpu_cost_function::GPUModel("model", true, orig_height, orig_width, 0, false, // switched cols and rows because the stored image is inverted?
//&(triangle_information[0])[0], &(triangle_information[1])[0], triangle_information[0].size() / 9, calibration_file_.camera_A_principal_); // BACKFACE CULLING APPEARS TO BE GIVING ERRORS

//ui.pose_progress->setValue(55);
//ui.pose_label->setText("Initializing tibial implant pose estimation...");
//ui.qvtk_widget->update();

///*Load JIT Model*/
//std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataLimaTib1024_08012019_HRProcessed_Tib_08022019_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module module(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model = &module;
//if (model == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Load JIT Z Model*/
//std::string pt_model_z_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataLimaTib1024_08012019_HRProcessed_Tib_08052019_Z_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model_z(torch::jit::load(pt_model_z_location, torch::kCUDA));
//torch::jit::Module module_z(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model_z = &module_z;
//if (model_z == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Send Each Segmented Image to GPU Tensor, Predict Orientation, Then Z (From Area), then X,Y.
//After this, convert to non (0,0) centered orientation.
//Finally, update */
//ui.pose_progress->setValue(65);
//ui.pose_label->setText("Estimating tibial implant poses...");
//ui.qvtk_widget->update();
//float* orientation = new float[3];
//float* z_norm = new float[1];
//torch::Tensor gpu_byte_placeholder(torch::zeros({ 1, 1, input_height, input_width }, torch::device(torch::kCUDA).dtype(torch::kByte)));
//for (int i = 0; i < ui.image_list_widget->count(); i++) {

//cv::Mat orig_inverted = loaded_frames[i].GetInvertedImage();
//cv::Mat padded;
//if (orig_inverted.cols > orig_inverted.rows)
//padded.create(orig_inverted.cols, orig_inverted.cols, orig_inverted.type());
//else
//padded.create(orig_inverted.rows, orig_inverted.rows, orig_inverted.type());
//unsigned int padded_width = padded.cols;
//unsigned int padded_height = padded.rows;
//padded.setTo(cv::Scalar::all(0));
//orig_inverted.copyTo(padded(cv::Rect(0, 0, orig_inverted.cols, orig_inverted.rows)));
//cv::resize(padded, padded, cv::Size(input_width, input_height));

//cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
//input_width * input_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
//std::vector<torch::jit::IValue> inputs;
//inputs.push_back(gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({ 2 })); // Must flip first
//cudaMemcpy(orientation, model->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//3 * sizeof(float), cudaMemcpyDeviceToHost);
//cudaMemcpy(z_norm, model_z->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//sizeof(float), cudaMemcpyDeviceToHost);

///*Flip Segment*/
//cv::Mat output_mat_seg = cv::Mat(orig_inverted.rows, orig_inverted.cols, CV_8UC1);
//cv::flip(orig_inverted, output_mat_seg, 0);

///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, -calibration_file_.camera_A_principal_.principal_distance_, orientation[1], orientation[2], orientation[0]));

///*Copy To Mat*/
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

///*OpenCV Image Container/Write Function*/
//cv::Mat projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image); /*Reverse before flip*/
//cv::Mat output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get Scale*/
//double z = -calibration_file_.camera_A_principal_.principal_distance_ * z_norm[0];

///*Reproject*/
///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, z, orientation[1], orientation[2], orientation[0]));
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image);
//output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get X and Y*/
//cv::Mat proj64;
//output_mat.convertTo(proj64, CV_64FC1);
//cv::Mat seg64;
//output_mat_seg.convertTo(seg64, CV_64FC1);
//cv::Point2d x_y_point = cv::phaseCorrelate(proj64, seg64) * (calibration_file_.camera_A_principal_.pixel_pitch_ * z * -1) /
//calibration_file_.camera_A_principal_.principal_distance_;
//double x = x_y_point.x;
//double y = -1 * x_y_point.y;


///*Convert from (0,0) Centered*/
//float za_rad = orientation[0] * pi / 180.0;
//float xa_rad = orientation[1] * pi / 180.0;
//float ya_rad = orientation[2] * pi / 180.0;
//float cz = cos(za_rad);
//float sz = sin(za_rad);
//float cx = cos(xa_rad);
//float sx = sin(xa_rad);
//float cy = cos(ya_rad);
//float sy = sin(ya_rad);
//Matrix_3_3 R_g(
//cz * cy - sz * sx * sy, -1.0 * sz * cx, cz * sy + sz * cy * sx,
//sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * cy * sx,
//-1.0 * cx * sy, sx, cx * cy);
//float theta_x = std::atan(-1.0 * y / z);
//float theta_y = std::asin(-1.0 * x / std::sqrt(x * x + y * y + z * z));
//Matrix_3_3 R_x(
//1, 0, 0,
//0, cos(theta_x), -sin(theta_x),
//0, sin(theta_x), cos(theta_x));
//Matrix_3_3 R_y(
//cos(theta_y), 0, sin(theta_y),
//0, 1, 0,
//-sin(theta_y), 0, cos(theta_y));
//Matrix_3_3 R_orig = calibration_file_.multiplication_mat_mat(R_y, calibration_file_.multiplication_mat_mat(R_x, R_g));
///*Rot Mat To Eul ZXY*/
///*Algorithm To Recover Z - X - Y Euler Angles*/
//float xa, ya, za;
//if (R_orig.A_32_ < 1) {
//if (R_orig.A_32_ > -1) {
//xa = asin(R_orig.A_32_);
//za = atan2(-1 * R_orig.A_12_, R_orig.A_22_);
//ya = atan2(-1 * R_orig.A_31_, R_orig.A_33_);

//}
//else {
//xa = -pi / 2.0;
//za = -1 * atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}
//}
//else {
//xa = pi / 2.0;
//za = atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}

//xa = xa * 180.0 / pi;
//ya = ya * 180.0 / pi;
//za = za * 180.0 / pi;

///*Update Model Pose*/
//model_locations_.SavePose(i, ui.model_list_widget->currentRow(), Point6D(x, y, z, xa, ya, za));

//ui.pose_progress->setValue(65 + 30 * (double)(i + 1) / (double)ui.image_list_widget->count());
//ui.qvtk_widget->update();
//qApp->processEvents();
//}

//ui.pose_progress->setValue(98);
//ui.pose_label->setText("Deleting old models...");
//ui.qvtk_widget->update();

///*Delete GPU Model*/
//delete gpu_mod;

///*Free Array*/
//free(host_image);

///*Free Values*/
//delete orientation;
//delete z_norm;

///*Update Model*/
//Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());
//model_actor_list[selected[0].row()]->SetPosition(loaded_pose.x, loaded_pose.y, loaded_pose.z);
//model_actor_list[selected[0].row()]->SetOrientation(loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
//ui.qvtk_widget->update();

//ui.pose_progress->setValue(100);
//ui.pose_label->setText("Finished!");
//ui.qvtk_widget->update();

///*Pose Estimate Progress and Label Not Visible*/
//ui.pose_progress->setVisible(0);
//ui.pose_label->setVisible(0);
//}
//void MainScreen::on_actionEstimate_Scapula_s_triggered() {
////Must be in Single Selection Mode to Load Pose
//if (ui.multiple_model_radio_button->isChecked()) {
//QMessageBox::critical(this, "Error!", "Must Be in Single Model Selection Mode to Estimate Kinematics!", QMessageBox::Ok);
//return;
//}

////Must load a model
//if (loaded_models.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load a model!", QMessageBox::Ok);
//return;
//}

////Must have loaded image
//if (loaded_frames.size() < 1) {
//QMessageBox::critical(this, "Error!", "Must load images!", QMessageBox::Ok);
//return;
//}

///*Pose Estimate Progress and Label Visible*/
//ui.pose_progress->setValue(5);
//ui.pose_progress->setVisible(1);
//ui.pose_label->setText("Initializing high resolution segmentation...");
//ui.pose_label->setVisible(1);
//ui.qvtk_widget->update();
//qApp->processEvents();


///*Segment*/
//segmentHelperFunction("C:/TorchScriptTrainedNetworks/HRNETSeg_BS24_dataAkiraMayPaperNoPatient18Scapula_512_072419_2_TORCH_SCRIPT.pt", 512, 512);
//unsigned int input_height = 512;
//unsigned int input_width = 512;
//unsigned int orig_height = loaded_frames[0].GetInvertedImage().rows;
//unsigned int orig_width = loaded_frames[0].GetInvertedImage().cols;
//unsigned char* host_image = (unsigned char*)malloc(orig_width * orig_height * sizeof(unsigned char));
//ui.pose_label->setText("Initializing STL model on GPU...");
//ui.qvtk_widget->update();

///*STL Information*/
//vector<vector<float>> triangle_information;
//QModelIndexList selected = ui.model_list_widget->selectionModel()->selectedRows();
//stl_reader_BIG::readAnySTL(QString::fromStdString(loaded_models[selected[0].row()].file_location_), triangle_information);

///*GPU Models for the current Model*/
//gpu_cost_function::GPUModel* gpu_mod = new gpu_cost_function::GPUModel("model", true, orig_height, orig_width, 0, false, // switched cols and rows because the stored image is inverted?
//&(triangle_information[0])[0], &(triangle_information[1])[0], triangle_information[0].size() / 9, calibration_file_.camera_A_principal_); // BACKFACE CULLING APPEARS TO BE GIVING ERRORS

//ui.pose_progress->setValue(55);
//ui.pose_label->setText("Initializing scapula pose estimation...");
//ui.qvtk_widget->update();

///*Load JIT Model*/
//std::string pt_model_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataAkiraPt18_HRProcessed_Sca_08062019_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module module(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model = &module;
//if (model == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Load JIT Z Model*/
//std::string pt_model_z_location = "C:/TorchScriptTrainedNetworks/HRNETPR_BS6_dataAkiraPt18_HRProcessed_Sca_08062019_Z_1_TORCH_SCRIPT.pt";
//// std::shared_ptr<torch::jit::Module> model_z(torch::jit::load(pt_model_z_location, torch::kCUDA));
//torch::jit::Module module_z(torch::jit::load(pt_model_location, torch::kCUDA));
//torch::jit::Module* model_z = &module_z;
//if (model_z == nullptr)
//{
//QMessageBox::critical(this, "Error!", QString::fromStdString("Cannot load PyTorch Torch Script model at: " + pt_model_location), QMessageBox::Ok);
//return;
//}

///*Send Each Segmented Image to GPU Tensor, Predict Orientation, Then Z (From Area), then X,Y.
//After this, convert to non (0,0) centered orientation.
//Finally, update */
//ui.pose_progress->setValue(65);
//ui.pose_label->setText("Estimating scapula poses...");
//ui.qvtk_widget->update();
//float* orientation = new float[3];
//float* z_norm = new float[1];
//torch::Tensor gpu_byte_placeholder(torch::zeros({ 1, 1, input_height, input_width }, torch::device(torch::kCUDA).dtype(torch::kByte)));
//for (int i = 0; i < ui.image_list_widget->count(); i++) {

//cv::Mat orig_inverted = loaded_frames[i].GetInvertedImage();
//cv::Mat padded;
//if (orig_inverted.cols > orig_inverted.rows)
//padded.create(orig_inverted.cols, orig_inverted.cols, orig_inverted.type());
//else
//padded.create(orig_inverted.rows, orig_inverted.rows, orig_inverted.type());
//unsigned int padded_width = padded.cols;
//unsigned int padded_height = padded.rows;
//padded.setTo(cv::Scalar::all(0));
//orig_inverted.copyTo(padded(cv::Rect(0, 0, orig_inverted.cols, orig_inverted.rows)));
//cv::resize(padded, padded, cv::Size(input_width, input_height));

//cudaMemcpy(gpu_byte_placeholder.data_ptr(), padded.data,
//input_width * input_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
//std::vector<torch::jit::IValue> inputs;
//inputs.push_back(gpu_byte_placeholder.to(torch::dtype(torch::kFloat)).flip({ 2 })); // Must flip first
//cudaMemcpy(orientation, model->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//3 * sizeof(float), cudaMemcpyDeviceToHost);
//cudaMemcpy(z_norm, model_z->forward(inputs).toTensor().to(torch::dtype(torch::kFloat)).data_ptr(),
//sizeof(float), cudaMemcpyDeviceToHost);

///*Flip Segment*/
//cv::Mat output_mat_seg = cv::Mat(orig_inverted.rows, orig_inverted.cols, CV_8UC1);
//cv::flip(orig_inverted, output_mat_seg, 0);

///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, -calibration_file_.camera_A_principal_.principal_distance_, orientation[1], orientation[2], orientation[0]));

///*Copy To Mat*/
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

///*OpenCV Image Container/Write Function*/
//cv::Mat projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image); /*Reverse before flip*/
//cv::Mat output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get Scale*/
//double z = -calibration_file_.camera_A_principal_.principal_distance_ * z_norm[0];

///*Reproject*/
///*Render*/
//gpu_mod->RenderPrimaryCamera(gpu_cost_function::Pose(0, 0, z, orientation[1], orientation[2], orientation[0]));
//cudaMemcpy(host_image, gpu_mod->GetPrimaryCameraRenderedImagePointer(), orig_width * orig_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//projection_mat = cv::Mat(orig_height, orig_width, CV_8UC1, host_image);
//output_mat = cv::Mat(orig_width, orig_height, CV_8UC1);
//cv::flip(projection_mat, output_mat, 0);

///*Get X and Y*/
//cv::Mat proj64;
//output_mat.convertTo(proj64, CV_64FC1);
//cv::Mat seg64;
//output_mat_seg.convertTo(seg64, CV_64FC1);
//cv::Point2d x_y_point = cv::phaseCorrelate(proj64, seg64) * (calibration_file_.camera_A_principal_.pixel_pitch_ * z * -1) /
//calibration_file_.camera_A_principal_.principal_distance_;
//double x = x_y_point.x;
//double y = -1 * x_y_point.y;


///*Convert from (0,0) Centered*/
//float za_rad = orientation[0] * pi / 180.0;
//float xa_rad = orientation[1] * pi / 180.0;
//float ya_rad = orientation[2] * pi / 180.0;
//float cz = cos(za_rad);
//float sz = sin(za_rad);
//float cx = cos(xa_rad);
//float sx = sin(xa_rad);
//float cy = cos(ya_rad);
//float sy = sin(ya_rad);
//Matrix_3_3 R_g(
//cz * cy - sz * sx * sy, -1.0 * sz * cx, cz * sy + sz * cy * sx,
//sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * cy * sx,
//-1.0 * cx * sy, sx, cx * cy);
//float theta_x = std::atan(-1.0 * y / z);
//float theta_y = std::asin(-1.0 * x / std::sqrt(x * x + y * y + z * z));
//Matrix_3_3 R_x(
//1, 0, 0,
//0, cos(theta_x), -sin(theta_x),
//0, sin(theta_x), cos(theta_x));
//Matrix_3_3 R_y(
//cos(theta_y), 0, sin(theta_y),
//0, 1, 0,
//-sin(theta_y), 0, cos(theta_y));
//Matrix_3_3 R_orig = calibration_file_.multiplication_mat_mat(R_y, calibration_file_.multiplication_mat_mat(R_x, R_g));
///*Rot Mat To Eul ZXY*/
///*Algorithm To Recover Z - X - Y Euler Angles*/
//float xa, ya, za;
//if (R_orig.A_32_ < 1) {
//if (R_orig.A_32_ > -1) {
//xa = asin(R_orig.A_32_);
//za = atan2(-1 * R_orig.A_12_, R_orig.A_22_);
//ya = atan2(-1 * R_orig.A_31_, R_orig.A_33_);

//}
//else {
//xa = -pi / 2.0;
//za = -1 * atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}
//}
//else {
//xa = pi / 2.0;
//za = atan2(R_orig.A_13_, R_orig.A_11_);
//ya = 0;
//}

//xa = xa * 180.0 / pi;
//ya = ya * 180.0 / pi;
//za = za * 180.0 / pi;

///*Update Model Pose*/
//model_locations_.SavePose(i, ui.model_list_widget->currentRow(), Point6D(x, y, z, xa, ya, za));

//ui.pose_progress->setValue(65 + 30 * (double)(i + 1) / (double)ui.image_list_widget->count());
//ui.qvtk_widget->update();
//qApp->processEvents();
//}
//QMessageBox::critical(this, "Error!", "zzzzzzzzzz!", QMessageBox::Ok);

//ui.pose_progress->setValue(98);
//ui.pose_label->setText("Deleting old models...");
//ui.qvtk_widget->update();

///*Delete GPU Model*/
//delete gpu_mod;

///*Free Array*/
//free(host_image);

///*Free Values*/
//delete orientation;
//delete z_norm;

///*Update Model*/
//Point6D loaded_pose = model_locations_.GetPose(ui.image_list_widget->currentRow(), selected[0].row());
//model_actor_list[selected[0].row()]->SetPosition(loaded_pose.x, loaded_pose.y, loaded_pose.z);
//model_actor_list[selected[0].row()]->SetOrientation(loaded_pose.xa, loaded_pose.ya, loaded_pose.za);
//ui.qvtk_widget->update();

//ui.pose_progress->setValue(100);
//ui.pose_label->setText("Finished!");
//ui.qvtk_widget->update();

///*Pose Estimate Progress and Label Not Visible*/
//ui.pose_progress->setVisible(0);
//ui.pose_label->setVisible(0);
//}