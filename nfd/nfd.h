#pragma once

#include <iostream>
#include <QString>
#include "../JTML/model.h"
#include "../JTML/frame.h"
#include "../JTML/calibration.h"

#include <gpu_model.cuh>
#include <gpu_intensity_frame.cuh>
#include <gpu_edge_frame.cuh>
#include <gpu_dilated_frame.cuh>
#include <gpu_metrics.cuh>
#include <vector>
#include <QModelIndex>
#include "cufft.h"
#include <opencv2/imgproc.hpp>

#include "nfd_instance.h"
#include "nfd_library.h"

using namespace gpu_cost_function;

class JTML_NFD {

public:
	JTML_NFD();
	~JTML_NFD();
	bool Initialize(
		Calibration cal_file,
		std::vector<Model> model_list,
		std::vector<Frame> frames_list,
		QModelIndexList selected_models,
		unsigned int primary_model_index,
		QString error_message
	);

	void Run();

private:
	bool successful_initialization_;
	Calibration calibration_;
	std::vector<Frame> frames_;

	GPUModel* gpu_principal_model_;

	Model primary_model_;
	std::vector<Model> all_models_;

	
};