#include "sym_trap.h"



sym_trap::sym_trap(QWidget* parent, Qt::WindowFlags flags) : QDialog(parent, flags)

{
	ui.setupUi(this);

	QObject::connect(this, SIGNAL(Done()), this, SLOT(close()));
	
	QFontMetrics font_metrics(this->font());

	this->setStyleSheet(this->styleSheet() += "QGroupBox { margin-top: " + QString::number(font_metrics.height() / 2) + "px; }");
	int group_box_to_top_button_y = font_metrics.height() / 2;

	//when button is clicked, call gather_dataset()
	QObject::connect(ui.gather_dataset, SIGNAL(clicked()), this, SLOT(gather_dataset()));
	

}

sym_trap::~sym_trap()
{

}


void sym_trap::gather_dataset() {
	// for loop

	//for {point_instance in search_space}
	cout << sym_trap_optimizer->EvaluateCostFunctionAtPoint(Point6D(.5,.5,.5,.5,.5,.5), 0) << "\n";

	 
	//todo: figure out what variables to pass in 





}
