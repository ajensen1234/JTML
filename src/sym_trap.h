#pragma once
#include <qdialog.h>
#include "optimizer_manager.h"
#include "optimizer_settings.h"
#include "ui_sym_trap.h"

#include <vector>

class sym_trap :public QDialog
{
	Q_OBJECT

public:
	sym_trap(QWidget* parent = 0, Qt::WindowFlags flags = 0);
	~sym_trap();
	OptimizerManager* sym_trap_optimizer = new OptimizerManager();

public slots:
	void gather_dataset();

private:
	Ui::symTrap ui;
	std::vector<Point6D> search_space;

signals:
	void Done();


};