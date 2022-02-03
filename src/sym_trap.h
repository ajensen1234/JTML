#pragma once
#include <qdialog.h>

#include "ui_sym_trap.h"

class sym_trap :public QDialog
{
	Q_OBJECT

public:
	sym_trap(QWidget* parent = 0, Qt::WindowFlags flags = 0);
	~sym_trap();

private:
	Ui::symTrap ui;

signals:
	void Done();

};