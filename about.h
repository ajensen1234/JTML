#ifndef ABOUT_H
#define ABOUT_H

#include <qdialog.h>
#include "ui_About.h"

//About JTA Popup Header
class About : public QDialog
{
	Q_OBJECT

public:
	About(QWidget *parent = 0, Qt::WindowFlags flags = 0);
	~About();
	void setVersion(int A, int B, int C); // Sets Version Number Label

private:
	Ui::aboutJTA ui;

};

#endif // ABOUT_H