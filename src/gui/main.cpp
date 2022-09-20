#include "gui/mainscreen.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	/*Otherwise Cant See TEXT*/
	VTK_MODULE_INIT(vtkRenderingOpenGL2); // Added post migration to Banks' lab computer
	VTK_MODULE_INIT(vtkRenderingFreeType);
	QApplication a(argc, argv);
	MainScreen w;
	w.show();
	return a.exec();
}
