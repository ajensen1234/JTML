#include <QtWidgets/QApplication>

#include "gui/mainscreen.h"

int main(int argc, char* argv[]) {
    /*Otherwise Cant See TEXT*/
    QApplication a(argc, argv);
    MainScreen w;
    w.show();
    return a.exec();
}
