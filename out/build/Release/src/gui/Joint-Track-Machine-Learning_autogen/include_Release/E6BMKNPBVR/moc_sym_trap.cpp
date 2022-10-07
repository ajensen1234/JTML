/****************************************************************************
** Meta object code from reading C++ file 'sym_trap.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../../../../../../include/gui/sym_trap.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'sym_trap.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_sym_trap_t {
    QByteArrayData data[12];
    char stringdata0[110];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_sym_trap_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_sym_trap_t qt_meta_stringdata_sym_trap = {
    {
QT_MOC_LITERAL(0, 0, 8), // "sym_trap"
QT_MOC_LITERAL(1, 9, 4), // "Done"
QT_MOC_LITERAL(2, 14, 0), // ""
QT_MOC_LITERAL(3, 15, 17), // "onCostFuncAtPoint"
QT_MOC_LITERAL(4, 33, 6), // "result"
QT_MOC_LITERAL(5, 40, 12), // "graphResults"
QT_MOC_LITERAL(6, 53, 14), // "graphResults2D"
QT_MOC_LITERAL(7, 68, 12), // "setIterCount"
QT_MOC_LITERAL(8, 81, 1), // "n"
QT_MOC_LITERAL(9, 83, 8), // "saveData"
QT_MOC_LITERAL(10, 92, 8), // "loadData"
QT_MOC_LITERAL(11, 101, 8) // "savePlot"

    },
    "sym_trap\0Done\0\0onCostFuncAtPoint\0"
    "result\0graphResults\0graphResults2D\0"
    "setIterCount\0n\0saveData\0loadData\0"
    "savePlot"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_sym_trap[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   54,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    1,   55,    2, 0x0a /* Public */,
       5,    0,   58,    2, 0x0a /* Public */,
       6,    0,   59,    2, 0x0a /* Public */,
       7,    1,   60,    2, 0x0a /* Public */,
       9,    0,   63,    2, 0x0a /* Public */,
      10,    0,   64,    2, 0x0a /* Public */,
      11,    0,   65,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Double, QMetaType::Double,    4,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void sym_trap::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<sym_trap *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->Done(); break;
        case 1: { double _r = _t->onCostFuncAtPoint((*reinterpret_cast< double(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 2: _t->graphResults(); break;
        case 3: _t->graphResults2D(); break;
        case 4: _t->setIterCount((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->saveData(); break;
        case 6: _t->loadData(); break;
        case 7: _t->savePlot(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (sym_trap::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&sym_trap::Done)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject sym_trap::staticMetaObject = { {
    QMetaObject::SuperData::link<QDialog::staticMetaObject>(),
    qt_meta_stringdata_sym_trap.data,
    qt_meta_data_sym_trap,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *sym_trap::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *sym_trap::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_sym_trap.stringdata0))
        return static_cast<void*>(this);
    return QDialog::qt_metacast(_clname);
}

int sym_trap::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void sym_trap::Done()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
