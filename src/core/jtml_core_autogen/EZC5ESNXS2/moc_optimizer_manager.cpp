// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/****************************************************************************
** Meta object code from reading C++ file 'optimizer_manager.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../../../include/core/optimizer_manager.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'optimizer_manager.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_OptimizerManager_t {
    QByteArrayData data[14];
    char stringdata0[213];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_OptimizerManager_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_OptimizerManager_t qt_meta_stringdata_OptimizerManager = {
    {
QT_MOC_LITERAL(0, 0, 16), // "OptimizerManager"
QT_MOC_LITERAL(1, 17, 13), // "UpdateOptimum"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 8), // "finished"
QT_MOC_LITERAL(4, 41, 14), // "OptimizedFrame"
QT_MOC_LITERAL(5, 56, 14), // "OptimizerError"
QT_MOC_LITERAL(6, 71, 13), // "UpdateDisplay"
QT_MOC_LITERAL(7, 85, 24), // "UpdateDilationBackground"
QT_MOC_LITERAL(8, 110, 15), // "CostFuncAtPoint"
QT_MOC_LITERAL(9, 126, 26), // "onUpdateOrientationSymTrap"
QT_MOC_LITERAL(10, 153, 19), // "onProgressBarUpdate"
QT_MOC_LITERAL(11, 173, 14), // "get_iter_count"
QT_MOC_LITERAL(12, 188, 8), // "Optimize"
QT_MOC_LITERAL(13, 197, 15) // "onStopOptimizer"

    },
    "OptimizerManager\0UpdateOptimum\0\0"
    "finished\0OptimizedFrame\0OptimizerError\0"
    "UpdateDisplay\0UpdateDilationBackground\0"
    "CostFuncAtPoint\0onUpdateOrientationSymTrap\0"
    "onProgressBarUpdate\0get_iter_count\0"
    "Optimize\0onStopOptimizer"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_OptimizerManager[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      12,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
      10,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    7,   74,    2, 0x06 /* Public */,
       3,    0,   89,    2, 0x06 /* Public */,
       4,   10,   90,    2, 0x06 /* Public */,
       5,    1,  111,    2, 0x06 /* Public */,
       6,    4,  114,    2, 0x06 /* Public */,
       7,    0,  123,    2, 0x06 /* Public */,
       8,    1,  124,    2, 0x06 /* Public */,
       9,    6,  127,    2, 0x06 /* Public */,
      10,    1,  140,    2, 0x06 /* Public */,
      11,    0,  143,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      12,    0,  144,    2, 0x0a /* Public */,
      13,    0,  145,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::UInt,    2,    2,    2,    2,    2,    2,    2,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Bool, QMetaType::UInt, QMetaType::Bool, QMetaType::QString,    2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
    QMetaType::Void, QMetaType::QString,    2,
    QMetaType::Void, QMetaType::Double, QMetaType::Int, QMetaType::Double, QMetaType::UInt,    2,    2,    2,    2,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double,    2,
    QMetaType::Void, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double, QMetaType::Double,    2,    2,    2,    2,    2,    2,
    QMetaType::Void, QMetaType::Int,    2,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void OptimizerManager::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<OptimizerManager *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->UpdateOptimum((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6])),(*reinterpret_cast< uint(*)>(_a[7]))); break;
        case 1: _t->finished(); break;
        case 2: _t->OptimizedFrame((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6])),(*reinterpret_cast< bool(*)>(_a[7])),(*reinterpret_cast< uint(*)>(_a[8])),(*reinterpret_cast< bool(*)>(_a[9])),(*reinterpret_cast< QString(*)>(_a[10]))); break;
        case 3: _t->OptimizerError((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 4: _t->UpdateDisplay((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< uint(*)>(_a[4]))); break;
        case 5: _t->UpdateDilationBackground(); break;
        case 6: _t->CostFuncAtPoint((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 7: _t->onUpdateOrientationSymTrap((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< double(*)>(_a[4])),(*reinterpret_cast< double(*)>(_a[5])),(*reinterpret_cast< double(*)>(_a[6]))); break;
        case 8: _t->onProgressBarUpdate((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->get_iter_count(); break;
        case 10: _t->Optimize(); break;
        case 11: _t->onStopOptimizer(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (OptimizerManager::*)(double , double , double , double , double , double , unsigned int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptimizerManager::UpdateOptimum)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (OptimizerManager::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptimizerManager::finished)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (OptimizerManager::*)(double , double , double , double , double , double , bool , unsigned int , bool , QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptimizerManager::OptimizedFrame)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (OptimizerManager::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptimizerManager::OptimizerError)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (OptimizerManager::*)(double , int , double , unsigned int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptimizerManager::UpdateDisplay)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (OptimizerManager::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptimizerManager::UpdateDilationBackground)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (OptimizerManager::*)(double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptimizerManager::CostFuncAtPoint)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (OptimizerManager::*)(double , double , double , double , double , double );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptimizerManager::onUpdateOrientationSymTrap)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (OptimizerManager::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptimizerManager::onProgressBarUpdate)) {
                *result = 8;
                return;
            }
        }
        {
            using _t = void (OptimizerManager::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&OptimizerManager::get_iter_count)) {
                *result = 9;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject OptimizerManager::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_OptimizerManager.data,
    qt_meta_data_OptimizerManager,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *OptimizerManager::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *OptimizerManager::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_OptimizerManager.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int OptimizerManager::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 12)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 12;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 12)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 12;
    }
    return _id;
}

// SIGNAL 0
void OptimizerManager::UpdateOptimum(double _t1, double _t2, double _t3, double _t4, double _t5, double _t6, unsigned int _t7)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t5))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t6))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t7))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void OptimizerManager::finished()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void OptimizerManager::OptimizedFrame(double _t1, double _t2, double _t3, double _t4, double _t5, double _t6, bool _t7, unsigned int _t8, bool _t9, QString _t10)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t5))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t6))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t7))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t8))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t9))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t10))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void OptimizerManager::OptimizerError(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void OptimizerManager::UpdateDisplay(double _t1, int _t2, double _t3, unsigned int _t4)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void OptimizerManager::UpdateDilationBackground()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}

// SIGNAL 6
void OptimizerManager::CostFuncAtPoint(double _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void OptimizerManager::onUpdateOrientationSymTrap(double _t1, double _t2, double _t3, double _t4, double _t5, double _t6)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t2))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t3))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t4))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t5))), const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t6))) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void OptimizerManager::onProgressBarUpdate(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 8, _a);
}

// SIGNAL 9
void OptimizerManager::get_iter_count()
{
    QMetaObject::activate(this, &staticMetaObject, 9, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
