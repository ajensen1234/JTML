// Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
// SPDX-License-Identifier: AGPL-3.0

/*About Window Header*/
#include "gui/about.h"

/*CUDA Headers*/
#include <cuda_runtime.h>

// About JTA Popup CPP
About::About(QWidget* parent, Qt::WindowFlags flags) : QDialog(parent, flags) {
    ui.setupUi(this);

    /*Check CUDA Stats*/
    int gpu_device_count = 0, device_count;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&device_count);
    if (cudaResultCode != cudaSuccess) device_count = 0;
    /* Machines with no GPUs can still report one emulation device */
    for (int device = 0; device < device_count; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999 &&
            properties.major >= 5) /* 9999 means emulation only */
            ++gpu_device_count;
    }
    /*If no Cuda Compatitble Devices*/
    if (device_count == 0) {
        ui.gpu_label->setText("None Found");
        ui.cc_label->setText("N/A");
    } else if (gpu_device_count == 0) {
        ui.gpu_label->setText(QString(properties.name) + " (Incompatible)");
        ui.cc_label->setText(QString::number(properties.major) + "." +
                             QString::number(properties.minor) + " (Too Low)");
    } else {
        ui.gpu_label->setText(QString(properties.name));
        ui.cc_label->setText(QString::number(properties.major) + "." +
                             QString::number(properties.minor));
    }

    /*Set Title Font to Larger*/
    QFont title_font = ui.copyright_label->font();
    title_font.setPointSize(1.5 * title_font.pointSize());
    title_font.setBold(true);
    ui.title_label->setFont(title_font);

    /*Font Metrics*/
    QFontMetrics title_metric(title_font);
    QFontMetrics text_metric(ui.copyright_label->font());

    /*Adjust for Title Height*/
    ui.detected_group_box->setStyleSheet(
        ui.detected_group_box->styleSheet() +=
        "QGroupBox { margin-top: " + QString::number(text_metric.height() / 2) +
        "px; }");

    /*Constants for Sizing Adjustments*/
    int GROUP_BOX_TO_LABEL_X = 50;
    int GROUP_BOX_TO_LABEL_Y = 30;
    int LABEL_TO_LABEL_X = 20;
    int LABEL_TO_LABEL_Y = 20;
    int LABEL_TO_GROUP_BOX_Y = 20;
    int group_box_to_top_button_y = text_metric.height() / 2;
    int INSIDE_BUTTON_PADDING_X = 80;
    int INSIDE_BUTTON_PADDING_Y = 30;

    /*Get Max Width*/
    int max_width =
        GROUP_BOX_TO_LABEL_X * 2 + LABEL_TO_LABEL_X +
        text_metric.horizontalAdvance(ui.gpu_description_label->text()) +
        text_metric.horizontalAdvance(ui.gpu_label->text());
    if (max_width <
        GROUP_BOX_TO_LABEL_X * 2 + LABEL_TO_LABEL_X +
            text_metric.horizontalAdvance(ui.cc_description_label->text()) +
            text_metric.horizontalAdvance(ui.cc_label->text()))
        max_width =
            GROUP_BOX_TO_LABEL_X * 2 + LABEL_TO_LABEL_X +
            text_metric.horizontalAdvance(ui.cc_description_label->text()) +
            text_metric.horizontalAdvance(ui.cc_label->text());
    int group_box_width = max_width;
    if (max_width < title_metric.horizontalAdvance(ui.title_label->text()))
        max_width = title_metric.horizontalAdvance(ui.title_label->text());

    /*Set Positions*/
    ui.title_label->setGeometry(
        QRect(306 + (max_width -
                     title_metric.horizontalAdvance(ui.title_label->text())) /
                        2,
              25, title_metric.horizontalAdvance(ui.title_label->text()),
              title_metric.height()));
    ui.copyright_label->setGeometry(QRect(
        306 + (max_width -
               text_metric.horizontalAdvance(ui.copyright_label->text())) /
                  2,
        LABEL_TO_LABEL_Y + ui.title_label->geometry().bottom(),
        text_metric.horizontalAdvance(ui.copyright_label->text()),
        text_metric.height()));
    ui.version_label->setGeometry(
        QRect(306 + (max_width -
                     text_metric.horizontalAdvance(ui.version_label->text())) /
                        2,
              LABEL_TO_LABEL_Y + ui.copyright_label->geometry().bottom(),
              text_metric.horizontalAdvance(ui.version_label->text()),
              text_metric.height()));
    ui.detected_group_box->setGeometry(
        QRect(306 + (max_width - group_box_width) / 2,
              ui.version_label->geometry().bottom() + LABEL_TO_GROUP_BOX_Y,
              group_box_width,
              group_box_to_top_button_y + 2 * GROUP_BOX_TO_LABEL_Y +
                  LABEL_TO_LABEL_Y + 2 * text_metric.height()));
    ui.gpu_description_label->setGeometry(QRect(
        GROUP_BOX_TO_LABEL_X, group_box_to_top_button_y + GROUP_BOX_TO_LABEL_Y,
        text_metric.horizontalAdvance(ui.gpu_description_label->text()),
        text_metric.height()));
    ui.cc_description_label->setGeometry(QRect(
        GROUP_BOX_TO_LABEL_X +
            text_metric.horizontalAdvance(ui.gpu_description_label->text()) -
            text_metric.horizontalAdvance(ui.cc_description_label->text()),
        ui.gpu_description_label->geometry().bottom() + LABEL_TO_LABEL_Y,
        text_metric.horizontalAdvance(ui.cc_description_label->text()),
        text_metric.height()));
    ui.gpu_label->setGeometry(
        QRect(ui.gpu_description_label->geometry().right() + LABEL_TO_LABEL_X,
              group_box_to_top_button_y + GROUP_BOX_TO_LABEL_Y,
              text_metric.horizontalAdvance(ui.gpu_label->text()),
              text_metric.height()));
    ui.cc_label->setGeometry(
        QRect(ui.gpu_description_label->geometry().right() + LABEL_TO_LABEL_X,
              ui.gpu_description_label->geometry().bottom() + LABEL_TO_LABEL_Y,
              text_metric.horizontalAdvance(ui.cc_label->text()),
              text_metric.height()));
    ui.close_button->setGeometry(
        QRect(306 + (max_width -
                     (text_metric.horizontalAdvance(ui.close_button->text()) +
                      INSIDE_BUTTON_PADDING_X)) /
                        2,
              ui.detected_group_box->geometry().bottom() + GROUP_BOX_TO_LABEL_Y,
              text_metric.horizontalAdvance(ui.close_button->text()) +
                  INSIDE_BUTTON_PADDING_X,
              text_metric.height() + INSIDE_BUTTON_PADDING_Y));

    /*Set Minimum Width and Height*/
    setFixedSize(331 + max_width, ui.close_button->geometry().bottom() + 25);
}

About::~About() {}

// Set Version Number
void About::setVersion(int A, int B, int C) {
    ui.version_label->setText("Version " + QString::number(A) + "." +
                              QString::number(B) + "." + QString::number(C));
}
