/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

/*Constants for resizing the spacings on the main screen*/

#ifndef MAINSCREEN_SIZE_CONSTANTS_H
#define MAINSCREEN_SIZE_CONSTANTS_H

/*Sizes for Main Window*/
int MINIMUM_WIDTH = 1600;  // 1900;
int MINIMUM_HEIGHT = 918;  // 900;

/*Minimum List Widget Size*/
const int MINIMUM_LIST_WIDGET_SIZE = 100;

/*Minimum Qvtk widget width*/
const int MINIMUM_QVTK_WIDGET_WIDTH = 831;

/*Border Paddings between object A and object B*/
const int GROUP_BOX_TO_BUTTON_PADDING_X = 25;
const int BUTTON_TO_BUTTON_PADDING_X = 11;
const int INSIDE_BUTTON_PADDING_X = 30;
const int INSIDE_RADIO_BUTTON_PADDING_X = 40;
const int APPLICATION_BORDER_TO_GROUP_BOX_PADDING_X = 55;
const int INSIDE_SPIN_BOX_PADDING_X = 25;
const int LABEL_TO_SPIN_BOX_PADDING_X = 15;
const int SPIN_BOX_TO_GROUP_BOX_PADDING_X = 60;
const int INSIDE_BUTTON_PADDING_RIGHT_COLUMN_X = 50;
const int GROUP_BOX_TO_QVTK_PADDING_X = 65;

const int GROUP_BOX_TO_GROUP_BOX_Y = 30;
const int GROUP_BOX_TO_BUTTON_PADDING_Y = 30;
const int BUTTON_TO_BUTTON_PADDING_Y = 11;
const int SPIN_BOX_TO_SPIN_BOX_PADDING_Y = 15;
const int INSIDE_BUTTON_PADDING_Y = 30;
const int INSIDE_RADIO_BUTTON_PADDING_Y = 10;
const int APPLICATION_BORDER_TO_GROUP_BOX_PADDING_Y = 40;
const int RADIO_BUTTON_TO_LIST_WIDGET_PADDING_Y = 25;
const int INSIDE_SPIN_BOX_PADDING_Y = 15;

/*Font Size*/
const int FONT_SIZE = 8;

#endif /* MAINSCREEN_SIZE_CONSTANTS_H */