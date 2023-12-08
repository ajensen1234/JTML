/*
 * Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
 * SPDX-License-Identifier: AGPL-3.0
 */

#pragma once

#if defined(_WIN32) || defined(_WIN64)

#define JTML_DLL __declspec(dllexport)

#else
#define JTML_DLL
#endif
