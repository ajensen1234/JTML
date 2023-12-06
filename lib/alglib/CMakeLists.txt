# Copyright 2023 Gary J. Miller Orthopaedic Biomechanics Lab
# SPDX-License-Identifier: AGPL-3.0

file(GLOB my-files *.cpp *.h)

add_library(alglib STATIC ${my-files})