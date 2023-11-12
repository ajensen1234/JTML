#pragma once

#if defined(_WIN32) || defined(_WIN64)

#define JTML_DLL __declspec(dllexport)

#else
#define JTML_DLL
#endif
