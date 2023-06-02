#pragma once

#if defined (_WIN32) || defined (_WIN64)

#ifdef JTML_EXPORTS
#define JTML_DLL JTML_DLL

#else
#define JTML_DLL __declspec(dllimport)

#endif

#else
#define JTML_DLL
#endif
