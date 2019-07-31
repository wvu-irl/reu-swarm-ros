#ifndef SIM_SETT_H
#define SIM_SETT_H

// overall number of bots
#define NUMBOTS 20

// Main sim window
// 	O is for overall
#define O_SIM_WID 300
#define O_SIM_HEI 600

#define O_SIM_WID_2 ((float)O_SIM_WID / 2.0f)
#define O_SIM_HEI_2 ((float)O_SIM_HEI / 2.0f)


// generic, helpful macros
#include <chrono>
#include <string>
#include <string.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define DPRINTF(form, dat...) \
	printf(("[%d.%09d] (%s:%d) " + std::string(form) + "\n").c_str(),\
			std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(),\
			std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() % 1000000000\
			, __FILENAME__, __LINE__, dat\
			)

#define DPUTS(form) \
	printf(("[%d.%09d] (%s:%d) " + std::string(form) + "\n").c_str(),\
			std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(),\
			std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() % 1000000000\
			, __FILENAME__, __LINE__\
			)

#endif /* SIM_SETT_H */
