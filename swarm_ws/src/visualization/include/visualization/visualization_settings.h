#ifndef VIS_SETTINGS_H
#define VIS_SETTINGS_H

/**
 * This file contains universal parameters for visualization
 *
 * This header is included by many files and is effectively "on top"
 */

#define WIDTH 1920
#define HEIGHT 1080

#define WIDTH_2 ((float)WIDTH / 2.0f)
#define HEIGHT_2 ((float)HEIGHT / 2.0f)

// table size with respect to the table
// 	in cm
#define TAB_WIDTH 100
#define TAB_HEIGHT 200

#define TAB_WIDTH_2 ((float)TAB_WIDTH / 2.0f)
#define TAB_HEIGHT_2 ((float)TAB_HEIGHT / 2.0f)

static int g_draw_level;

#endif
