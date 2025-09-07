#ifndef COMMON_H
#define COMMON_H

extern "C" {
    #include <pspkernel.h>
    #include <pspdisplay.h>
    #include <pspdebug.h>
    #include <pspgu.h>
    #include <psputility.h>
    #include <psptypes.h>
    #include <pspctrl.h>
    #include <psprtc.h> // relogio do psp (real-time clock)
    #include <psppower.h>

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <stdint.h>
    #include <math.h>
    #include <stdio.h>
    #include <ctype.h>
    #include <time.h>
    #include <fcntl.h>
    #include <unistd.h>
}
#include <cstdlib> // rand e srand

#define DEBUG 0 // enable some process print

typedef struct AppMetrics {
    float total_generation_time_s;
    float tokens_per_second;
    int generated_token_count;
    int free_memory_kb;
    int cpu_clock_freq;
    int bus_clock_freq;
} AppMetrics;

#define SCREEN_WIDTH 480
#define SCREEN_HEIGHT 272
#define BUF_WIDTH	(512)

#define TOKENIZER_BIN_PATH "tok512.bin"
#define MODEL_PATH "stories260K.bin"

#define print pspDebugScreenPrintf
//volatile int done = 0;
extern volatile int done;

#define PARAMS_OPTIONS_COUNT 3
const u32 COLOR_WHITE = 0xFFFFFFFF;
const u32 COLOR_YELLOW = 0XFF00FFFF;
const u32 COLOR_GREEN = 0xFF00FF00;

// functions new name
#define print pspDebugScreenPrintf
#define setTextColor pspDebugScreenSetTextColor
#define clearScreen pspDebugScreenClear
#define setXY pspDebugScreenSetXY

#define newScreen(x, y) \
    do { \
        clearScreen(); \
        setXY(x, y); \
    } while (0)

extern SceCtrlData pad, old_pad;
extern volatile int done;

int exit_callback(int arg1, int arg2, void *common);
int CallbackThread(SceSize args, void *argp);
int SetupCallbacks(void);
void delay(int ms);
long time_in_ms(void);
void print_mem_info(const char* stage);
bool is_btn_pressed(SceCtrlData pad, SceCtrlData old_pad, int button);
bool handle_text_input(char* buffer, int buffer_size, const char* title);

#endif