#include "common.h"

SceCtrlData pad, old_pad;

volatile int done = 0;

// psp callbacks
int exit_callback(int arg1, int arg2, void *common)
{
	done = 1;
	return 0;
}

int CallbackThread(SceSize args, void *argp)
{
	int cbid = sceKernelCreateCallback("Exit Callback", exit_callback, NULL);
	sceKernelRegisterExitCallback(cbid);
	sceKernelSleepThreadCB();
	return 0;
}

int SetupCallbacks(void)
{
	int thid = sceKernelCreateThread("update_thread", CallbackThread, 0x11, 0xFA0, PSP_THREAD_ATTR_USER, 0);
	if(thid >= 0)
		sceKernelStartThread(thid, 0, 0);
	return thid;
}

void delay(int ms) { sceKernelDelayThread(ms * 1000); }

void print_mem_info(const char* stage) {
    //newScreen(0, pspDebugScreenGetY() + 1);
    print("[%s] Free memory: %d KB\n", stage, sceKernelMaxFreeMemSize() / 1024);
    sceKernelDelayThread(2000000);
}

bool is_btn_pressed(SceCtrlData pad, SceCtrlData old_pad, int button) {
    return (pad.Buttons & button) && !(old_pad.Buttons & button);
}

bool handle_text_input(char* buffer, int buffer_size, const char* title) {
    static int cursor_x = 0;
    static int cursor_y = 0;

    // layout do teclado
    const char* keyboard_layout[] = {
        "1234567890",
        "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm",
        " "
    };
    int num_rows = 5;
    
    // d-pad navigation (simplifiquei)
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_UP)) cursor_y--;
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_DOWN)) cursor_y++;
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_LEFT)) cursor_x--;
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_RIGHT)) cursor_x++;

    // logic wrap
    if (cursor_y < 0) cursor_y = num_rows - 1;
    if (cursor_y >= num_rows) cursor_y = 0;

    int current_row_len = strlen(keyboard_layout[cursor_y]);
    if (cursor_x < 0) cursor_x = current_row_len - 1;
    if (cursor_x >= current_row_len) cursor_x = 0;

    //button actions
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_CROSS)) {
        int len = strlen(buffer);
        if (len < buffer_size - 1) {
            // add char here to buffer
            buffer[len] = keyboard_layout[cursor_y][cursor_x];
            buffer[len + 1] = '\0';
        }
    }

    if (is_btn_pressed(pad, old_pad, PSP_CTRL_CIRCLE)) {
        int len = strlen(buffer);
        if (len > 0) {
            // backspace last char
            buffer[len - 1] = '\0';
        }
    }

    if (is_btn_pressed(pad, old_pad, PSP_CTRL_START)) {
        return true; // finish editing
    }

    newScreen(0, 1);
    print("%s\n\n", title);
    
    print(" > %s_\n\n", buffer);

    // drawing keyoard layout (maube move)
    for (int y = 0; y < num_rows; y++) {
        for (int x = 0; x < strlen(keyboard_layout[y]); x++) {
            if (y == cursor_y && x == cursor_x) {
                pspDebugScreenPrintf("[%c]", keyboard_layout[y][x]);
            } else {
                pspDebugScreenPrintf(" %c ", keyboard_layout[y][x]);
            }
        }
        pspDebugScreenPrintf("\n");
    }
    
    print("\n\n[X] Write | [O] Backspace | [START] Confirm\n");

    return false; // return false if not finish eddting yet
}