#include "common.h"

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

// utilities: time

// long time_in_ms() {
//   // return time in milliseconds, for benchmarking the model speed
//   struct timespec time;
//   clock_gettime(CLOCK_REALTIME, &time);
//   return time.tv_sec * 1000 + time.tv_nsec / 1000000;
// }
