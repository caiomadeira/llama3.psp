/*

Inference for Llama-3 Transformer Model
PSP port by Caio Madeira

*/
#include "src/tokenizer.h"
#include "src/generate.h"
#include "src/nnet.h"
#include "src/sampler.h"
#include "src/transformer.h"

float temperature = 0.9f;
float topp = 0.9f;
int steps = 256;

extern "C" {
    PSP_MODULE_INFO("Llama3PSP", 0, 1, 0);
    PSP_MAIN_THREAD_ATTR(THREAD_ATTR_USER | THREAD_ATTR_VFPU);
}

typedef enum {
    INITIAL,
    GENERATE,
    RESULT,
    EXIT,
} Screens;

typedef enum {
    TEMPERATURE, 
    TOPP,
    STEPS,
} ParameterSelection;

void init_app(void) {
    scePowerSetClockFrequency(333, 333, 166);
	SetupCallbacks();
	pspDebugScreenInit();
}

void print_header(AppMetrics* metrics) {
    newScreen(0, 0);
    print("Llama 3 PSP \n");
    print("Free memory: %d Kb\n", metrics->free_memory_kb);
    print("Clock: %d/%d MHz\n", metrics->cpu_clock_freq, metrics->bus_clock_freq);
    print("\n------------- CONTROLS --------------------------\n");
    print("[^][v] Select parameter | [<-][->] Change value\n");
    print("[X] Generate\n");
    print("[0] Back to Menu | [START] Exit\n");
    print("\n-------------------------------------------------\n");
}

void build(Transformer *transformer, char* checkpoint_path, Tokenizer* tokenizer, 
    char* tokenizer_path, int vocab_size,  Sampler *sampler, 
    float temperature, float topp, unsigned long long rng_seed, int steps) {

    build_transformer(transformer, checkpoint_path);
    if (steps == 0 || steps > transformer->config.sequence_len) { steps = transformer->config.sequence_len; }

    build_tokenizer(tokenizer, tokenizer_path, transformer->config.vocab_size);
    // build_sampler(sampler, transformer->config.vocab_size, temperature, topp, rng_seed);
}

void free_all(char* generated_text, Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler) {
    if (generated_text) free(generated_text);
    free_transformer(transformer);
    free_tokenizer(tokenizer);
    free_sampler(sampler);
}

void parameter_selection(ParameterSelection* selected_param) {
    if (is_btn_pressed(pad, old_pad, PSP_CTRL_UP)) {
        *selected_param = (ParameterSelection)(*selected_param - 1);
        if (*selected_param < TEMPERATURE) {
            *selected_param = STEPS;
        }
    }

    if (is_btn_pressed(pad, old_pad, PSP_CTRL_DOWN)) {
        *selected_param = (ParameterSelection)(*selected_param + 1);
        if (*selected_param > STEPS) {
            *selected_param = TEMPERATURE;
        }
    }

    switch(*selected_param) {
        case TEMPERATURE:
            if (is_btn_pressed(pad, old_pad, PSP_CTRL_LEFT)) {  temperature -= 0.05f; }
            if (is_btn_pressed(pad, old_pad, PSP_CTRL_RIGHT)) {  temperature += 0.05f; }
            if (temperature < 0.0f) temperature = 0.0f;
            if (temperature > 1.0f) temperature = 1.0f;
            break;
        case TOPP:
            if (is_btn_pressed(pad, old_pad, PSP_CTRL_LEFT))  { topp -= 0.05f; }
            if (is_btn_pressed(pad, old_pad, PSP_CTRL_RIGHT)) { topp += 0.05f; }
            if (topp < 0.0f) topp = 0.0f;
            if (topp > 1.0f) topp = 1.0f;
            break;
        case STEPS:
            if (is_btn_pressed(pad, old_pad, PSP_CTRL_LEFT))  { steps -= 8; }
            if (is_btn_pressed(pad, old_pad, PSP_CTRL_RIGHT)) { steps += 8; }
            if (steps < 8) steps = 8;
            if (steps > 512) steps = 512;
            break;
    }
}

int main(int argc, char* argv[])
{
    init_app();

    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;
    AppMetrics metrics = {0};
    Screens current_screen = INITIAL;
    ParameterSelection selected_param = TEMPERATURE;

    char *checkpoint_path = MODEL_PATH;
    char *tokenizer_path = TOKENIZER_BIN_PATH;

    char prompt_text[100] = "Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked";

    unsigned long long rng_seed = 0;
    time_t currentTime;
    char* generated_text = NULL;

    sceKernelLibcTime(&currentTime);
    rng_seed = (unsigned long long)currentTime;

    sceCtrlSetSamplingCycle(0);
    sceCtrlSetSamplingMode(PSP_CTRL_MODE_ANALOG);

    metrics.free_memory_kb = sceKernelMaxFreeMemSize() / 1024;
    metrics.cpu_clock_freq = scePowerGetCpuClockFrequencyInt();
    metrics.bus_clock_freq = scePowerGetBusClockFrequencyInt();

    print_header(&metrics);
    build(&transformer, checkpoint_path, &tokenizer, 
        tokenizer_path, transformer.config.vocab_size, &sampler, 
        temperature, topp, rng_seed, steps);
       
	while(!done) {
        old_pad = pad;
        sceCtrlReadBufferPositive(&pad, 1);

        switch (current_screen)
        {
            case INITIAL: {
                newScreen(0, 0);
                print_header(&metrics);
                parameter_selection(&selected_param);
                print("\n----------------- PARAMETERS --------------------\n");
                setTextColor(selected_param == TEMPERATURE ? COLOR_YELLOW : COLOR_WHITE);
                print("%c Temperature: %.2f\n", selected_param == TEMPERATURE ? '>' : ' ', temperature);
                
                setTextColor(selected_param == TOPP ? COLOR_YELLOW : COLOR_WHITE);
                print("%c Top-p: %.2f\n", selected_param == TOPP ? '>' : ' ', topp);

                setTextColor(selected_param == STEPS ? COLOR_YELLOW : COLOR_WHITE);
                print("%c Steps: %d\n", selected_param == STEPS ? '>' : ' ', steps);     

                setTextColor(COLOR_WHITE);
                print("\n-------------------------------------------------\n");
                print("Prompt: %s\n", prompt_text);

                if (is_btn_pressed(pad, old_pad, PSP_CTRL_CROSS)) { current_screen = GENERATE; }
                if (is_btn_pressed(pad, old_pad, PSP_CTRL_START)) { done = 1; }

                break;
            }
            case GENERATE: {
                    newScreen(0, 0);
                    print_header(&metrics);
                    
                    if (generated_text != NULL) {
                        free(generated_text);
                    }

                    time_t new_time;
                    sceKernelLibcTime(&new_time);
                    rng_seed = (unsigned long long)new_time;
                    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

                    u64 start_ticks;
                    sceRtcGetCurrentTick(&start_ticks);
                    
                    generated_text = generate(&transformer, &tokenizer, &sampler, prompt_text, steps, &metrics.generated_token_count);
                    
                    u64 end_ticks;
                    sceRtcGetCurrentTick(&end_ticks);
                    u64 tick_diff = end_ticks - start_ticks;
                    long tick_resolution = sceRtcGetTickResolution(); 
                    metrics.total_generation_time_s = (double)tick_diff / (double)tick_resolution;

                    if (metrics.total_generation_time_s > 0) {
                        metrics.tokens_per_second = (float)metrics.generated_token_count / metrics.total_generation_time_s;
                    } else { metrics.tokens_per_second = 0.0f; }

                    current_screen = RESULT;
                    break;
                }
            case RESULT: {
                    newScreen(0, 0);
                    //print_header(&metrics);
                    
                    if (generated_text != NULL) {
                            print("Prompt: %s\n\n", prompt_text);
                            print("Generated text: %s\n\n", generated_text);
                        } else { print("Error: generated text is NUll.\n"); break; }

                    print("Time: %.2f sec | Tokens/s: %.2f\n\n", metrics.total_generation_time_s, metrics.tokens_per_second);
                    print("[0] Back to Menu.\n");
                    if (is_btn_pressed(pad, old_pad, PSP_CTRL_CIRCLE)) { current_screen = INITIAL; }
                    break;
                }

                case EXIT: {
                    done = 1;
                    break;
                }

                default: {
                    current_screen = INITIAL; // in error case
                    break;
                }
            }       
            sceDisplayWaitVblankStart();
       }

    free_all(generated_text, &transformer, &tokenizer, &sampler);
    sceKernelExitGame();
    return 0;
}