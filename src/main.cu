#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <cuda_runtime.h>
#include "main.h"

#define USE_CUDA true
#define SAVE_OUTPUT false
#define MAX_STEP 2000

using namespace std;

int main(){

    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "could not initialize sdl2: %s\n", SDL_GetError());
        return 1;
    }
    if(SDL_CreateWindowAndRenderer(SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_RESIZABLE, &window, &renderer)){
        return 3;
    }
    if (window == NULL) {
        fprintf(stderr, "could not create window: %s\n", SDL_GetError());
        return 1;
    }
    bool run = true;
    SDL_Event event;

    Pendulum *pendulums = (Pendulum*)malloc(N * sizeof(Pendulum));
    Pendulum *d_pendulums;


    for (int i = 0; i < N; i++) {
        int x = i % SCREEN_WIDTH;
        int y = i / SCREEN_WIDTH;
        pendulums[i] = Pendulum(
            (x - SCREEN_WIDTH/2) * 2 * PI / SCREEN_WIDTH, 
            (y - SCREEN_HEIGHT/2) * 2 * PI / SCREEN_HEIGHT
        );
    }
    int blockSize, gridSize;
    if (USE_CUDA) {
        cudaMalloc(&d_pendulums, N * sizeof(Pendulum));
        blockSize = 1024;
        gridSize = (int)ceil((float)(N/blockSize));
        cudaMemcpy(d_pendulums, pendulums, N * sizeof(Pendulum), cudaMemcpyHostToDevice);
    }

    int step = 0;
    while(run && step < MAX_STEP){
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);

        if (USE_CUDA) {
            computeAccelerationsCUDA<<<gridSize, blockSize>>>(d_pendulums, N);
            updatePendulumsCUDA<<<gridSize, blockSize>>>(d_pendulums, N);
            cudaMemcpy(pendulums, d_pendulums, N * sizeof(Pendulum), cudaMemcpyDeviceToHost);
        } else {
            for (int i = 0; i < N; i++) {
                computeAccelerations(&pendulums[i]);
                updatePendulums(&pendulums[i]);
            }
        }

        //computePositions(&pendulums[12300]);
        //drawPendulum(&pendulums[12300], renderer);
        drawFractal(pendulums, renderer);
        SDL_RenderPresent(renderer);
        if(SDL_PollEvent(&event) && event.type == SDL_QUIT){
            run = false;
        }

        if (SAVE_OUTPUT){
            saveScreenshot(step, renderer);
        }
        step++;
    }
    SDL_DestroyWindow(window);
    SDL_Quit();
    free(pendulums);
    cudaFree(d_pendulums);
    return 0;
}

void computePositions(Pendulum *pendulum) {
    pendulum->x1 = pendulum->x0 + pendulum->r1 * sin(pendulum->o1);
    pendulum->y1 = pendulum->y0 + pendulum->r1 * cos(pendulum->o1);
    pendulum->x2 = pendulum->x1 + pendulum->r2 * sin(pendulum->o2);
    pendulum->y2 = pendulum->y1 + pendulum->r2 * cos(pendulum->o2);
}

void computeAccelerations(Pendulum *pendulum) {

    pendulum->o1_a = (
        -g*(2*pendulum->m1+pendulum->m2)*sin(pendulum->o1) 
        -pendulum->m2*g*sin(pendulum->o1-2*pendulum->o2)
        -2*sin(pendulum->o1-pendulum->o2)*pendulum->m2
        *(pow(pendulum->o2_v,2)*pendulum->r2 + pow(pendulum->o1_v,2)*pendulum->r1*cos(pendulum->o1-pendulum->o2)))
        /(pendulum->r1*(2*pendulum->m1+pendulum->m2-pendulum->m2*cos(2*pendulum->o1-2*pendulum->o2)));

    pendulum->o2_a = (
        2*sin(pendulum->o1-pendulum->o2)
        *(
            pow(pendulum->o1_v,2)*pendulum->r1*(pendulum->m1+pendulum->m2)
            +g*(pendulum->m1+pendulum->m2)*cos(pendulum->o1)
            +pow(pendulum->o2_v,2)*pendulum->r2*pendulum->m2*cos(pendulum->o1-pendulum->o2)
        )/(pendulum->r2*(2*pendulum->m1+pendulum->m2-pendulum->m2*cos(2*pendulum->o1-2*pendulum->o2)))
    );
}

__global__
void computeAccelerationsCUDA(Pendulum *pendulums, int n) {

    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < n) {
        pendulums[id].o1_a = (
            -g*(2*pendulums[id].m1+pendulums[id].m2)*sinf(pendulums[id].o1) 
            -pendulums[id].m2*g*sinf(pendulums[id].o1-2*pendulums[id].o2)
            -2*sinf(pendulums[id].o1-pendulums[id].o2)*pendulums[id].m2
            *(powf(pendulums[id].o2_v,2)*pendulums[id].r2 + powf(pendulums[id].o1_v,2)*pendulums[id].r1*cosf(pendulums[id].o1-pendulums[id].o2)))
            /(pendulums[id].r1*(2*pendulums[id].m1+pendulums[id].m2-pendulums[id].m2*cosf(2*pendulums[id].o1-2*pendulums[id].o2)));

        pendulums[id].o2_a = (
            2*sinf(pendulums[id].o1-pendulums[id].o2)
            *(
                powf(pendulums[id].o1_v,2)*pendulums[id].r1*(pendulums[id].m1+pendulums[id].m2)
                +g*(pendulums[id].m1+pendulums[id].m2)*cosf(pendulums[id].o1)
                +powf(pendulums[id].o2_v,2)*pendulums[id].r2*pendulums[id].m2*cosf(pendulums[id].o1-pendulums[id].o2)
            )/(pendulums[id].r2*(2*pendulums[id].m1+pendulums[id].m2-pendulums[id].m2*cosf(2*pendulums[id].o1-2*pendulums[id].o2)))
        );
    }

}

void updatePendulums(Pendulum *pendulum) {
    pendulum->o1_v += pendulum->o1_a;
    pendulum->o2_v += pendulum->o2_a;
    pendulum->o1 += pendulum->o1_v;
    pendulum->o2 += pendulum->o2_v;

    pendulum->o1_v *= drag;
    pendulum->o2_v *= drag;
}

__global__
void updatePendulumsCUDA(Pendulum *pendulums, int n) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id < n) {
        pendulums[id].o1_v += pendulums[id].o1_a;
        pendulums[id].o2_v += pendulums[id].o2_a;
        pendulums[id].o1 += pendulums[id].o1_v;
        pendulums[id].o2 += pendulums[id].o2_v;

        pendulums[id].o1_v *= drag;
        pendulums[id].o2_v *= drag;
    }
}

Uint32 couleur(Uint8 r, Uint8 v, Uint8 b, Uint8 a){
   return (int(a)<<24) + (int(b)<<16) + (int(v)<<8) + int(r);
}

Uint32 pickColor(float o1, float o2) {
    int radius = 127;
    return couleur(
        127 + radius * cos(o1) * sin(o2),
        127 + radius * sin(o1) * sin(o2),
        127 + radius * cos(o2),
        255
    );
}

void drawPendulum(Pendulum *pendulum, SDL_Renderer *renderer) {
    Uint32 color = pickColor(pendulum->o1, pendulum->o2);
    aalineColor(renderer, pendulum->x0, pendulum->y0, pendulum->x1, pendulum->y1, color);
    filledCircleColor(renderer, pendulum->x1, pendulum->y1, pendulum->m1, color);
    aalineColor(renderer, pendulum->x1, pendulum->y1, pendulum->x2, pendulum->y2, color);
    filledCircleColor(renderer, pendulum->x2, pendulum->y2, pendulum->m2, color);
}

void drawFractal(Pendulum *pendulums, SDL_Renderer *renderer) {
    for (int i = 0; i < N; i++) {
        int x = i % SCREEN_WIDTH;
        int y = i / SCREEN_WIDTH;
        Uint32 color = pickColor(pendulums[i].o1, pendulums[i].o2);
        pixelColor(renderer, y, x, color);
    }
}

void saveScreenshot(int id, SDL_Renderer *renderer){
    const Uint32 format = SDL_PIXELFORMAT_ARGB8888;
    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, SCREEN_WIDTH, SCREEN_HEIGHT, 32, format);
    SDL_RenderReadPixels(renderer, NULL, format, surface->pixels, surface->pitch);
    SDL_SaveBMP(surface, ("renders/" + std::to_string(id) + ".bmp").c_str());
    SDL_FreeSurface(surface);
}