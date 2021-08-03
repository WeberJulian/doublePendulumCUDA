#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>

using namespace std;

#define SCREEN_WIDTH 1200
#define SCREEN_HEIGHT 640
#define PI 3.14159265
#define g 1.0

class Pendulumn {
    public:
        float r1 = 200;
        float r2 = 200;
        float m1 = 20;
        float m2 = 20;
        float x0 = SCREEN_WIDTH/2;
        float y0 = 30;
        float x1 = 0;
        float y1 = 0;
        float x2 = 0; 
        float y2 = 0;
        float o1 = 0;
        float o2 = 0;
        float o1_v = 0;
        float o2_v = 0;
        float o1_a = 0;
        float o2_a = 0;

        Pendulumn(float o1, float o2) {
            this->o1 = o1;
            this->o2 = o2;
        }
};

void computePositions(Pendulumn *pendulum) {
    pendulum->x1 = pendulum->x0 + pendulum->r1 * sin(pendulum->o1);
    pendulum->y1 = pendulum->y0 + pendulum->r1 * cos(pendulum->o1);
    pendulum->x2 = pendulum->x1 + pendulum->r2 * sin(pendulum->o2);
    pendulum->y2 = pendulum->y1 + pendulum->r2 * cos(pendulum->o2);
}

void computeAccelerations(Pendulumn *pendulum) {

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
    Pendulumn pendulum(PI/2, PI/8);
    while(run){
        SDL_SetRenderDrawColor( renderer, 0, 0, 0, 0 );
		SDL_RenderClear( renderer );

        computeAccelerations(&pendulum);
        pendulum.o1_v += pendulum.o1_a;
        pendulum.o2_v += pendulum.o2_a;
        pendulum.o1 += pendulum.o1_v;
        pendulum.o2 += pendulum.o2_v;

        pendulum.o1_v *= 0.9999;
        pendulum.o2_v *= 0.9999;

        computePositions(&pendulum);
        aalineRGBA(renderer, pendulum.x0, pendulum.y0, pendulum.x1, pendulum.y1, 255, 255, 255, 255);
        filledCircleRGBA(renderer, pendulum.x1, pendulum.y1, pendulum.m1, 255, 255, 255, 255);
        aalineRGBA(renderer, pendulum.x1, pendulum.y1, pendulum.x2, pendulum.y2, 255, 255, 255, 255);
        filledCircleRGBA(renderer, pendulum.x2, pendulum.y2, pendulum.m2, 255, 255, 255, 255);
        SDL_RenderPresent(renderer);
        SDL_Delay(20);
        if(SDL_PollEvent(&event) && event.type == SDL_QUIT){
            run = false;
        }
        
    }
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}