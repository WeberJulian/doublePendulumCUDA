#define SCREEN_WIDTH 512
#define SCREEN_HEIGHT 512
#define N SCREEN_HEIGHT * SCREEN_WIDTH
#define PI 3.14159265
#define g 1.0
#define drag 1
#define dt 1
#define radius 127

class Pendulum {
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

        Pendulum(float o1, float o2) {
            this->o1 = o1;
            this->o2 = o2;
        }
};

time_t get_ms_now();
void computePositions(Pendulum *pendulum);
void computeAccelerations(Pendulum *pendulum);
__global__ void computeAccelerationsCUDA(Pendulum *pendulums, int n);
void updatePendulums(Pendulum *pendulum);
__global__ void updatePendulumsCUDA(Pendulum *pendulums, int n);
Uint32 couleur(Uint8 r, Uint8 v, Uint8 b, Uint8 a);
Uint32 pickColor(float o1, float o2);
void drawPendulum(Pendulum *pendulum, SDL_Renderer *renderer);
void drawFractal(Pendulum *pendulums, SDL_Renderer *renderer);
void saveScreenshot(int id, SDL_Renderer *renderer);
__global__ void drawFractalTextureCUDA(unsigned char *pixels, Pendulum *pendulums, int n);
void drawFractalTexture(unsigned char *pixels, Pendulum *pendulums);