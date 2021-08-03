# Double Pendulum fractal

## Result

![](render.gif)

## How it works
Each pixel correspond to a double pendulum and the pixel coordinate (x,y) are maped to initial angles (-pi,pi). 

The function `Uint32 pickColor(float o1, float o2)` associate to a combination of angle, an intresting color.

We then for each timestamp compute the physics update on the GPU with CUDA.


## Inspiration

Compared to the original implementation, I added a dampening factor to the simulation and add support for CUDA.

[![inspiration](https://img.youtube.com/vi/n7JK4Ht8k8M/0.jpg)](https://www.youtube.com/watch?v=n7JK4Ht8k8M)

## Benchmark

| device  | precision | time
| ------------- | ------------- |------------- |
| 5900x  | fp32  | 57.733s
| RTX 3090  | fp32  | 113.067s