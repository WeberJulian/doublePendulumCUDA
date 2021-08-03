# Double Pendulum fractal

## Result

![](render.gif)

## How it works
Each pixel correspond to a double pendulum and the pixel coordinate (x,y) are maped to initial angles (-pi,pi). 

The function `Uint32 pickColor(float o1, float o2)` associate to a combination of angle, an intresting color.

We then for each timestamp compute the physics update on the GPU with CUDA.


## Inspiration

<iframe width="560" height="315"
src="https://www.youtube.com/embed/n7JK4Ht8k8M" 
frameborder="0" 
allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
allowfullscreen></iframe>