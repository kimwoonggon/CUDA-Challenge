# Day 36: GPU-Accelerated Ray Tracing of an Animated Sphere with PyCUDA

**Objective:**
- **Implement Basic Ray Tracing on GPU:** Develop a PyCUDA program that uses a custom CUDA kernel to perform basic ray tracing to render a sphere.
- **Understand Ray-Sphere Intersection:** Learn the fundamental concept of ray-sphere intersection, a core component of many ray tracing algorithms.
- **Animate the Scene:** Make the rendered sphere move over time by updating its position in the CUDA kernel, and save the animation as a GIF.
- **Visualize Ray Tracing Results:** Display the rendered animation directly within the README.

**Key Learnings:**
- **Ray Tracing Fundamentals:**
    - **Ray Casting:** Understood the basic principle of ray tracing: casting rays from a virtual camera through each pixel of the screen into the scene to determine what object (if any) is visible at that pixel.
    - **Camera and Rays:** Learned how to define a simple camera position and generate ray directions for each pixel based on image dimensions and an aspect ratio.
- **Ray-Sphere Intersection:**
    - **Mathematical Formula:** Implemented the mathematical formula to determine if a ray intersects with a sphere. This involves solving a quadratic equation derived from the ray and sphere equations.
    - **Hit Time (`t_hit`):** Understood that the solution(s) to the quadratic equation represent the "hit times" along the ray where an intersection occurs. Only positive hit times indicate intersections in front of the camera.
- **Basic Shading (Lambertian):**
    - **Surface Normals:** Calculated the normal vector at the point of intersection on the sphere's surface. The normal is a vector perpendicular to the surface at that point and is crucial for shading.
    - **Lambertian Reflection:** Implemented a simple Lambertian shading model, where the brightness of a surface point depends on the angle between the surface normal and the direction of the light source.
- **CUDA Kernel for Ray Tracing (`render`):**
    - **Pixel-Parallel Rendering:** Developed a `render` CUDA kernel where each thread calculates the color of a single pixel by casting a ray and performing the intersection and shading calculations.
    - **Camera Setup:** Defined a basic camera position and generated ray directions based on the pixel's screen coordinates, incorporating an aspect ratio correction.
    - **Sphere Definition:** Defined the sphere's center and radius. The sphere's center is animated using sine and cosine functions based on the `time` parameter.
    - **Background Color:** Set a background color for pixels where no sphere is hit.
    - **Intersection Test:** Called the `intersect` device function to check for ray-sphere intersection.
    - **Color Calculation:** If an intersection occurs, calculated the surface normal, defined a light direction, and used Lambertian shading to determine the pixel's color.
    - **Output to Display Array:** Wrote the calculated RGB color values to the `disp` array.
- **Device Function (`intersect`):**
    - Implemented a CUDA device function to encapsulate the ray-sphere intersection logic, making the `render` kernel cleaner and more organized.
- **Animation with Matplotlib's `FuncAnimation`:**
    - Similar to Day 35, used `FuncAnimation` to update the scene by changing the `time` variable passed to the CUDA kernel, resulting in the sphere's animated movement.
- **Saving Animation as GIF:**
    - Saved the generated frames as a GIF file named `intersection.gif`.

**Code Implementation Details:**

- **CUDA Kernel (`GPUkernel` string):**
    - **`intersect` Device Function:** Takes ray origin (`ro`), ray direction (`rd`), sphere center (`sphere_c`), sphere radius (`sphere_r`), and an output parameter for the hit time (`t_hit`). Returns `true` if an intersection occurs, `false` otherwise.
    - **`render` Kernel:** Takes `disp`, `img_w`, `img_h`, and `time` as input.
        - Calculates ray origin and direction for each pixel.
        - Defines the animated sphere.
        - Calls `intersect` to check for intersection.
        - Calculates color based on intersection and Lambertian shading.
        - Writes color to the output display array.
- **Python Script:**
    - **Import Libraries:** Imports NumPy, PyCUDA, Matplotlib.
    - **Compile CUDA Kernel:** Compiles the `GPUkernel` string.
    - **Get Kernel Function:** Retrieves the `render` kernel.
    - **Image Dimensions and Array Initialization:** Sets up image dimensions and the display array.
    - **Thread and Block Configuration:** Defines thread and block sizes.
    - **Matplotlib Setup:** Creates a figure and an `imshow` plot.
    - **Animation Function (`animate`):**
        - Updates the `time` variable.
        - Launches the `render` kernel.
        - Reshapes the output array for display.
        - Updates the `imshow` plot.
    - **Create and Save Animation:** Uses `FuncAnimation` to generate and save the animation as `intersection.gif` in the `/mnt/d/CUDA/day36/` directory.

**Output:**
- **`intersection.gif`:** A GIF animation file is generated in the same directory as this README.md. The animation, embedded below, shows a red sphere moving in a sinusoidal path against a black background, with basic shading applied.

<img src="intersection.gif">

**Conclusion:**
- **Basic Ray Tracing on GPU:** Successfully implemented a fundamental ray tracing algorithm on the GPU using PyCUDA to render a sphere.
- **Understanding Ray-Sphere Intersection:** Gained practical experience with the ray-sphere intersection calculation.
- **Animated Scene Rendering:** Demonstrated the ability to animate objects in a GPU-rendered scene by updating parameters in the CUDA kernel over time.
