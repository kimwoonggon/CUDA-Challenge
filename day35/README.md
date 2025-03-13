# Day 35: GPU-Accelerated Animation with PyCUDA

**Objective:**
- **Create a GPU-Rendered Animation:**  Develop a PyCUDA program that uses a custom CUDA kernel to generate frames for an animation entirely on the GPU.
- **Visualize GPU Computation:** Demonstrate how to leverage the GPU for real-time rendering and visualization tasks, moving beyond pure computation to visual output.
- **Generate GIF Animation:** Learn to save the GPU-rendered animation frames as a GIF file for easy sharing and viewing, and embed it directly in the README.

**Key Learnings:**
- **PyCUDA for Graphics/Visualization:**
    - **Using PyCUDA for Rendering:** Explored how PyCUDA can be used not just for general-purpose GPU computing but also for graphics and visualization tasks by directly controlling pixel values within a CUDA kernel.
    - **Interfacing PyCUDA with Matplotlib:**  Learned to combine PyCUDA with Matplotlib's animation functionalities to visualize the results of GPU computations in a dynamic way.
- **CUDA Kernel for Image Generation (`render`):**
    - **Pixel-Parallel Rendering:** Implemented a `render` CUDA kernel that performs pixel-parallel image generation. Each CUDA thread is responsible for calculating the color value of a single pixel in the output image.
    - **Procedural Image Generation:** Used mathematical functions (sin, sqrt) within the kernel to procedurally generate image patterns based on pixel coordinates and time. This demonstrates a common technique in computer graphics for creating dynamic and interesting visuals algorithmically.
    - **Time-Based Animation:** Incorporated a `time` parameter into the kernel's calculations, allowing the generated image to change over time, thus creating animation frames.
- **Animation with Matplotlib's `FuncAnimation`:**
    - **Dynamic Plotting:** Utilized Matplotlib's `FuncAnimation` to create an animation by repeatedly calling a function (`animate` in this case) that updates the image data.
    - **Blitting for Performance:** Employed `blit=True` in `FuncAnimation` to optimize the animation update process by only redrawing the parts of the plot that have changed (the image itself), resulting in smoother and more efficient animation.
- **Saving Animation as GIF:**
    - **Using Pillow Writer:**  Learned how to save a Matplotlib animation as a GIF file using the 'pillow' writer, which is a common and effective method for creating GIF animations from Matplotlib figures.
    - **GIF File Output and Embedding:** Generated `cuda_animation.gif`, showcasing the animated output of the GPU-based rendering. This animation is embedded below.

**Code Implementation Details:**

- **CUDA Kernel (`GPUkernel` string):**
    - **`render` Kernel:** Takes `disp` (output display array), `img_w`, `img_h`, and `time` as input.
    - **Pixel Coordinate Calculation:** Calculates normalized pixel coordinates `x` and `y` (from 0 to 1) based on thread indices and image dimensions.
    - **Wave Interference Pattern:** Generates two sinusoidal waves (`wave1`, `wave2`) that are modulated by time and pixel position.
    - **Color Generation:** Combines the waves to create an "interference" pattern and maps this pattern to RGB color channels to produce visually dynamic colors.
    - **Output to Display Array:** Writes the calculated RGB color values to the `disp` array at the appropriate pixel location.
- **Python Script:**
    - **Import Libraries:** Imports NumPy, PyCUDA (driver, autoinit, compiler), Matplotlib (pyplot, animation).
    - **Compile CUDA Kernel:** Compiles the `GPUkernel` string using `SourceModule`.
    - **Get Kernel Function:** Retrieves the `render` kernel function using `module.get_function("render")`.
    - **Image Dimensions and Array Initialization:** Sets image width (`img_w`), height (`img_h`), calculates the number of pixels (`n_pix`), and initializes a NumPy array `disp` to store the pixel data (RGB float32 format).
    - **Thread and Block Configuration:** Defines `threads` (block dimensions) and calculates `blocks` (grid dimensions) for kernel launch, ensuring coverage of all pixels.
    - **Matplotlib Setup:** Creates a Matplotlib figure and axes, and initializes an `imshow` plot to display the image data.
    - **Animation Function (`animate`):**
        - Takes the frame number as input.
        - Calculates the `time` value based on the frame number.
        - Launches the `render` CUDA kernel, passing the `disp` array, image dimensions, and `time` as arguments.
        - Reshapes the `disp` array into a `(img_h, img_w, 3)` format for Matplotlib.
        - Updates the image data in the `img_plot` using `img_plot.set_array()`.
        - Returns `[img_plot]` for `FuncAnimation`.
    - **Create and Save Animation:** Uses `FuncAnimation` to create the animation by calling the `animate` function repeatedly. Saves the animation as a GIF file named `cuda_animation.gif` in the `/mnt/d/CUDA/day35/` folder (as specified in the code) using the 'pillow' writer. Sets `fps=20` for 20 frames per second animation speed.

**Output:**
- The animation, embedded below, displays a dynamically changing interference pattern of sinusoidal waves, rendered entirely on the GPU using the custom CUDA kernel.

<img src="cuda_animation.gif">

**Conclusion:**
- **GPU-Accelerated Animation Achieved:** Successfully created a GPU-accelerated animation using PyCUDA, demonstrating the capability to render dynamic visuals using CUDA kernels.
- **PyCUDA and Matplotlib Integration:** Effectively integrated PyCUDA with Matplotlib's animation tools to visualize GPU computations in motion.
- **Pixel-Parallel Rendering and Procedural Generation:**  Showcased pixel-parallel rendering and procedural image generation techniques within a CUDA kernel, opening possibilities for more complex GPU-based visual effects and simulations.
- This project illustrates the exciting potential of combining GPU computing with visualization techniques to create dynamic and visually rich applications.
