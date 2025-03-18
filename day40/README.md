# Day 40: N-Body Simulation

**Objective:**
- **Implement N-Body Simulation on GPU:** Develop a CUDA C program to simulate the gravitational interactions of a large number of bodies (N-body problem) using the parallel processing power of the GPU.
- **Understand Gravitational Interactions:** Learn about the fundamental physics of gravitational forces between multiple bodies.
- **Utilize Shared Memory for Optimization:** Implement a CUDA kernel that effectively uses shared memory to reduce global memory access and improve the performance of the simulation.
- **Explore Parallel Algorithms for Physics Simulation:** Gain experience in designing parallel algorithms for simulating physical systems on the GPU.

**Key Learnings:**
- **N-Body Simulation:**
    - **Gravitational Force Calculation:** Understood how to calculate the gravitational force between every pair of bodies in the simulation using Newton's Law of Universal Gravitation.
    - **Time Integration:** Learned about the basic time integration step used to update the positions and velocities of the bodies based on the calculated forces.
- **CUDA Implementation for Physics:**
    - **Kernel Design for Pairwise Interactions:** Developed a CUDA kernel (`nbody`) that efficiently calculates the forces between all pairs of particles in parallel.
    - **Shared Memory Optimization:** Implemented a tiling strategy using shared memory (`sdata`, `tileA`, `tileB`) to store the positions and masses of subsets of particles, enabling faster access for force calculations within each thread block. This significantly reduces the number of accesses to slower global memory.
    - **Thread and Block Organization for N-Body:** Configured the thread and block dimensions to effectively distribute the workload of calculating pairwise forces across the GPU's parallel architecture.
    - **`extern __shared__`:** Learned how to declare dynamically sized shared memory within a CUDA kernel.
- **Performance Considerations:**
    - **Computational Complexity:** Understood the O(N^2) computational complexity of the direct summation N-body simulation and how GPUs can help manage this for large N.
    - **Memory Access Patterns:** Recognized the importance of optimizing memory access patterns on the GPU, which is the primary motivation for using shared memory in this implementation.

**Code Implementation Details:**

- **Includes and Defines:**
    - `cstdio`, `cstdlib`, `cuda_runtime.h`, `math.h`, `time.h`: Standard C/CUDA libraries for input/output, memory management, mathematical functions, and time functions.
    - `M_PI`: Defines the value of Pi.
    - `THREADS`: Defines the number of threads per block (1024 in this case).
    - `N`: Defines the number of bodies to simulate (2^16 = 65536).
    - `G`: Defines the gravitational constant.
- **`nbody` Global Function:**
    - **Shared Memory Allocation:** Declares a dynamically sized shared memory array `sdata`.
    - **Tiling Strategy:** Divides the particles into tiles that fit into shared memory. The kernel iterates through these tiles.
    - **Loading Tiles into Shared Memory:** Each thread block loads a tile of particle data (position and mass) into shared memory. Double buffering with `tileA` and `tileB` is used to overlap computation with data loading for the next tile.
    - **Force Calculation:** For each particle `i`, each thread calculates the gravitational force exerted on it by all other particles within the currently loaded tile from shared memory. It avoids calculating the force of a particle on itself.
    - **Position and Velocity Update (Semi-Implicit Euler):** Implements a basic semi-implicit Euler method for time integration:
        - First, it updates the position of each particle based on its current velocity and the calculated acceleration from the previous step.
        - Then, it calculates the new acceleration based on the updated positions.
        - Finally, it updates the velocity based on the average of the old and new accelerations.
- **`init` Function:** Initializes the positions, velocities, and masses of the N bodies. It uses random values for most particles but sets up a small initial configuration of a central heavy body and a few lighter bodies orbiting it for the first few particles (if N > 5).
- **`main` Function:**
    - Defines the size of the data arrays.
    - Allocates memory on the GPU using `cudaMallocManaged` for the positions (x, y, z), masses (m), and velocities (vx, vy, vz) of all N bodies.
    - Initializes the particle data using the `init` function.
    - Prints the initial state of the first 5 bodies.
    - Sets the simulation time step (`dt`) and the number of simulation steps (`steps`).
    - Calculates the required shared memory size for the kernel.
    - Records the start time using CUDA events.
    - Executes the main simulation loop, launching the `nbody` kernel for each time step.
    - Synchronizes the device after each kernel launch.
    - Records the stop time and calculates the total and average step time using CUDA events.
    - Prints the final state of the first 5 bodies.
    - Frees the allocated memory on the GPU and destroys the CUDA events.

**Output:**
Initial sample:

Body 0: pos=(0.0000,0.0000,0.0000), vel=(0.000000,0.000000,0.000000), mass=100000.00

Body 1: pos=(-0.0000,0.5000,0.0406), vel=(-0.005000,-0.000000,0.000000), mass=25523.34

Body 2: pos=(-0.5000,-0.0000,-0.0499), vel=(0.000000,-0.005000,0.000000), mass=18855.96

Body 3: pos=(0.0000,-0.5000,-0.0104), vel=(0.005000,0.000000,0.000000), mass=19458.86

Body 4: pos=(0.5000,0.0000,0.0172), vel=(-0.000000,0.005000,0.000000), mass=16307.56




Finished 100 steps in 3713.542 ms (37.135 ms/step)




Final sample:

Body 0: pos=(-0.0000,0.0000,0.0000), vel=(-0.000053,0.000001,0.000056), mass=100000.00

Body 1: pos=(-0.0050,0.4997,0.0405), vel=(-0.004997,-0.000679,-0.000183), mass=25523.34

Body 2: pos=(-0.4997,-0.0050,-0.0499), vel=(0.000651,-0.005091,0.000047), mass=18855.96

Body 3: pos=(0.0050,-0.4997,-0.0104), vel=(0.004952,0.000584,0.000070), mass=19458.86

Body 4: pos=(0.4996,0.0050,0.0172), vel=(-0.000740,0.005039,-0.000001), mass=16307.56
