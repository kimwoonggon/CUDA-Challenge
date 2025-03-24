# Day 43: 2D Wave Equation Simulation

**Objective:**
- **Simulate the 2D Wave Equation on GPU:** Develop a CUDA C++ program to simulate the propagation of waves in a 2D medium using the finite difference method, accelerated by the GPU.
- **Understand Finite Difference Method for PDEs:** Learn how to discretize a partial differential equation (PDE), specifically the wave equation, using finite differences in both space and time.
- **Implement CUDA for Grid-Based Simulations:** Gain experience in using CUDA to perform parallel computations on a 2D grid, which is common in many scientific simulation tasks.
- **Visualize Wave Propagation:** Observe the numerical simulation of a wave propagating through a medium based on initial conditions.

**Key Learnings:**
- **2D Wave Equation:**
    - Understood the mathematical form of the 2D wave equation, which describes how waves propagate through space and time.
    - Recognized the key parameters involved, such as the wave speed (related to `C`).
- **Finite Difference Method:**
    - **Discretization:** Learned how to approximate the continuous derivatives in the wave equation using discrete differences on a spatial and temporal grid.
    - **Stencil Operation:** Identified the stencil (the neighboring grid points used in the update rule) for the 2D Laplacian operator.
    - **Time Stepping:** Understood the concept of advancing the solution in discrete time steps, where the state at the next time step depends on the states at the current and previous time steps.
- **CUDA for Parallel Simulation:**
    - **Kernel Design for Grid Updates:** Developed a CUDA kernel (`wave_step`) to update the wave displacement at each grid point in parallel.
    - **Thread and Block Mapping to Grid:** Configured the thread and block dimensions to map each GPU thread to a specific cell in the 2D grid.
    - **Boundary Conditions:** Implemented simple boundary conditions where the edges of the grid are not updated (effectively fixed boundaries).
    - **Data Management for Time Stepping:** Used three arrays (`u_prev`, `u_curr`, `u_next`) to store the wave displacement at the previous, current, and next time steps, and efficiently swapped them in each iteration.
- **Simulation Parameters:**
    - Understood the role of parameters like the grid size (`N`), time step (`dt`), spatial step (`dx`), and the constant `C` in the stability and behavior of the wave simulation.

**Code Implementation Details:**

- **Includes:**
    - `cstdio`: Standard input/output library.
    - `cstdlib`: Standard general utilities library.
    - `cuda_runtime.h`: CUDA runtime API.
    - `math.h`: Mathematical functions.
- **Defines:**
    - `N`: Size of the 2D grid (N x N).
    - `THREADS`: Number of threads per block.
    - `BLOCKS`: Number of thread blocks required to cover the grid.
    - `STEPS`: Number of time steps to simulate.
    - `C`: A constant related to the wave speed.
- **`wave_step` Global Function:**
    - **Thread Mapping:** Calculates the 2D index (`i`, `j`) of the grid point corresponding to the current thread.
    - **Boundary Check:** Ensures that the computation is performed only for the interior grid points (excluding the boundaries).
    - **Laplacian Calculation:** Approximates the 2D Laplacian of the wave displacement using the values of the four neighboring cells (`up`, `down`, `left`, `right`).
    - **Time Update:** Implements the finite difference formula to calculate the wave displacement at the next time step (`u_next[idx]`) based on the current (`u_curr[idx]`) and previous (`u_prev[idx]`) values, and the calculated Laplacian.
- **`init` Function:** Initializes the wave displacement arrays (`u_prev`, `u_curr`) to zero. It then sets an initial impulse at the center of the grid by assigning a non-zero amplitude and velocity to the central grid point.
- **`main` Function:**
    - Defines the size of the grid and allocates managed memory on the GPU for the three wave displacement arrays (`u_prev`, `u_curr`, `u_next`).
    - Sets the simulation parameters: time step (`dt`), spatial step (`dx`), and calculates `dx2`.
    - Initializes the wave using the `init` function.
    - Records the start time using CUDA events.
    - Executes the main simulation loop for the specified number of steps:
        - Launches the `wave_step` kernel on the GPU.
        - Synchronizes the device after the kernel launch.
        - Swaps the roles of the arrays (`u_prev`, `u_curr`, `u_next`) to prepare for the next time step. This efficient swapping avoids unnecessary data copying.
    - Records the stop time and calculates the total and average step time using CUDA events.
    - Prints the total simulation time and the average time per step.
    - Prints a small sample of the wave displacement values around the center of the grid after the simulation.
    - Frees the allocated memory on the GPU and destroys the CUDA events.
