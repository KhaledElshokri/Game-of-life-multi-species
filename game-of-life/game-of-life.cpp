#include <windows.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <cassert>

//
// Configuration
//
const int WIN_W = 1024;
const int WIN_H = 768;
const int GRID_W = WIN_W;
const int GRID_H = WIN_H;
const size_t GRID_N = (size_t)GRID_W * (size_t)GRID_H;

// choose species count 5 to 10 randomly at start
std::mt19937 rng_seed((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
std::uniform_int_distribution<int> d_spec(5, 10);
const int NUM_SPECIES = d_spec(rng_seed);

inline size_t idx(size_t x, size_t y) { return y * (size_t)GRID_W + x; }

static void check_cl(cl_int err, const char* msg) {
  if (err != CL_SUCCESS) {
    std::cerr << "OpenCL Error (" << err << "): " << msg << std::endl;
    exit(EXIT_FAILURE);
  }
}

GLuint compile_shader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char buf[2048]; buf[0] = 0;
    glGetShaderInfoLog(s, 2048, NULL, buf);
    std::cerr << "Shader compile error: " << buf << std::endl;
  }
  return s;
}

const char* vshader_src = R"(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

const char* fshader_src = R"(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform usampler2D uTex;
void main(){
    uvec4 uc = texture(uTex, vUV);
    vec4 c = vec4(uc) / 255.0;
    FragColor = c;
}
)";


// OpenCL kernel = compute next generation and write colored pixel to output image
const char* cl_kernel_src = R"CLC(
__kernel void gol_sim_and_color(
    const int width,
    const int height,
    const int num_species,
    const uint frame_seed,
    __global const uchar* grid_in,
    __global uchar* grid_out,
    write_only image2d_t out_tex
){
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    if (gx >= width || gy >= height) return;
    int index = gy * width + gx;
    uchar cur = grid_in[index];

    // neighbor counts for species 0..num_species
    uchar counts[11];
    for (int i=0;i<=num_species;i++) counts[i] = 0;

    int x0 = (gx>0 ? gx-1 : 0);
    int x1 = (gx<width-1 ? gx+1 : width-1);
    int y0 = (gy>0 ? gy-1 : 0);
    int y1 = (gy<height-1 ? gy+1 : height-1);

    for (int yy=y0; yy<=y1; ++yy) {
        int row = yy * width;
        for (int xx=x0; xx<=x1; ++xx) {
            if (xx==gx && yy==gy) continue;
            uchar s = grid_in[row + xx];
            counts[s] = counts[s] + (uchar)1;
        }
    }

    uchar candidates[11];
    uchar cand_cnt = 0;
    for (int s=1; s<=num_species; ++s) {
        uchar neighbors = counts[s];
        uchar cur_alive = (cur == (uchar)s) ? 1 : 0;
        uchar next_alive = 0;
        if (cur_alive) {
            next_alive = (neighbors == 2 || neighbors == 3) ? 1 : 0;
        } else {
            next_alive = (neighbors == 3) ? 1 : 0;
        }
        if (next_alive) {
            candidates[cand_cnt++] = (uchar)s;
        }
    }

    uchar final_species = 0;
    if (cand_cnt > 0) {
        uint state = (uint)(gx * 73856093u ^ gy * 19349663u ^ frame_seed);
        // simple LCG
        state = (1103515245u * state + 12345u);
        uint pick = state % cand_cnt;
        final_species = candidates[pick];
    }

    // write grid_out
    grid_out[index] = final_species;

    // palette (same order as host)
    const uchar palette[11][3] = {
        {0,0,0}, {255,0,0}, {0,255,0}, {0,0,255}, {255,255,0},
        {255,0,255}, {0,255,255}, {255,165,0}, {128,0,128}, {192,192,192}, {255,255,255}
    };


    uint4 color = (uint4)(
        (uint)palette[final_species][0],
        (uint)palette[final_species][1],
        (uint)palette[final_species][2],
        255u
    );
    write_imageui(out_tex, (int2)(gx, gy), color);
}
)CLC";

int main() {
  // --- GLFW + OpenGL init ---
  if (!glfwInit()) {
    std::cerr << "GLFW init failed\n";
    return -1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H, "GOL OpenCL-OpenGL Shared (Option B)", NULL, NULL);
  if (!window) {
    std::cerr << "Window creation failed\n";
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0); // no vsync

  if (glewInit() != GLEW_OK) {
    std::cerr << "GLEW init failed\n";
    return -1;
  }


  // Setup full-screen quad
  float quadVerts[] = {
      -1.f,-1.f, 0.f,0.f,
       1.f,-1.f, 1.f,0.f,
       1.f, 1.f, 1.f,1.f,
      -1.f, 1.f, 0.f,1.f
  };
  unsigned int quadIdx[] = { 0,1,2,2,3,0 };
  GLuint vao, vbo, ebo;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIdx), quadIdx, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
  glBindVertexArray(0);

  // Compile GL shader
  GLuint vs = compile_shader(GL_VERTEX_SHADER, vshader_src);
  GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fshader_src);
  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glLinkProgram(prog);
  {
    GLint ok2; glGetProgramiv(prog, GL_LINK_STATUS, &ok2);
    if (!ok2) {
      char buf[2048]; buf[0] = 0;
      glGetProgramInfoLog(prog, 2048, NULL, buf);
      std::cerr << "GL Link error: " << buf << std::endl;
    }
  }
  glDeleteShader(vs); glDeleteShader(fs);
  glUseProgram(prog);
  glUniform1i(glGetUniformLocation(prog, "uTex"), 0);

  // Create an OpenGL texture that will be shared with OpenCL
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);

  // integer textures use GL_NEAREST for filtering but must use GL_RGBA_INTEGER as the format
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // internal format GL_RGBA8UI and format GL_RGBA_INTEGER
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, GRID_W, GRID_H, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, nullptr);
  glBindTexture(GL_TEXTURE_2D, 0);


  // Initialize host grid (random)
  std::vector<uint8_t> host_grid(GRID_N);
  {
    std::mt19937 rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> d(0, NUM_SPECIES);
    for (size_t y = 0; y < GRID_H; ++y) for (size_t x = 0; x < GRID_W; ++x) host_grid[idx(x, y)] = (uint8_t)d(rng);
  }

  // -------- OpenCL setup (with GL sharing) --------
  cl_int clerr;
  cl_platform_id platform = nullptr;
  cl_uint num_platforms = 0;
  clerr = clGetPlatformIDs(1, &platform, &num_platforms);
  check_cl(clerr, "clGetPlatformIDs");

  // Prefer GPU device
  cl_device_id device = nullptr;
  cl_uint num_devices = 0;
  clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
  if (clerr != CL_SUCCESS) {
    // fallback to any device
    clerr = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
    check_cl(clerr, "clGetDeviceIDs fallback");
  }

  // Prepare context properties for WGL sharing (Windows)
  cl_context_properties props[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
      CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
      CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
      0
  };

  cl_context clContext = clCreateContext(props, 1, &device, NULL, NULL, &clerr);
  if (clerr != CL_SUCCESS || clContext == NULL) {
    clContext = clCreateContext(NULL, 1, &device, NULL, NULL, &clerr);
    check_cl(clerr, "clCreateContext fallback");
  }

  // Create command queue
  const cl_queue_properties qprop[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
  cl_command_queue clQueue = clCreateCommandQueueWithProperties(clContext, device, qprop, &clerr);
  check_cl(clerr, "clCreateCommandQueueWithProperties");

  // Build program
  const char* src_ptr = cl_kernel_src;
  cl_program program = clCreateProgramWithSource(clContext, 1, &src_ptr, nullptr, &clerr);
  check_cl(clerr, "clCreateProgramWithSource");
  clerr = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (clerr != CL_SUCCESS) {
    // print build log
    size_t logsz = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
    std::vector<char> logbuf(logsz + 1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsz, logbuf.data(), NULL);
    std::cerr << "CL Build log:\n" << logbuf.data() << "\n";
    check_cl(clerr, "clBuildProgram");
  }

  cl_kernel kernel = clCreateKernel(program, "gol_sim_and_color", &clerr);
  check_cl(clerr, "clCreateKernel");

  // Create two device buffers for ping-pong (gridA = current input, gridB = next output)
  cl_mem d_grid_a = clCreateBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
    sizeof(uint8_t) * GRID_N, host_grid.data(), &clerr);
  check_cl(clerr, "clCreateBuffer grid_a");

  cl_mem d_grid_b = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
    sizeof(uint8_t) * GRID_N, nullptr, &clerr);
  check_cl(clerr, "clCreateBuffer grid_b");

  // Create CL image from GL texture (shared)
  cl_mem cl_tex = clCreateFromGLTexture(clContext, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, tex, &clerr);
  cl_image_format fmt;
  size_t rb;
  clGetImageInfo(cl_tex, CL_IMAGE_FORMAT, sizeof(fmt), &fmt, &rb);
  std::cout << "CL channel order: " << fmt.image_channel_order
    << "   type: " << fmt.image_channel_data_type << "\n";

  if (clerr != CL_SUCCESS) {
    std::cerr << "clCreateFromGLTexture failed: " << clerr << std::endl;
    check_cl(clerr, "clCreateFromGLTexture");
  }

  // Set constant kernel args that don't change per-frame: width, height, num_species
  check_cl(clSetKernelArg(kernel, 0, sizeof(int), &GRID_W), "clSetKernelArg 0");
  check_cl(clSetKernelArg(kernel, 1, sizeof(int), &GRID_H), "clSetKernelArg 1");
  check_cl(clSetKernelArg(kernel, 2, sizeof(int), &NUM_SPECIES), "clSetKernelArg 2");
  // args 3 to 5 will be set per-frame: frame_seed, grid_in, grid_out, and image is arg 6
  // But ordering in kernel: (width, height, num_species, frame_seed, grid_in, grid_out, out_tex)
  // So arg index 3 is frame_seed, 4 grid_in, 5 grid_out, 6 out_tex

  // same cl_mem
  check_cl(clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_tex), "clSetKernelArg 6 (out_tex)");

  // Timing prep
  uint32_t frame = 1;
  cl_mem cur_grid = d_grid_a;
  cl_mem next_grid = d_grid_b;

  size_t gws[2] = { (size_t)GRID_W, (size_t)GRID_H };
  size_t* lws = NULL;

  std::cout << "Starting main loop. GRID " << GRID_W << "x" << GRID_H << ", num_species=" << NUM_SPECIES << "\n";

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Compute frame seed
    uint32_t frame_seed = frame * 1640531527u + 123456789u;

    // Set per-frame kernel args
    check_cl(clSetKernelArg(kernel, 3, sizeof(uint32_t), &frame_seed), "clSetKernelArg 3 (frame_seed)");
    check_cl(clSetKernelArg(kernel, 4, sizeof(cl_mem), &cur_grid), "clSetKernelArg 4 (grid_in)");
    check_cl(clSetKernelArg(kernel, 5, sizeof(cl_mem), &next_grid), "clSetKernelArg 5 (grid_out)");
    // arg 6 was set earlier to cl_tex (the image2d_t from gl texture)

    // ensure any GL work is finished before CL acquires the texture
    glFinish();

    // Acquire GL texture for CL
    cl_mem globjs[] = { cl_tex };
    check_cl(clEnqueueAcquireGLObjects(clQueue, 1, globjs, 0, NULL, NULL), "clEnqueueAcquireGLObjects");

    // compute next grid and write colors to image (which is the GL texture)
    check_cl(clEnqueueNDRangeKernel(clQueue, kernel, 2, NULL, gws, lws, 0, NULL, NULL), "clEnqueueNDRangeKernel");

    // Release GL texture back to GL
    check_cl(clEnqueueReleaseGLObjects(clQueue, 1, globjs, 0, NULL, NULL), "clEnqueueReleaseGLObjects");

    // Wait for CL work to finish
    check_cl(clFinish(clQueue), "clFinish");

    // render the shared texture with OpenGL (no upload needed)
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);
    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();

    // Swap grids
    std::swap(cur_grid, next_grid);

    ++frame;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = t1 - t0;

    if ((frame & 31) == 0) {
      double fps = 1000.0 / elapsed.count();
      std::cout << " fps ~ " << fps << " species=" << NUM_SPECIES << '\n';
    }

    // close on ESC
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
  }

  // Cleanup
  clReleaseMemObject(d_grid_a);
  clReleaseMemObject(d_grid_b);
  clReleaseMemObject(cl_tex);

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(clQueue);
  clReleaseContext(clContext);

  glDeleteTextures(1, &tex);
  glDeleteProgram(prog);
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &ebo);
  glDeleteVertexArrays(1, &vao);

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
