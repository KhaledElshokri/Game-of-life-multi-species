#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

// Window / grid size
const int WIN_W = 1024;
const int WIN_H = 768;
const int GRID_W = WIN_W;
const int GRID_H = WIN_H;

// Number of species (5..10 typical)
const int NUM_SPECIES = 5;

// Colors for species (RGB each 0-255) - index 0 reserved for dead
const uint8_t SPECIES_COLORS[11][3] = {
    {0, 0, 0},        // 0 dead: black
    {255, 0, 0},      // species 1: red
    {0, 255, 0},      // species 2: green
    {0, 0, 255},      // species 3: blue
    {255, 255, 0},    // species 4: yellow
    {255, 0, 255},    // species 5: magenta
    {0, 255, 255},    // species 6: cyan
    {255, 165, 0},    // species 7: orange
    {128, 0, 128},    // species 8: purple
    {192, 192, 192},  // species 9: light gray
    {255, 255, 255}   // species 10: white
};

// index in 1D vector for (x,y)
inline int idx(int x, int y) { return y * GRID_W + x; }

// Shared state
std::vector<uint8_t> grid_cur(GRID_W* GRID_H);   // species id per cell
std::vector<uint8_t> grid_next(GRID_W* GRID_H);  // next generation
std::vector<uint8_t> pixels_curr(GRID_W* GRID_H * 3); // RGB pixel buffer for texture
std::vector<uint8_t> pixels_next(GRID_W* GRID_H * 3); // RGB pixel buffer for texture
std::vector<std::vector<uint8_t>> neighbor_counts(NUM_SPECIES + 1); // per-species neighbor counts

// Thread sync
int num_worker_threads = std::max(1u, std::thread::hardware_concurrency() - 2);
std::atomic<int> workers_remaining{ 0 };
std::mutex mtx;
std::condition_variable cv_workers_done;
std::condition_variable cv_start_step;
bool start_step = false;
bool stop_all = false;

// Precompute neighbor counts for all species
void compute_all_neighbor_counts(const std::vector<uint8_t>& grid) {
  for (uint8_t s = 1; s <= NUM_SPECIES; ++s) {
    auto& nc = neighbor_counts[s];
    nc.assign(GRID_W * GRID_H, 0);

    for (int y = 0; y < GRID_H; ++y) {
      int row = y * GRID_W;
      for (int x = 0; x < GRID_W; ++x) {
        if (grid[row + x] != s) continue;
        for (int yy = std::max(0, y - 1); yy <= std::min(GRID_H - 1, y + 1); ++yy) {
          int row2 = yy * GRID_W;
          for (int xx = std::max(0, x - 1); xx <= std::min(GRID_W - 1, x + 1); ++xx) {
            if (xx == x && yy == y) continue;
            ++nc[row2 + xx];
          }
        }
      }
    }
  }
}

// Worker thread function: handles rows [y0, y1)
void worker_thread_func(int id, int y0, int y1) {
  while (true) {
    // Wait for start signal
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv_start_step.wait(lk, [] { return start_step || stop_all; });
      if (stop_all) return;
    }

    // Compute assigned rows
    for (int y = y0; y < y1; ++y) {
      for (int x = 0; x < GRID_W; ++x) {
        // choose species with most live neighbors that results in alive state
        uint8_t final_species = 0; // dead by default
        int best_score = -1000;
				int index = idx(x, y);

        for (uint8_t s = 1; s <= NUM_SPECIES; ++s) {
          int neighbors = neighbor_counts[s][index];
          uint8_t cur = grid_cur[index];
          bool cur_alive = (cur == s);
          bool next_alive = false;
          // Apply standard Game of Life rules per species
          if (cur_alive) {
            if (neighbors < 2) next_alive = false;
            else if (neighbors == 2 || neighbors == 3) next_alive = true;
            else next_alive = false;
          }
          else {
            if (neighbors == 3) next_alive = true;
            else next_alive = false;
          }
          if (next_alive) {
            // Candidate species to occupy this cell.
            // Use neighbors as tie-breaker (more neighbors wins). If equal, smaller species id wins.
            int score = neighbors * 10 - s; // deterministic maybe change this for tbb
            if (score > best_score) {
              best_score = score;
              final_species = s;
            }
          }
        } // end species loop

        grid_next[index] = final_species;
				// SIMD-friendly pixel assignment
				const uint8_t* color = SPECIES_COLORS[final_species];
				uint8_t* dst = &pixels_next[index * 3];
				dst[0] = color[0];
        dst[1] = color[1];
        dst[2] = color[2];
       }
    }

    // Signal completion for this worker
    if (workers_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::unique_lock<std::mutex> lk(mtx);
      start_step = false; // reset
      cv_workers_done.notify_one();
    }

    // Wait here for next start (loop continues)
  }
}

// Tiny vertex & fragment shader to draw textured quad
const char* vshader_src = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";
const char* fshader_src = R"(
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
void main() {
    FragColor = texture(uTex, vUV);
}
)";

GLuint compile_shader(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, NULL);
  glCompileShader(s);
  GLint ok;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char buf[1024]; glGetShaderInfoLog(s, 1024, NULL, buf);
    std::cerr << "Shader compile error: " << buf << std::endl;
  }
  return s;
}

int main() {

  // Init GLFW with core profile 3.3
  if (!glfwInit()) return -1;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

  // Window setup
  GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H, "Multi-Species Game of Life", NULL, NULL);
  if (!window) { glfwTerminate(); return -1; }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);

  if (glewInit() != GLEW_OK) {
    std::cerr << "GLEW init failed\n";
    return -1;
  }

  // Setup simple textured quad (full screen)
  float quadVerts[] = {
    // positions    // uvs
    -1.f, -1.f,     0.f, 0.f,
     1.f, -1.f,     1.f, 0.f,
     1.f,  1.f,     1.f, 1.f,
    -1.f,  1.f,     0.f, 1.f
  };
  unsigned int quadIdx[] = { 0,1,2, 2,3,0 };
  GLuint vao, vbo, ebo;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIdx), quadIdx, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);

  // Compile shader
  GLuint vs = compile_shader(GL_VERTEX_SHADER, vshader_src);
  GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fshader_src);
  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glLinkProgram(prog);
  GLint ok; 
  glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (!ok) {
    char buf[1024]; 
    glGetProgramInfoLog(prog, 1024, NULL, buf);
    std::cerr << "Program link error: " << buf << std::endl;
  }
  glDeleteShader(vs);
  glDeleteShader(fs);

  // Create texture
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, GRID_W, GRID_H, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

  // Initialize grid randomly
  std::mt19937 rng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<int> d(0, NUM_SPECIES);
  for (int y = 0; y < GRID_H; ++y) {
   for (int x = 0; x < GRID_W; ++x) {
        grid_cur[idx(x, y)] = (uint8_t)d(rng);
			  uint8_t s = grid_cur[idx(x, y)];
        pixels_curr[idx(x, y) * 3 + 0] = SPECIES_COLORS[s][0];
        pixels_curr[idx(x, y) * 3 + 1] = SPECIES_COLORS[s][1];
			  pixels_curr[idx(x, y) * 3 + 2] = SPECIES_COLORS[s][2];
   }
  }

  // Launch worker threads
  std::vector<std::thread> workers;
  workers.reserve(num_worker_threads);
  int rows_per_worker = GRID_H / num_worker_threads;
  for (int i = 0; i < num_worker_threads; ++i) {
    int y0 = i * rows_per_worker;
    int y1 = (i == num_worker_threads - 1) ? GRID_H : (y0 + rows_per_worker);
    workers.emplace_back(worker_thread_func, i, y0, y1);
  }

  // Main loop of the application
  while (!glfwWindowShouldClose(window)) {

    auto start =  std::chrono::high_resolution_clock::now();

		// Precompute neighbor counts for all species
		compute_all_neighbor_counts(grid_cur);

    // signal workers to start computing next generation
    {
      std::unique_lock<std::mutex> lk(mtx);
      workers_remaining = num_worker_threads;
      start_step = true;
      cv_start_step.notify_all();
    }

    // Wait for workers to finish
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv_workers_done.wait(lk, [] { return workers_remaining == 0; });
    }

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> elapsed = end - start;
		std::cout << "Frame time: " << elapsed.count() << " ms\n";

    // swap buffers (producer-consumer double buffer)
    grid_cur.swap(grid_next);
    pixels_curr.swap(pixels_next);

    // Convert to pixels (colors) and upload texture
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GRID_W, GRID_H, GL_RGB, GL_UNSIGNED_BYTE, pixels_curr.data());

    // Draw textured quad full-screen
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);
    glBindVertexArray(vao);
    // Texture unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();

  }

  // Shutdown: notify workers to stop and join
  {
    std::unique_lock<std::mutex> lk(mtx);
    stop_all = true;
    cv_start_step.notify_all();
  }
  for (auto& t : workers) if (t.joinable()) t.join();

  // Cleanup GL
  glDeleteTextures(1, &tex);
  glDeleteProgram(prog);
  glDeleteBuffers(1, &vbo);
  glDeleteBuffers(1, &ebo);
  glDeleteVertexArrays(1, &vao);

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
