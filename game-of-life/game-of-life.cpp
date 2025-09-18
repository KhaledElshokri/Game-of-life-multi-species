#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"


int main(void)
{
  /*
  * OpenGl Initialisation
  */
  GLFWwindow* window;
  /* Initialize the library */
  if (!glfwInit()) {
    return -1;
  }
  window = glfwCreateWindow(1024, 768, "Welcome to the game of life!", NULL, NULL);
  if (!window)
  {
    glfwTerminate();
    return -1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  if (glewInit() != GLEW_OK)
  {
    std::cout << "ERROR: GLEW INIT" << std::endl;
  }

  {
    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
      // Rendering Here 
      glClear(GL_COLOR_BUFFER_BIT);

      glBegin(GL_TRIANGLES);
      glVertex2d(-0.5, -0.5);
      glVertex2d(-0.5, 0.5);
      glVertex2d(0.5, 0.5);
      glVertex2d(0.5, -0.5);
      glVertex2d(-0.5, -0.5);
      glVertex2d(0.5, 0.5);
      glEnd();

      /* Swap front and back buffers */
      glfwSwapBuffers(window);

      /* Poll for and process events */
      glfwPollEvents();
    }

  }

  glfwTerminate();
  return 0;
}