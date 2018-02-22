/*
 * GL01Hello.cpp: Test OpenGL/GLUT C/C++ Setup
 * Tested under Eclipse CDT with MinGW/Cygwin and CodeBlocks with MinGW
 * To compile with -lfreeglut -lglu32 -lopengl32
 */

#include <GL/glut.h>  // GLUT, include glu.h and gl.h

/* Handler for window-repaint event. Call back when the window first appears and
   whenever the window needs to be re-painted. */
void display() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color to black and opaque
    glClear(GL_COLOR_BUFFER_BIT);         // Clear the color buffer (background)

     //Draw a Red 1x1 Square centered at origin
//    glBegin(GL_QUADS);              // Each set of 4 vertices form a quad
//    glColor3f(1.0f, 0.0f, 0.0f); // Red
//    glVertex2f(-0.5f, -0.5f);    // x, y
//    glVertex2f( 0.5f, -0.5f);
//    glVertex2f( 0.5f,  0.5f);
//    glVertex2f(-0.5f,  0.5f);
//    glEnd();
    glRectf(0.0f,0.0f, 0.5f, 0.5f);
//	glBegin (GL_QUADS);
//
//	glColor3f (0.0f, 0.0f, 1.0f); // BLUE COLOR LEFT END
//	glVertex2f (-1.0f, 0.0f);
//	glVertex2f (-1.0f, 1.0f);
//
//    glColor3f(0.0,1.0,0.0);
//    glVertex2f (0.0, 0.0);
//    glVertex2f (0.0, 1);
////
////    glVertex2f (1/2.0, 0.0f);
////    glVertex2f (1/2.0, 1,0.0);
//
//	glColor3f (1.0f, 0.0f, 0.0f); //RED COLOR RIGHT END
//	glVertex2f (1.0f, 0.0f);
//	glVertex2f (.5f, 0.5f);
//	glEnd ();
    glFlush();  // Render now
}

/* Main function: GLUT runs as a console application starting at main()  */
int main(int argc, char** argv) {
    glutInit(&argc, argv);                 // Initialize GLUT
    glutCreateWindow("OpenGL Setup Test"); // Create a window with the given title
    glutInitWindowSize(500, 500);   // Set the window's initial width & height
    glutInitWindowPosition(50, 50); // Position the window's initial top-left corner
    glutDisplayFunc(display); // Register display callback handler for window re-paint
    glutMainLoop();           // Enter the event-processing loop
    return 0;
}