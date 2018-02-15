#include <math.h>               //for various math functions

#ifdef LINUX
#include <GL/glut.h>            //the GLUT graphics library
#endif

#ifdef MACOS
#include <GLUT/glut.h>            //the GLUT graphics library
#endif

//Global variables -----------------------------------------------------------------------------------------------------
int winWidth, winHeight;        //Size of the GL window
int n_values = 256;             //Number of values for the color map

float max(float x, float y) { return x > y ? x : y; }

float min(float x, float y) { return x < y ? x : y; }

//heatmap: Implements a heatmap color palette (Black-Red-Yellow-White).
void heatmap(float value, float* R, float* G, float* B)
{
    //Clamp value between 0 and 1
    if (value<0)
        value=0;
    if (value>1)
        value=1;

    //For now we don't mess with S,V, only with Hue
    //Normalise value to [0,3] 0->Black, 1->Full red, 2-> Orange, 3->Full yellow.
    value *= 4;
    *R = max(0, 1 - fabs(value/2-1)) + max(0, 1 - fabs(value/2-2));
    *B = max(0, 1 - fabs(value - 4));
    *G = max(0, 1 - fabs(value-3)) + max(0, 1 - fabs(value-4));
}

int mod (int a, int b)
{
    if(b < 0) //you can check for b == 0 separately and do what you want
        return mod(a, -b);
    int ret = a % b;
    if(ret < 0)
        ret+=b;
    return ret;
}

void color_test(int i, float* R, float* G, float* B)
{
    float r[] = {1, 0, 0};
    float g[] = {0, 1, 0};
    float b[] = {0, 0, 1};
    int ind = mod(i,3);
    *R = r[ind];
    *G = g[ind];
    *B = b[ind];
}

void draw_legend(void)
{
    float R, G, B;
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBegin(GL_QUADS);
    float step = (float) winHeight / (float) n_values;
    for (int j = 0; j < n_values; ++j)
    {
        //Normalise value (j) to [0,1]
        float v = (float) j/((float) n_values);

        float y0 = step*j;
        float x0 = 0;
        float y1 = step*(j+1);
        float x1 = winWidth;

        heatmap(v, &R, &G, &B);
//        R = 1; G = 0; B = 0;
//        color_test(j, &R, &G, &B);
        //Draw quad
        glColor3f(R,G,B);
        glVertex2f(x0, y0);
        glVertex2f(x1, y0);
        glVertex2f(x1, y1);
        glVertex2f(x0, y1);
    }
    glEnd();
}

//display: Handle window redrawing events. Simply delegates to visualize().
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    draw_legend();
    glFlush();
    glutSwapBuffers();
}


void reshape(int w, int h)
{
    glViewport(0.0f, 0.0f, (GLfloat)w, (GLfloat)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
    winWidth = w; winHeight = h;
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(50,500);
    glutCreateWindow("Legend visualizer");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(glutPostRedisplay);

    glutMainLoop();			//calls do_one_simulation_step, keyboard, display, drag, reshape
    return 0;
}