//
// Created by mario on 28-3-18.
//

#include <iostream>
#include "GL/glui.h"
#include "simulation.h"
#include "visualization.h"

GLUI_RadioGroup *colormap_radio;
GLUI_RadioGroup *glyphtype;
GLUI_RadioGroup *dataset_radio;
GLUI_RadioGroup *glyph_radio;
GLUI_Button *button;
int   main_window;
GLUI *glui_v_subwindow;
int segments = 0;
GLUI_Spinner *clamp_max_spinner;
GLUI_Spinner *clamp_min_spinner;
GLUI *glui;
const int RADIO_COLOR_MAP = 0;
const int RADIO_DATASET = 1;
const int RADIO_GLYPH = 2;
int spin;
int left_button = GLUT_UP;  //left button is initially not pressed
int right_button = GLUT_UP; //right button is initially not pressed
int middle_button = GLUT_UP; //middle button is initially not pressed
Visualization* vis = new Visualization(50);
Simulation* sim = vis->sim;
int last_my = 0;
int last_mx = 0;

float eye[3]; //point representing the eye
float lookat[3]; //point towards which eye has too point

using namespace std;

void radio_cb( int control )
{
    switch (control)
    {
        case RADIO_COLOR_MAP:   vis->scalar_col = colormap_radio->get_int_val();         break;
        case RADIO_DATASET:     vis->display_dataset = dataset_radio->get_int_val();     break;
        case RADIO_GLYPH:       vis->vGlyph = (glyph_radio->get_int_val());              break;
    }
}

void color_bands_cb(int control)
{
    vis->create_textures();
}

void glyph_button_cb(int control)
{
    printf( "callback: %d\n", control );
    vis->glyph = true;
    if (vis->glyph)
    {
        printf("Glyph view enabled");
    } else {
        printf ("Glyph view disabled");
    }
}
void glyphtype_cb(int control)
{
    vis->typeGlyph = glyphtype->get_int_val();
}

void divergence_cb( int control )
{
    if (vis->display_divergence)
    {
        clamp_min_spinner->set_int_limits(-1, 0);
        clamp_min_spinner->set_int_val(-1);
    }
    else
    {
        clamp_min_spinner->set_int_limits(0, 1);
        clamp_min_spinner->set_int_val(0);
    }
}

void enable_height_plot( int control )
{
    //Set default perspective and stuff
}

//------ INTERACTION CODE STARTS HERE -----------------------------------------------------------------

//display: Handle window redrawing events. Simply delegates to visualize().
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-.5, .5, -.5, .5, 1, 10000);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    if (vis->height_plot) {
        gluLookAt(eye[0], eye[1], eye[2], lookat[0], lookat[1], lookat[2], 0, 0, 1);
        vis->draw_3d_grid();
    }

    vis->visualize();

    glFlush();
    glutSwapBuffers();
}

//reshape: Handle window resizing (reshaping) events
void reshape(int w, int h)
{
    int tx, ty, tw, th;
    GLUI_Master.get_viewport_area(&tx, &ty, &tw, &th);
    glViewport(tx, ty, tw, th);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, (GLdouble)w, 0.0, (GLdouble)h, -10.0, 10.0);
    vis->winWidth = w; vis->winHeight = h;
    vis->gridWidth = vis->winWidth - vis->legend_size - vis->legend_text_len;
    vis->gridHeight = vis->winHeight;
    vis->wn = (fftw_real)vis->gridWidth  / (fftw_real)(sim->DIM + 1);
    vis->hn = (fftw_real)vis->gridHeight / (fftw_real)(sim->DIM + 1);

    lookat[0] = vis->gridWidth/2.0f;
    lookat[1] = vis->gridHeight/2.0f;

    glutPostRedisplay();
}

//keyboard: Handle key presses
void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 't': sim->dt -= 0.001; break;
        case 'T': sim->dt += 0.001; break;
        case 'c': vis->color_dir = 1 - vis->color_dir; break;
        case 'd': vis->display_dataset = !vis->display_dataset; break;
        case 'S': vis->vec_scale *= 1.2; break;
        case 's': vis->vec_scale *= 0.8; break;
        case 'V': sim->visc *= 5; break;
        case 'v': sim->visc *= 0.2; break;
        case 'i': vis->apply_mode = !vis->apply_mode; break;
        case 'x':
            vis->draw_smoke = 1 - vis->draw_smoke;
            if (vis->draw_smoke==0) vis->draw_vecs = 1;
            break;
        case 'y':
            vis->draw_vecs = 1 - vis->draw_vecs;
            if (vis->draw_vecs==0) vis->draw_smoke = 1;
            break;
        case 'm':
            vis->scalar_col++;
            if (vis->scalar_col>vis->COLOR_HEATMAP)
                vis->scalar_col=vis->COLOR_BLACKWHITE;
            break;
        case 'a': vis->frozen = 1-vis->frozen; break;
        case '+': vis->n_colors = min(256, vis->n_colors+1); break;
        case '-': vis->n_colors = max(5, vis->n_colors-1); break;
        case '[': clamp_min_spinner->set_float_val(sim->max(0, vis->clamp_min-0.1f)); break;
        case ']': clamp_min_spinner->set_float_val(vis->clamp_min+0.1f); break;
        case '{': clamp_max_spinner->set_float_val(sim->max(0, vis->clamp_max-0.1f)); break;
        case '}': clamp_max_spinner->set_float_val(vis->clamp_max+0.1f); break;
        case 'q': exit(0);
    }
    glui->sync_live(); //Synchronise live variables to update keyboard changes in gui.
}

void mouseCallback(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON)
        left_button = state;
    else if (button == GLUT_RIGHT_BUTTON)
        right_button = state;
    else if (button == GLUT_MIDDLE_BUTTON)
        middle_button = state;

    last_mx = x;
    last_my = y;
}

//add_matter: When the user drags with the left mouse button pressed, add a force that corresponds to the direction of
// the mouse cursor movement. Also inject some new matter into the field at the mouse location.
void add_matter(int mx, int my)
{
    int xi,yi,X,Y;
    double  dx, dy, len;
    static int lmx=0,lmy=0;				//remembers last mouse location

    // Compute the array index that corresponds to the cursor location
    xi = (int)sim->clamp((double)(sim->DIM + 1) * ((double)mx / (double)vis->gridWidth));
    yi = (int)sim->clamp((double)(sim->DIM + 1) * ((double)(vis->gridHeight - my) / (double)vis->gridHeight));

    X = xi;
    Y = yi;

    if (X > (sim->DIM - 1))
        X = sim->DIM - 1;
    if (Y > (sim->DIM - 1))
        Y = sim->DIM - 1;
    if (X < 0)
        X = 0;
    if (Y < 0)
        Y = 0;

    // Add force at the cursor location
    my = vis->winHeight - my;
    dx = mx - lmx;
    dy = my - lmy;
    len = sqrt(dx * dx + dy * dy);
    if (len != 0.0)
    {
        dx *= 0.1 / len;
        dy *= 0.1 / len;
    }
    sim->fx[Y * sim->DIM + X] += dx;
    sim->fy[Y * sim->DIM + X] += dy;
    sim->rho[Y * sim->DIM + X] = 10.0f;
    lmx = mx;
    lmy = my;
}

void crossproduct(float a[3], float b[3], float res[3])
{
    res[0] = (a[1] * b[2] - a[2] * b[1]);
    res[1] = (a[2] * b[0] - a[0] * b[2]);
    res[2] = (a[0] * b[1] - a[1] * b[0]);
}

void normalize(float v[3])
{
    float l = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    l = 1 / (float)sqrt(l);

    v[0] *= l;
    v[1] *= l;
    v[2] *= l;
}

float length(float v[3])
{
    return (float)sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

int orbit_view(int mx, int my)
{
    int dx = -(mx-last_mx);
    int dy = my-last_my;
    float neye[3], neye2[3];
    float len;
    float f[3], r[3], u[3];

    neye[0] = eye[0] - lookat[0];
    neye[1] = eye[1] - lookat[1];
    neye[2] = eye[2] - lookat[2];

    // first rotate in the x/z plane
    float theta = -dx * 0.007;
    neye2[0] = cos(theta)*neye[0] + sin(theta)*neye[1];
    neye2[1] = -sin(theta)*neye[0] + cos(theta)*neye[1];
    neye2[2] = eye[2];


    // now rotate vertically
    theta = -dy * 0.007;

    f[0] = -neye2[0];
    f[1] = -neye2[1];
    f[2] = -neye2[2];
    u[0] = 0;
    u[1] = 0;
    u[2] = 1;
    crossproduct(f, u, r);
    crossproduct(r, f, u);
    len = length(f);
    normalize(f);
    normalize(u);

    neye[0] = len * ((float)cos(theta)*f[0] + (float)sin(theta)*u[0]);
    neye[1] = len * ((float)cos(theta)*f[1] + (float)sin(theta)*u[1]);
    neye[2] = len * ((float)cos(theta)*f[2] + (float)sin(theta)*u[2]);

    eye[0] = lookat[0] - neye[0];
    eye[1] = lookat[1] - neye[1];
    eye[2] = lookat[2] - neye[2];

    last_mx=mx; last_my=my;
}

void zoom(int my)
{
    int dy = (my - last_my)*0.5;
    float neye[3]; //normalized eye, for scaling purposes
    neye[0] = eye[0];
    neye[1] = eye[1];
    neye[2] = eye[2];
    normalize(neye);

    eye[0] = eye[0] + neye[0]*dy;
    eye[1] = eye[1] + neye[1]*dy;
    eye[2] = eye[2] + neye[2]*dy;
}

// drag: select which action to map mouse drag to, depending on pressed button
void drag(int mx, int my)
{
    if (!vis->height_plot)
    {
       add_matter(mx, my);
    }
    else
    {
       if (left_button == GLUT_DOWN && right_button == GLUT_DOWN) return;  //Do nothing if both buttons pressed.
       glutSetWindow(glui->get_glut_window_id());

       if (middle_button == GLUT_DOWN)
           zoom(my);
       else
       {
           if (left_button == GLUT_DOWN)
               add_matter(mx, my);
           else if (right_button == GLUT_DOWN)
               orbit_view(mx, my);
       }

    }
}

void do_one_simulation_step()
{
    if (!vis->frozen)
    {
        if ( glutGetWindow() != main_window) glutSetWindow(main_window);
        vis->do_one_simulation_step();
    }
}

static void TimeEvent(int te)
{

    spin++;  // increase cube rotation by 1
    if (spin > 360) spin = 0; // if over 360 degress, start back at zero.
    glutPostRedisplay();  // Update screen with new rotation data
    glutTimerFunc( 100, TimeEvent, 1);  // Reset our timmer.
}

void reset_simulation()
{
    sim->reset_simulation();
}

//main: The main program
int main(int argc, char **argv)
{
    printf("Fluid Flow Simulation and Visualization\n");
    printf("=======================================\n");
    printf("Click and drag the mouse to steer the flow!\n");
    printf("T/t:   increase/decrease simulation timestep\n");
    printf("S/s:   increase/decrease hedgehog scaling\n");
    printf("+/-:   increase/decrease number of colors in color map\n");
    printf("[/]:   increase/decrease min clamping limit\n");
    printf("[/]:   increase/decrease max clamping limit\n");
    printf("c:     toggle direction coloring on/off\n");
    printf("d:     cycle through datasets\n");
    printf("i:     toggle application method of the color map\n");
    printf("V/v:   increase decrease fluid viscosity\n");
    printf("x:     toggle drawing matter on/off\n");
    printf("y:     toggle drawing hedgehogs on/off\n");
    printf("m:     toggle thru scalar coloring\n");
    printf("a:     toggle the animation on/off\n");
    printf("q:     quit\n\n");


    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    vis->winWidth = 900; vis->winHeight = 900;
    glutInitWindowSize(vis->winWidth,vis->winHeight);
    glutInitWindowPosition( 50, 50 );


    main_window = glutCreateWindow("Real-time smoke simulation and visualization");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(drag);
    glutMouseFunc(mouseCallback);
    glutReshapeFunc(reshape);
    glutTimerFunc( 10, TimeEvent, 1);
    glui = GLUI_Master.create_glui_subwindow(main_window, GLUI_SUBWINDOW_LEFT);


    GLUI_Panel *colormap_panel = new GLUI_Panel( glui, "Colour map type" );
    colormap_radio = new GLUI_RadioGroup(colormap_panel, (&vis->scalar_col), RADIO_COLOR_MAP, radio_cb);
    new GLUI_RadioButton( colormap_radio, "Greyscale" );
    new GLUI_RadioButton( colormap_radio, "Rainbow" );
    new GLUI_RadioButton( colormap_radio, "Heatmap" );

    GLUI_Panel *dataset_panel = new GLUI_Panel( glui, "Dataset to be Mapped" );
    dataset_radio = new GLUI_RadioGroup(dataset_panel, (&vis->display_dataset), RADIO_DATASET, radio_cb);
    new GLUI_RadioButton( dataset_radio, "Density" );
    new GLUI_RadioButton( dataset_radio, "Velocity" );
    new GLUI_RadioButton( dataset_radio, "Force" );

    GLUI_Panel *glyphvector_panel = new GLUI_Panel( glui, "vector value for vis->glyph" );
    glyph_radio = new GLUI_RadioGroup(glyphvector_panel , (&vis->vGlyph), RADIO_GLYPH, radio_cb);
    new GLUI_RadioButton( glyph_radio, "fluid velocity" );
    new GLUI_RadioButton( glyph_radio, "force" );

    GLUI_Panel *glyphtype_panel = new GLUI_Panel( glui, "choose the type of vis->glyph" );
    glyphtype = new GLUI_RadioGroup(glyphtype_panel , (&vis->typeGlyph), 6, glyphtype_cb);
    new GLUI_RadioButton( glyphtype, "default" );
    new GLUI_RadioButton( glyphtype, "cones" );
    new GLUI_RadioButton( glyphtype, "arrows" );

    GLUI_Panel *clamping_panel = new GLUI_Panel( glui, "Clamping options" );
    glui->add_checkbox_to_panel(clamping_panel, "Apply clamping", &vis->apply_mode);
    clamp_max_spinner = glui->add_spinner_to_panel(clamping_panel, "Clamp max", GLUI_SPINNER_FLOAT, &vis->clamp_max);
    clamp_min_spinner = glui->add_spinner_to_panel(clamping_panel, "Clamp min", GLUI_SPINNER_FLOAT, &vis->clamp_min);
    clamp_max_spinner->set_int_limits(0, 1);
    clamp_min_spinner->set_int_limits(-1, 1);
    clamp_max_spinner->set_float_val(1.0f);
    clamp_min_spinner->set_float_val(0.0f);

    glui->add_checkbox("Use texture mapping", &vis->texture_mapping);
    glui->add_checkbox("Dynamic scaling", &vis->dynamic_scalling);
    glui->add_checkbox("Show divergence", &vis->display_divergence, -1, divergence_cb);
    glui->add_checkbox("Render smoke", &vis->draw_smoke);
    glui->add_checkbox("Render glyphs", &vis->draw_vecs);
    glui->add_checkbox("Height plot", &vis->height_plot, -1, enable_height_plot);

    GLUI_Spinner *color_bands_spinner = glui->add_spinner("Number of colours", GLUI_SPINNER_INT, &vis->n_colors, -1, color_bands_cb);
    color_bands_spinner->set_int_limits(3, 256);

    glui->add_button("Reset", -1, (GLUI_Update_CB) reset_simulation);

    new GLUI_Button( glui, "QUIT", 0,(GLUI_Update_CB)exit );


    glui->set_main_gfx_window( main_window );
    GLUI_Master.set_glutIdleFunc(do_one_simulation_step);

//    glui->hide();

    vis->create_textures();


//    eye[0] = vis->gridWidth/2;
//    eye[1] = vis->gridHeight/2;
//    eye[2] = (int) sim->max((float) vis->gridWidth, (float) vis->gridHeight);
    eye[0] = 0;
    eye[1] = 0;
    eye[2] = 100;

    cout <<  vis->gridHeight << endl;

    glutMainLoop();			//calls do_one_simulation_step, keyboard, display, drag, reshape
    return 0;
}

