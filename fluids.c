// Usage: Drag with the mouse to add smoke to the fluid. This will also move a "rotor" that disturbs
//        the velocity field at the mouse location. Press the indicated keys to change options
//--------------------------------------------------------------------------------------------------

#include <rfftw.h>              //the numerical simulation FFTW library
#include <math.h>               //for various math functions

#include "GL/glui.h"

#include <string.h>
#include <float.h>
#include <assert.h>


#ifdef LINUX
#include <GL/glut.h>            //the GLUT graphics library
#endif

#ifdef MACOS
#include <GLUT/glut.h>            //the GLUT graphics library
#endif

// GLUI parameters and variables

GLUI_RadioGroup *c_Map;
GLUI_RadioGroup *dataset;
int cMap = 0;
int dSet = 0;
int   main_window;
GLUI *glui_v_subwindow;
int segments = 0;
void control_cb( int control )
{
    printf( "callback: %d\n", control );
    cMap = (c_Map->get_int_val());
    printf( "          radio group: %d\n", cMap );


}
//--- SIMULATION PARAMETERS ------------------------------------------------------------------------
const int DIM = 50;				//size of simulation grid
double dt = 0.4;				//simulation time step
float visc = 0.001;				//fluid viscosity
fftw_real *vx, *vy;             //(vx,vy)   = velocity field at the current moment
fftw_real *vx0, *vy0;           //(vx0,vy0) = velocity field at the previous moment
fftw_real *fx, *fy;	            //(fx,fy)   = user-controlled simulation forces, steered with the mouse
fftw_real *rho, *rho0;			//smoke density at the current (rho) and previous (rho0) moment
rfftwnd_plan plan_rc, plan_cr;  //simulation domain discretization

//--- VISUALIZATION PARAMETERS ---------------------------------------------------------------------
int   winWidth, winHeight;      //size of the graphics window, in pixels
int   gridWidth, gridHeight;    //size of the simulation grid in pixels
int   color_dir = 0;            //use direction color-coding or not
float vec_scale = 1000;			//scaling of hedgehogs
int   draw_smoke = 1;           //draw the smoke or not
int   draw_vecs = 0;            //draw the vector field or not
const int COLOR_BLACKWHITE=0;   //different types of color mapping: black-and-white, rainbow, banded
const int COLOR_RAINBOW=1;
const int COLOR_HEATMAP=2;
int   scalar_col = 0;           //method for scalar coloring
int   frozen = 0;               //toggles on/off the animation
<<<<<<< HEAD
int n_values = 0;
int   legend_size = 50;
=======
int   n_colors = 256;           //number of colors used in the color map
int   legend_size = 50;         //width of the legend
int   legend_text_len = 100;
const int DATASET_VELOCITY=1;   //Velocity is the dataset to be displayed
const int DATASET_DENSITY=0;    //Density is the dataset to be displayed
int   display_dataset = 0;      // The dataset to be displayed
const int APPLY_SCALING = 0;    //Use the scaling method to apply the color map to the dataset
const int APPLY_CLAMP = 1;      //Use the clamping method for applying the color map to the dataset
int apply_mode = 0;
float clamp_min = 0.0f;
float clamp_max = 1.0f;




//------ SIMULATION CODE STARTS HERE -----------------------------------------------------------------


//init_simulation: Initialize simulation data structures as a function of the grid size 'n'.
//                 Although the simulation takes place on a 2D grid, we allocate all data structures as 1D arrays,
//                 for compatibility with the FFTW numerical library.
void init_simulation(int n)
{
	int i; size_t dim;

	dim     = n * 2*(n/2+1)*sizeof(fftw_real);        //Allocate data structures
	vx       = (fftw_real*) malloc(dim);
	vy       = (fftw_real*) malloc(dim);
	vx0      = (fftw_real*) malloc(dim);
	vy0      = (fftw_real*) malloc(dim);
	dim     = n * n * sizeof(fftw_real);
	fx      = (fftw_real*) malloc(dim);
	fy      = (fftw_real*) malloc(dim);
	rho     = (fftw_real*) malloc(dim);
	rho0    = (fftw_real*) malloc(dim);
	plan_rc = rfftw2d_create_plan(n, n, FFTW_REAL_TO_COMPLEX, FFTW_IN_PLACE);
	plan_cr = rfftw2d_create_plan(n, n, FFTW_COMPLEX_TO_REAL, FFTW_IN_PLACE);

	for (i = 0; i < n * n; i++)                      //Initialize data structures to 0
	{ vx[i] = vy[i] = vx0[i] = vy0[i] = fx[i] = fy[i] = rho[i] = rho0[i] = 0.0f; }
}


//FFT: Execute the Fast Fourier Transform on the dataset 'vx'.
//     'dirfection' indicates if we do the direct (1) or inverse (-1) Fourier Transform
void FFT(int direction,void* vx)
{
	if(direction==1) rfftwnd_one_real_to_complex(plan_rc,(fftw_real*)vx,(fftw_complex*)vx);
	else             rfftwnd_one_complex_to_real(plan_cr,(fftw_complex*)vx,(fftw_real*)vx);
}

int clamp(float x)
{ return ((x)>=0.0?((int)(x)):(-((int)(1-(x))))); }

float max(float x, float y)
{ return x > y ? x : y; }

float min(float x, float y) {return x < y ? x : y;}

//solve: Solve (compute) one step of the fluid flow simulation
void solve(int n, fftw_real* vx, fftw_real* vy, fftw_real* vx0, fftw_real* vy0, fftw_real visc, fftw_real dt)
{
	fftw_real x, y, x0, y0, f, r, U[2], V[2], s, t;
	int i, j, i0, j0, i1, j1;

	for (i=0;i<n*n;i++)
	{
		vx[i] += dt*vx0[i];
		vx0[i] = vx[i];
		vy[i] += dt*vy0[i];
		vy0[i] = vy[i];
	}

	for ( x=0.5f/n,i=0 ; i<n ; i++,x+=1.0f/n )
		for ( y=0.5f/n,j=0 ; j<n ; j++,y+=1.0f/n )
		{
			x0 = n*(x-dt*vx0[i+n*j])-0.5f;
			y0 = n*(y-dt*vy0[i+n*j])-0.5f;
			i0 = clamp(x0); s = x0-i0;
			i0 = (n+(i0%n))%n;
			i1 = (i0+1)%n;
			j0 = clamp(y0); t = y0-j0;
			j0 = (n+(j0%n))%n;
			j1 = (j0+1)%n;
			vx[i+n*j] = (1-s)*((1-t)*vx0[i0+n*j0]+t*vx0[i0+n*j1])+s*((1-t)*vx0[i1+n*j0]+t*vx0[i1+n*j1]);
			vy[i+n*j] = (1-s)*((1-t)*vy0[i0+n*j0]+t*vy0[i0+n*j1])+s*((1-t)*vy0[i1+n*j0]+t*vy0[i1+n*j1]);
		}

	for(i=0; i<n; i++)
		for(j=0; j<n; j++)
		{
			vx0[i+(n+2)*j] = vx[i+n*j];
			vy0[i+(n+2)*j] = vy[i+n*j];
		}

	FFT(1,vx0);
	FFT(1,vy0);

	for (i=0;i<=n;i+=2)
	{
		x = 0.5f*i;
		for (j=0;j<n;j++)
		{
			y = j<=n/2 ? (fftw_real)j : (fftw_real)j-n;
			r = x*x+y*y;
			if ( r==0.0f ) continue;
			f = (fftw_real)exp(-r*dt*visc);
			U[0] = vx0[i  +(n+2)*j];
			V[0] = vy0[i  +(n+2)*j];
			U[1] = vx0[i+1+(n+2)*j];
			V[1] = vy0[i+1+(n+2)*j];

			vx0[i  +(n+2)*j] = f*((1-x*x/r)*U[0]     -x*y/r *V[0]);
			vx0[i+1+(n+2)*j] = f*((1-x*x/r)*U[1]     -x*y/r *V[1]);
			vy0[i+  (n+2)*j] = f*(  -y*x/r *U[0] + (1-y*y/r)*V[0]);
			vy0[i+1+(n+2)*j] = f*(  -y*x/r *U[1] + (1-y*y/r)*V[1]);
		}
	}

	FFT(-1,vx0);
	FFT(-1,vy0);

	f = 1.0/(n*n);
	for (i=0;i<n;i++)
		for (j=0;j<n;j++)
		{
			vx[i+n*j] = f*vx0[i+(n+2)*j];
			vy[i+n*j] = f*vy0[i+(n+2)*j];
		}
}


// diffuse_matter: This function diffuses matter that has been placed in the velocity field. It's almost identical to the
// velocity diffusion step in the function above. The input matter densities are in rho0 and the result is written into rho.
void diffuse_matter(int n, fftw_real *vx, fftw_real *vy, fftw_real *rho, fftw_real *rho0, fftw_real dt)
{
	fftw_real x, y, x0, y0, s, t;
	int i, j, i0, j0, i1, j1;

	for ( x=0.5f/n,i=0 ; i<n ; i++,x+=1.0f/n )
		for ( y=0.5f/n,j=0 ; j<n ; j++,y+=1.0f/n )
		{
			x0 = n*(x-dt*vx[i+n*j])-0.5f;
			y0 = n*(y-dt*vy[i+n*j])-0.5f;
			i0 = clamp(x0);
			s = x0-i0;
			i0 = (n+(i0%n))%n;
			i1 = (i0+1)%n;
			j0 = clamp(y0);
			t = y0-j0;
			j0 = (n+(j0%n))%n;
			j1 = (j0+1)%n;
			rho[i+n*j] = (1-s)*((1-t)*rho0[i0+n*j0]+t*rho0[i0+n*j1])+s*((1-t)*rho0[i1+n*j0]+t*rho0[i1+n*j1]);
		}
}

//set_forces: copy user-controlled forces to the force vectors that are sent to the solver.
//            Also dampen forces and matter density to get a stable simulation.
void set_forces(void)
{
	int i;
	for (i = 0; i < DIM * DIM; i++)
	{
		rho0[i]  = 0.995 * rho[i];
		fx[i] *= 0.85;
		fy[i] *= 0.85;
		vx0[i]    = fx[i];
		vy0[i]    = fy[i];
	}
}


//do_one_simulation_step: Do one complete cycle of the simulation:
//      - set_forces:
//      - solve:            read forces from the user
//      - diffuse_matter:   compute a new set of velocities
//      - gluPostRedisplay: draw a new visualization frame
void do_one_simulation_step(void)
{
	if (!frozen)
	{
		set_forces();
		solve(DIM, vx, vy, vx0, vy0, visc, dt);
		diffuse_matter(DIM, vx, vy, rho, rho0, dt);
		glutPostRedisplay();
	}
}


//------ VISUALIZATION CODE STARTS HERE -----------------------------------------------------------------


//rainbow: Implements a color palette, mapping the scalar 'value' to a rainbow color RGB
void rainbow(float value,float* R,float* G,float* B)
{
	const float dx=0.8;
	if (value<0)
		value=0;
	if (value>1)
		value=1;
	value = (6-2*dx)*value+dx;
	*R = max(0.0,(3-fabs(value-4)-fabs(value-5))/2);
	*G = max(0.0,(4-fabs(value-2)-fabs(value-4))/2);
	*B = max(0.0,(3-fabs(value-1)-fabs(value-2))/2);
}

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

float conform_to_bands(float vy)
{
    vy *= n_colors;
    vy = (int)(vy);
    vy/= n_colors;
    return vy;
}

fftw_real scale(fftw_real min, fftw_real max, fftw_real value)
{
    return (value - min)/(max-min);
}

//set_colormap: Sets different types of colormaps
void set_colormap(float vy)
{
	float R,G,B;

    scalar_col = cMap;
	if (scalar_col==COLOR_BLACKWHITE)


    vy = conform_to_bands(vy);

    if (scalar_col==COLOR_BLACKWHITE)

		R = G = B = vy;
	else if (scalar_col==COLOR_RAINBOW)
		rainbow(vy,&R,&G,&B);
    else if (scalar_col==COLOR_HEATMAP)
        heatmap(vy, &R, &G, &B);

	glColor3f(R,G,B);
}

//draw text at x, y location
void draw_text(const char* text, int x, int y)
{
    int len = strlen( text );

    glColor3f( 1.0f, 1.0f, 1.0f );
    glRasterPos2f( x, y );
    for( int i = 0; i < len; i++ ) {
        glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, text[i] );
    }
}

void draw_legend(fftw_real min_v, fftw_real max_v)
{

    n_values = segments;
    float R, G, B;
//    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//    glBegin(GL_QUADS);
    float step = (float) winHeight / (float) n_values;
    for (int j = 0; j < n_values; ++j)


    float step = (float) winHeight / (float) n_colors;
    for (int j = 0; j < n_colors; ++j)

    {
        //Normalise value (j) to [0,1]

        float v = (float) j/((float) n_colors);

        float y0 = step*j;
        float x0 = winWidth-legend_size-legend_text_len; //do not hardcode legend size
        float y1 = step*(j+1);

        float x1 = winWidth;
        heatmap(v, &R, &G, &B);
        glColor3f(R,G,B);
		glRecti(x0,y0,x1,y1);
// R = 1; G = 0; B = 0;
// color_test(j, &R, &G, &B);
//Draw quad

//        glVertex2f(x0, y0);
//        glVertex2f(x1, y0);
//        glVertex2f(x1, y1);
//        glVertex2f(x0, y1);

        float x1 = winWidth-legend_text_len;

        set_colormap(v);

		glRecti(x0,y0,x1,y1);


    }
    char string[48];
    snprintf (string, sizeof(string), "%f", min_v);
    draw_text(string, winWidth-legend_text_len, 0);
	snprintf (string, sizeof(string), "%f", max_v);
    draw_text(string, winWidth-legend_text_len, winHeight-15);
}

//direction_to_color: Set the current color by mapping a direction vector (x,y), using
//                    the color mapping method 'method'. If method==1, map the vector direction
//                    using a rainbow colormap. If method==0, simply use the white color
void direction_to_color(float x, float y, int method)
{
	float r,g,b,f;
	if (method)
	{
		f = atan2(y,x) / 3.1415927 + 1;
		r = f;
		if(r > 1)
			r = 2 - r;
		g = f + .66667;
		if(g > 2)
			g -= 2;
		if(g > 1)
			g = 2 - g;
		b = f + 2 * .66667;
		if(b > 2)
			b -= 2;
		if(b > 1)
			b = 2 - b;
	}
	else
	{
		r = g = b = 1;
	}
	glColor3f(r,g,b);
}

//find_min_max: return the min and max values in a dataset
void find_min_max(fftw_real* min_v, fftw_real* max_v, fftw_real* dataset)
{
    *max_v = FLT_MIN;
    *min_v = FLT_MAX;
    for (int i = 0; i < DIM; ++i) {
        if (dataset[i] < *min_v) *min_v = dataset[i];
        if (dataset[i] > *max_v) *max_v = dataset[i];
    }

}

//prepare_dataset: perform the required transformations in order to make dataset ready to display (clamping, scalling...)
void prepare_dataset(fftw_real* dataset, fftw_real* min_v, fftw_real* max_v)
{
    size_t dim = DIM * 2*(DIM/2+1);

    assert(display_dataset == DATASET_DENSITY || display_dataset == DATASET_VELOCITY);

    //Copy proper dataset source
    if (display_dataset == DATASET_DENSITY)
    {
        dataset = memcpy(dataset, rho, dim * sizeof(rho));
    }
    else    //DATASET_VELOCITY
    {
        for (int i = 0; i < dim; ++i)
            dataset[i] = sqrt(pow(vx[i],2) + pow(vy[i],2));
    }

    //Apply transformation
    if (apply_mode == APPLY_SCALING)    //Apply scaling
    {
        find_min_max(min_v, max_v, dataset);
        for (int i = 0; i < dim; ++i)
            dataset[i] = scale(*min_v, *max_v, dataset[i]);
    }
    else                                //Apply clamping
    {
        memcpy(dataset, rho, sizeof(rho)*dim);
        for (int i = 0; i < dim; ++i)
        {
            dataset[i] = min(clamp_max, dataset[i]);
            dataset[i] = max(clamp_min, dataset[i]);
        }
    }
}

//visualize: This is the main visualization function
void visualize(void)
{
	int        i, j, idx;
	fftw_real  wn = (fftw_real)gridWidth  / (fftw_real)(DIM + 1);   // Grid cell width
	fftw_real  hn = (fftw_real)gridHeight / (fftw_real)(DIM + 1);  // Grid cell heigh

    fftw_real min_v, max_v;
    size_t dim = DIM * 2*(DIM/2+1);
    fftw_real* dataset = (fftw_real*) calloc(dim, sizeof(fftw_real));
    prepare_dataset(dataset, &min_v, &max_v); //scale, clamp or compute magnitude for the required dataset

	if (draw_smoke)
	{
		int idx0, idx1, idx2, idx3;
		double px0, py0, px1, py1, px2, py2, px3, py3;
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glBegin(GL_TRIANGLES);
		for (j = 0; j < DIM - 1; j++)            //draw smoke
		{
			for (i = 0; i < DIM - 1; i++)
			{
				px0 = wn + (fftw_real)i * wn;
				py0 = hn + (fftw_real)j * hn;
				idx0 = (j * DIM) + i;


				px1 = wn + (fftw_real)i * wn;
				py1 = hn + (fftw_real)(j + 1) * hn;
				idx1 = ((j + 1) * DIM) + i;


				px2 = wn + (fftw_real)(i + 1) * wn;
				py2 = hn + (fftw_real)(j + 1) * hn;
				idx2 = ((j + 1) * DIM) + (i + 1);


				px3 = wn + (fftw_real)(i + 1) * wn;
				py3 = hn + (fftw_real)j * hn;
				idx3 = (j * DIM) + (i + 1);

                fftw_real v0, v1, v2, v3;

                v0 = dataset[idx0];
                v1 = dataset[idx1];
                v2 = dataset[idx2];
                v3 = dataset[idx3];


                set_colormap(v0);    glVertex2f(px0, py0);
                set_colormap(v1);    glVertex2f(px1, py1);
                set_colormap(v2);    glVertex2f(px2, py2);


                set_colormap(v0);    glVertex2f(px0, py0);
                set_colormap(v2);    glVertex2f(px2, py2);
                set_colormap(v3);    glVertex2f(px3, py3);

			}
		}
		glEnd();
	}

	if (draw_vecs)
	{
		glBegin(GL_LINES);				//draw velocities
		for (i = 0; i < DIM; i++)
			for (j = 0; j < DIM; j++)
			{
				idx = (j * DIM) + i;
				direction_to_color(vx[idx],vy[idx],color_dir);
				glVertex2f(wn + (fftw_real)i * wn, hn + (fftw_real)j * hn);
				glVertex2f((wn + (fftw_real)i * wn) + vec_scale * vx[idx], (hn + (fftw_real)j * hn) + vec_scale * vy[idx]);
			}
		glEnd();
	}
    draw_legend(min_v, max_v);
}

//------ INTERACTION CODE STARTS HERE -----------------------------------------------------------------

//display: Handle window redrawing events. Simply delegates to visualize().
void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
    visualize();

	glFlush();
	glutSwapBuffers();
}

//reshape: Handle window resizing (reshaping) events
void reshape(int w, int h)
{
	glViewport(0.0f, 0.0f, (GLfloat)w, (GLfloat)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
	winWidth = w; winHeight = h;
    gridWidth = winWidth - legend_size - legend_text_len;
    gridHeight = winHeight;
}

//keyboard: Handle key presses
void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 't': dt -= 0.001; break;
		case 'T': dt += 0.001; break;
		case 'c': color_dir = 1 - color_dir; break;
        case 'd': display_dataset = !display_dataset; break;
		case 'S': vec_scale *= 1.2; break;
		case 's': vec_scale *= 0.8; break;
		case 'V': visc *= 5; break;
		case 'v': visc *= 0.2; break;
        case 'i': apply_mode = !apply_mode; break;
		case 'x':
			draw_smoke = 1 - draw_smoke;
			if (draw_smoke==0) draw_vecs = 1;
			break;
		case 'y':
			draw_vecs = 1 - draw_vecs;
			if (draw_vecs==0) draw_smoke = 1;
			break;
		case 'm':
			scalar_col++;
			if (scalar_col>COLOR_HEATMAP)
				scalar_col=COLOR_BLACKWHITE;
			break;
		case 'a': frozen = 1-frozen; break;
        case '+': n_colors = min(256, n_colors+1); break;
        case '-': n_colors = max(5, n_colors-1); break;
        case '[': clamp_min = max(0, clamp_min-0.1f); break;
        case ']': clamp_min = clamp_min+0.1f; break;
        case '{': clamp_max = max(0, clamp_max-0.1f); break;
        case '}': clamp_max = clamp_max+0.1f; break;
		case 'q': exit(0);
	}
}



// drag: When the user drags with the mouse, add a force that corresponds to the direction of the mouse
//       cursor movement. Also inject some new matter into the field at the mouse location.
void drag(int mx, int my)
{
	int xi,yi,X,Y;
	double  dx, dy, len;
	static int lmx=0,lmy=0;				//remembers last mouse location

	// Compute the array index that corresponds to the cursor location
	xi = (int)clamp((double)(DIM + 1) * ((double)mx / (double)gridWidth));
	yi = (int)clamp((double)(DIM + 1) * ((double)(gridHeight - my) / (double)gridHeight));

	X = xi;
	Y = yi;

	if (X > (DIM - 1))
		X = DIM - 1;
	if (Y > (DIM - 1))
		Y = DIM - 1;
	if (X < 0)
		X = 0;
	if (Y < 0)
		Y = 0;

	// Add force at the cursor location
	my = winHeight - my;
	dx = mx - lmx;
	dy = my - lmy;
	len = sqrt(dx * dx + dy * dy);
	if (len != 0.0)
	{
		dx *= 0.1 / len;
		dy *= 0.1 / len;
	}
	fx[Y * DIM + X] += dx;
	fy[Y * DIM + X] += dy;
	rho[Y * DIM + X] = 10.0f;
	lmx = mx;
	lmy = my;
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

	glutInitWindowSize(550,500);
    glutInitWindowPosition( 50, 50 );
    main_window = glutCreateWindow("Real-time smoke simulation and visualization");

	glutInitWindowSize(650,500);
	glutCreateWindow("Real-time smoke simulation and visualization");

	glutDisplayFunc(display);

    GLUI_Master.set_glutIdleFunc(do_one_simulation_step);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(drag);


    //Testing UI stuff
    //GLUI *glui = GLUI_Master.create_glui( "GLUI" );
    GLUI *glui = glui_v_subwindow = GLUI_Master.create_glui_subwindow
            (main_window, GLUI_SUBWINDOW_LEFT);
    GLUI_Panel *colormap_panel = new GLUI_Panel( glui, "Colour map type" );
    c_Map = new GLUI_RadioGroup(colormap_panel, (&cMap), 4, control_cb);
    new GLUI_RadioButton( c_Map, "Greyscale" );
    new GLUI_RadioButton( c_Map, "Rainbow" );
    new GLUI_RadioButton( c_Map, "something weird" );
    new GLUI_RadioButton( c_Map, "Red-To-White" );

    GLUI_Panel *dataset_panel = new GLUI_Panel( glui, "Dataset to be Mapped" );
    dataset = new GLUI_RadioGroup(dataset_panel, (&dSet), 4, control_cb);
    new GLUI_RadioButton( dataset, "Density" );
    new GLUI_RadioButton( dataset, "Velocity" );
    new GLUI_RadioButton( dataset, "Force" );

    (new GLUI_Spinner( glui, "Number of colours", &segments ))
            ->set_int_limits( 3, 60 );
    new GLUI_Button( glui, "QUIT", 0,(GLUI_Update_CB)exit );
    glui->set_main_gfx_window( main_window );
    glutReshapeFunc(reshape);
//    temp_foo = getFoo(5);
//    type = *((int *)temp_foo);
//    printf(":::%d:::",type);
//    printString(temp_foo);

	init_simulation(DIM);	//initialize the simulation data structures

	glutMainLoop();			//calls do_one_simulation_step, keyboard, display, drag, reshape
	return 0;
}
