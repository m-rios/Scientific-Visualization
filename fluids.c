// Usage: Drag with the mouse to add smoke to the fluid. This will also move a "rotor" that disturbs
//        the velocity field at the mouse location. Press the indicated keys to change options
//--------------------------------------------------------------------------------------------------

#include <rfftw.h>              //the numerical simulation FFTW library
#include <math.h>               //for various math functions

#include "GL/glui.h"

#include <float.h>
#include <assert.h>


#ifdef LINUX
#include <GL/glut.h>            //the GLUT graphics library
#endif

#ifdef MACOS
#include <GLUT/glut.h>            //the GLUT graphics library
#endif


//--- SIMULATION PARAMETERS ------------------------------------------------------------------------
const int DIM = 50;				//size of simulation grid
double dt = 0.4;				//simulation time step
float visc = 0.001;			//fluid viscosity
fftw_real *vx, *vy;             //(vx,vy)   = velocity field at the current moment
fftw_real *vx0, *vy0;           //(vx0,vy0) = velocity field at the previous moment
fftw_real *fx, *fy;	            //(fx,fy)   = user-controlled simulation forces, steered with the mouse
fftw_real *rho, *rho0;			//smoke density at the current (rho) and previous (rho0) moment
rfftwnd_plan plan_rc, plan_cr;  //simulation domain discretization

//--- VISUALIZATION PARAMETERS ---------------------------------------------------------------------
int   winWidth, winHeight;      //size of the graphics window, in pixels
int   gridWidth, gridHeight;    //size of the simulation grid in pixels
fftw_real  wn, hn;              //size of the grid cell in pixels
int   color_dir = 0;            //use direction color-coding or not
float vec_scale = 400;			//scaling of hedgehogs
int   draw_smoke = 1;           //draw the smoke or not
int   draw_vecs = 0;            //draw the vector field or not
int draw_iLines = 0;
const int COLOR_BLACKWHITE=0;   //different types of color mapping: black-and-white, rainbow, banded
const int COLOR_RAINBOW=1;
const int COLOR_HEATMAP=2;
int   scalar_col = 1;           //method for scalar coloring
int   frozen = 0;               //toggles on/off the animation
//todo: to be dynamically set by the user in the UI
float isoValue = 0.80;          //Isovalue set by the user
int n_values = 0;
int   legend_size = 50;

int   n_colors = 256;           //number of colors used in the color map

int   legend_text_len = 85;
const int DATASET_DENSITY=0;    //Density is the dataset to be displayed
const int DATASET_VELOCITY=1;   //Velocity is the dataset to be displayed
const int DATASET_FORCE=2;      //Force is the dataset to be displayed
int   display_dataset = 0;      // The dataset to be displayed
const int APPLY_SCALING = 0;    //Use the scaling method to apply the color map to the dataset
const int APPLY_CLAMP = 1;      //Use the clamping method for applying the color map to the dataset
int apply_mode = 1;
float clamp_min = 0.0f;
float clamp_max = 1.0f;
unsigned int	textureID[3];
int texture_mapping = 1;
int display_divergence = 0;
int dynamic_scalling = 0;
int height_plot = 0;

//--- GLUI PARAMETERS ------------------------------------------------------------------------------
GLUI_RadioGroup *colormap_radio;
GLUI_RadioGroup *glyphtype;
GLUI_RadioGroup *dataset_radio;
GLUI_RadioGroup *glyph_radio;
GLUI_Button *button;
int sGlyph = 0;
int vGlyph = 0;
int typeGlyph = 0;
bool glyph = false;
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

void reset_simulation(void)
{
    init_simulation(DIM);
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
        //if ( glutGetWindow() != main_window ) glutSetWindow(main_window);
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
    vy/= n_colors-1;
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
    vy = conform_to_bands(vy);
	if (scalar_col==COLOR_BLACKWHITE)
        R = G = B = vy;
	else if (scalar_col==COLOR_RAINBOW)
		rainbow(vy,&R,&G,&B);
    else if (scalar_col==COLOR_HEATMAP)
        heatmap(vy, &R, &G, &B);

	glColor3f(R,G,B);
}

void get_colormap(float vy, float *R, float *G, float *B)
{
    vy = conform_to_bands(vy);
    if (scalar_col==COLOR_BLACKWHITE)
        *R = *G = *B = vy;
    else if (scalar_col==COLOR_RAINBOW)
        rainbow(vy,R,G,B);
    else if (scalar_col==COLOR_HEATMAP)
        heatmap(vy, R, G, B);
}

//draw text at x, y location
void draw_text(const char* text, int x, int y)
{
    int len = strlen( text );

    glColor3f( 1.0f, 1.0f, 1.0f );
    glPushMatrix();
    glTranslatef(x, y, 0.0f);
    glScalef(0.15, 0.15, 0.15);
    for( int i = 0; i < len; i++ ) {
//        glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, text[i] );
        glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, (int) text[i]);
    }
    glPopMatrix();
}

void draw_legend(fftw_real min_v, fftw_real max_v)
{
    float step = (float) (winHeight - 2 *hn) / (float) n_colors;

    if (texture_mapping) glEnable(GL_TEXTURE_1D);

    for (int j = 0; j < n_colors; ++j)
    {
        //Normalise value (j) to [0,1]

        float v = (float) j/((float) (n_colors-1));

        float y0 = hn+step*j;
        float x0 = gridWidth; //do not hardcode legend size
        float y1 = hn+step*(j+1);
        float x1 = gridWidth + legend_size;

        if (texture_mapping)
        {
            glTexCoord1f(v);
        }
        else
        {
            set_colormap(v);
        }

        glRecti(x0,y0,x1,y1);

    }
    if (texture_mapping) glDisable(GL_TEXTURE_1D);
    char string[48];
    snprintf (string, sizeof(string), "%1.3f", min_v);
    draw_text(string, winWidth-legend_text_len, hn);
    snprintf (string, sizeof(string), "%1.3f", max_v);
    draw_text(string, winWidth-legend_text_len, (DIM-1)*hn);
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
        if (dataset[i] <= *min_v) *min_v = dataset[i];
        if (dataset[i] >= *max_v) *max_v = dataset[i];
    }

}

//compute_divergence: computes from the x,y vector field the divergence and assigns it to dataset
void compute_divergence(fftw_real *x, fftw_real *y, fftw_real *dataset)
{
//    fftw_real  wn = (fftw_real)gridWidth  / (fftw_real)(DIM + 1);
//    fftw_real  hn = (fftw_real)gridHeight / (fftw_real)(DIM + 1);
    for (int j = 0; j < DIM - 1; j++)
    {
        for (int i = 0; i < DIM - 1; i++)
		{
            int next_x = (j * DIM) + (i + 1); //next in x
            int next_y = ((j + 1) * DIM) + i; //next cell in y

            int current = (j * DIM) + i;
			dataset[current] = ((x[next_x] - x[current])/wn + (y[next_y] - y[current])/hn)*1000;	//Divergence operator
        }
    }
}

//prepare_dataset: perform the required transformations in order to make dataset ready to display (clamping, scalling...)
void prepare_dataset(fftw_real* dataset, fftw_real* min_v, fftw_real* max_v)
{
    size_t dim = DIM * 2*(DIM/2+1);

    assert(display_dataset == DATASET_DENSITY || display_dataset == DATASET_VELOCITY || display_dataset == DATASET_FORCE);

    //Copy proper dataset source
    if (display_dataset == DATASET_DENSITY)
    {
        memcpy(dataset, rho, dim * sizeof(rho));
    }
    else if (display_dataset == DATASET_VELOCITY)
    {
		if (display_divergence)
		{
			compute_divergence(vx, vy, dataset);
		}
		else
		{
			for (int i = 0; i < dim; ++i)
				dataset[i] = sqrt(pow(vx[i],2) + pow(vy[i],2));
		}
    }
    else if (display_dataset == DATASET_FORCE)
    {
		if (display_divergence)
		{
			compute_divergence(fx, fy, dataset);
		}
		else
		{
			for (int i = 0; i < dim; ++i)
				dataset[i] = sqrt(pow(fx[i],2) + pow(fy[i],2));
		}
    }

    //Apply transformation
    if (apply_mode == APPLY_SCALING)    //Apply scaling
    {
		if (!dynamic_scalling) return;
        find_min_max(min_v, max_v, dataset);
        for (int i = 0; i < dim; ++i)
            dataset[i] = scale(*min_v, *max_v, dataset[i]);
    }
    else if (apply_mode == APPLY_CLAMP) //Apply clamping
    {
        for (int i = 0; i < dim; ++i)
        {
            dataset[i] = min(clamp_max, dataset[i]);
            dataset[i] = max(clamp_min, dataset[i]);
            dataset[i] = scale(clamp_min, clamp_max, dataset[i]);
            *min_v = (fftw_real) clamp_min;
            *max_v = (fftw_real) clamp_max;
        }
    }
}

void draw_grid()
{
    glBegin(GL_LINES);
    glColor3f(1, 1, 1); //Lines are white
    // Draw vertical lines
    for (int i = 0; i < DIM; ++i) {
        glVertex3f(wn + (fftw_real) i *wn, hn, 0.0);
        glVertex3f((wn + (fftw_real) i * wn), hn*DIM, 0.0);
    }

    //Draw horizontal lines
    for (int j = 0; j < DIM; ++j) {
        glVertex3f(wn, hn + (fftw_real) j*hn, 0.0);
        glVertex3f(wn*DIM, hn + (fftw_real) j*hn, 0.0);
    }
    glEnd();
}

void draw_smoke_textures(void)
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_LIGHTING);
    glShadeModel(GL_SMOOTH);

    glEnable(GL_TEXTURE_1D);
    //     to the currently-active colormap.
    glBindTexture(GL_TEXTURE_1D,textureID[scalar_col]);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);

    int idx;

    fftw_real min_v, max_v;
    size_t dim = DIM * 2*(DIM/2+1);
    fftw_real* dataset = (fftw_real*) calloc(dim, sizeof(fftw_real));
    prepare_dataset(dataset, &min_v, &max_v);

    int idx0, idx1, idx2, idx3;
    double px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3;
    glBegin(GL_TRIANGLES);
    for (int j = 0; j < DIM - 1; j++)
    {
        for (int i = 0; i < DIM - 1; i++)
        {

            px0 = wn + (fftw_real)i * wn;
            py0 = hn + (fftw_real)j * hn;
            idx0 = (j * DIM) + i;
            if (height_plot) pz0 = dataset[idx0];

            px1 = wn + (fftw_real)i * wn;
            py1 = hn + (fftw_real)(j + 1) * hn;
            idx1 = ((j + 1) * DIM) + i;
            if (height_plot) pz1 = dataset[idx1];

            px2 = wn + (fftw_real)(i + 1) * wn;
            py2 = hn + (fftw_real)(j + 1) * hn;
            idx2 = ((j + 1) * DIM) + (i + 1);
            if (height_plot) pz2 = dataset[idx2];

            px3 = wn + (fftw_real)(i + 1) * wn;
            py3 = hn + (fftw_real)j * hn;
            idx3 = (j * DIM) + (i + 1);
            if (height_plot) pz3 = dataset[idx3];

            fftw_real v0, v1, v2, v3;

            v0 = dataset[idx0];
            v1 = dataset[idx1];
            v2 = dataset[idx2];
            v3 = dataset[idx3];

            glTexCoord1f(v0);    glVertex3f(px0,py0, pz0);
            glTexCoord1f(v1);    glVertex3f(px1,py1, pz1);
            glTexCoord1f(v2);    glVertex3f(px2,py2, pz2);
//            glTexCoord1f(v3);    glVertex2f(px3,py3);

            glTexCoord1f(v0);    glVertex3f(px0,py0, pz0);
            glTexCoord1f(v3);    glVertex3f(px3,py3, pz3);
            glTexCoord1f(v2);    glVertex3f(px2,py2, pz2);

        }
    }
    glEnd();
    glDisable(GL_TEXTURE_1D);
    draw_legend(min_v, max_v);
}

void draw_smoke_default()
{
    int        i, j, idx;
//    fftw_real  wn = (fftw_real)gridWidth  / (fftw_real)(DIM + 1);   // Grid cell width
//    fftw_real  hn = (fftw_real)gridHeight / (fftw_real)(DIM + 1);  // Grid cell heigh

    fftw_real min_v, max_v;
    size_t dim = DIM * 2*(DIM/2+1);
    fftw_real* dataset = (fftw_real*) calloc(dim, sizeof(fftw_real));
    prepare_dataset(dataset, &min_v, &max_v); //scale, clamp or compute magnitude for the required dataset

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
    draw_legend(min_v, max_v);
}

float intersection_point(float pi,float pj,float vi, float vj)
{

    float q = 0.0;
        if (vj > vi) {
            q = ((pi * (vj - isoValue)) + (pj * (isoValue - vi)))/ (vj - vi);
        } else {
            q = ((pj * (vi - isoValue)) + (pi * (isoValue - vj))) / (vi - vj);
        }
        //printf("Q value %f between %f and %f:: \n",q,pi,pj);
        return (q);

}

int isoDecider(float j, float k) {
//    if(((j > k)?((j - k)> 0.0001 && (j - k) < 10000):(k - j) > 0.0001 && (k - j) < 10000))
//    {
        if ((j >= isoValue && k <= isoValue) || (j <= isoValue && k >= isoValue)){
            return (1);
        } else return (0);

//    } else return (0);
}

void compute_isolines()
{
    //This is to be implemented only for density
    //printf("Begin drawing ISOlines");
    //todo: Need to find out why the contour is patchy or how to fill the missing links.
    fftw_real min_v, max_v;
    size_t dim = DIM * 2*(DIM/2+1);
    fftw_real* dataset = (fftw_real*) calloc(dim, sizeof(fftw_real));
    prepare_dataset(dataset, &min_v, &max_v);
    int lookuptable[(DIM)*(DIM)][4];
    int yIdx = 0 ,dec = 0,temp = 0;
    int        i, j;
    for (i = 0;i < ((DIM -1 )*(DIM -1)); i++) // initialized the array to zeros
        for (j = 0;j<4;j++) {
            lookuptable[i][j] = 0;
        }
    fftw_real  wn = (fftw_real)gridWidth  / (fftw_real)(DIM + 1);   // Grid cell width
    fftw_real  hn = (fftw_real)gridHeight / (fftw_real)(DIM + 1);  // Grid cell heigh

    //Check the marching squares patterns and determine the sides to join.
    glBegin(GL_LINES);
    glColor3f(0.0,0.0,0.0);
    for (j = 0;j < DIM - 1; j++) {
        for (i = 0; i < DIM - 1; i++) {

            int v0x = wn + (fftw_real)i * wn;
            int v0y = hn + (fftw_real)j * hn;
            int idx0 = (j * DIM) + i;


            int v1x = wn + (fftw_real)i * wn;
            int v1y = hn + (fftw_real)(j + 1) * hn;
            int idx1 = ((j + 1) * DIM) + i;


            int v2x = wn + (fftw_real)(i + 1) * wn;
            int v2y = hn + (fftw_real)(j + 1) * hn;
            int idx2 = ((j + 1) * DIM) + (i + 1);


            int v3x = wn + (fftw_real)(i + 1) * wn;
            int v3y = hn + (fftw_real)j * hn;
            int idx3 = (j * DIM) + (i + 1);

            fftw_real r0, r1, r2, r3;

            r0 = rho[idx0] ;
            r1 = rho[idx1] ;
            r2 = rho[idx2] ;
            r3 = rho[idx3] ;

            if (r0 > isoValue)
            {lookuptable[i + j][yIdx] = 1;}
            if (r1 > isoValue)
            {lookuptable[i + j][yIdx +1] = 1;}
            if (r2 > isoValue)
            {lookuptable[i + j][yIdx + 2] = 1;}
            if (r3 > isoValue)
            {lookuptable[i + j][yIdx + 3] = 1;}

            for (int k=0;k<4;k++)
            {
                temp = lookuptable[i+j][k];
                dec = dec + temp * pow(2, k);
            }
// Dont remove the below section, handy for viewing the grid
//            glVertex2f(v0x,v0y);
//            glVertex2f(v1x,v1y);
//            glVertex2f(v2x,v2y);
//            glVertex2f(v3x,v3y);
//
//            glVertex2f(v0x,v0y);
//            glVertex2f(v3x,v3y);
//            glVertex2f(v1x,v1y);
//            glVertex2f(v2x,v2y);
            switch (dec) {
                case 1 :
                case 14:
                    if (isoDecider(r0,r3) && isoDecider(r2,r3)) {

                        glVertex2f(intersection_point(v0x, v3x, r0, r3),
                                   intersection_point(v0y, v3y, r0, r3));
                        glVertex2f(intersection_point(v2x, v3x, r2, r3),
                                   intersection_point(v2y, v3y, r2, r3));
                    }
                    break;

                case 2 :
                case 13:
                    if (isoDecider(r1,r2) && isoDecider(r3,r2)) {
                        glVertex2f(intersection_point(v1x, v2x, r1, r2),
                                   intersection_point(v1y, v2y, r1, r2));
                        glVertex2f(intersection_point(v3x, v2x, r3, r2),
                                   intersection_point(v3y, v2y, r3, r2));
                    }
                    break;
                case 3 :
                case 12:
                    if (isoDecider(r0,r3) && isoDecider(r1,r2)) {
                        glVertex2f(intersection_point(v0x, v3x, r0, r3),
                                   intersection_point(v0y, v3y, r0, r3));
                        glVertex2f(intersection_point(v1x, v2x, r1, r2),
                                   intersection_point(v1y, v2y, r1, r2));
                    }
                    break;
                case 4 :
                case 11:
                    if (isoDecider(r0,r1) && isoDecider(r2,r1)) {
                        glVertex2f(intersection_point(v0x, v1x, r0, r1),
                                   intersection_point(v0y, v1y, r0, r1));
                        glVertex2f(intersection_point(v2x, v1x, r2, r1),
                                   (intersection_point(v2y, v1y, r2, r1)));
                    }
                    break;
                case 5 :
                case 10:

                    srand(time(NULL));
                    if ((rand()%10) >= 5) {
                        if ((isoDecider(r1, r0) && isoDecider(r1, r2)) &&
                            (isoDecider(r3, r1) && isoDecider(r3, r2))) {
                                glVertex2f(intersection_point(v1x, v0x, r1, r0),
                                           intersection_point(v1y, v0y, r1, r0));
                                glVertex2f(intersection_point(v1x,v2x,r1,r2),
                                           intersection_point(v1y,v2y,r1,r2));
                                glVertex2f(intersection_point(v3x,v1x,r3,r1),
                                           intersection_point(v3y,v1y,r3,r1));
                                glVertex2f(intersection_point(v3x,v2x,r3,r2),
                                           intersection_point(v3y,v2y,r3,r2));
                            }
                    } else if ((isoDecider(r1,r0) && isoDecider(r3,r0)) &&
                               (isoDecider(r1,r2)) && isoDecider(r3,r2)) {

                                glVertex2f(intersection_point(v1x,v0x,r1,r0),
                                           intersection_point(v1y,v0y,r1,r0));
                                glVertex2f(intersection_point(v3x,v0x,r3,r0),
                                           intersection_point(v3y,v0y,r3,r0));
                                glVertex2f(intersection_point(v1x,v2x,r1,r2),
                                           intersection_point(v1y,v2y,r1,r2));
                                glVertex2f(intersection_point(v3x,v2x,r3,r2),
                                           intersection_point(v3y,v2y,r3,r2));
                    }
                    break;
                case 6 :
                case 9 :
                    if (isoDecider(r1,r0) && isoDecider(r2,r3)) {
                        glVertex2f(intersection_point(v1x, v0x, r1, r0),
                                   intersection_point(v1y, v0y, r1, r0));
                        glVertex2f(intersection_point(v2x, v3x, r2, r3),
                                   intersection_point(v2y, v3y, r2, r3));
                    }
                    break;
                case 7 :
                case 8 :
                    if (isoDecider(r1,r0) && isoDecider(r3,r0)) {
                        glVertex2f(intersection_point(v1x, v0x, r1, r0),
                                   intersection_point(v1y, v0y, r1, r0));
                        glVertex2f(intersection_point(v3x, v0x, r3, r0),
                                   intersection_point(v3y, v0y, r3, r0));
                    }
                    break;
                default:
                    break;//case 0 and 16

            }
            dec = 0;yIdx = 0;
        }
    }
    glEnd();
}

//visualize: This is the main visualization function
void visualize(void)
{
	int        i, j, idx;

    fftw_real min_v, max_v;
    size_t dim = DIM * 2*(DIM/2+1);
    fftw_real* dataset = (fftw_real*) calloc(dim, sizeof(fftw_real));
    prepare_dataset(dataset, &min_v, &max_v); //scale, clamp or compute magnitude for the required dataset

    if (height_plot) draw_grid();

	if (draw_smoke)
	{
        if (texture_mapping)
            draw_smoke_textures();
        else
            draw_smoke_default();
    }
    if (draw_iLines)
    {
        compute_isolines();
    }
	if (draw_vecs)
	{
		if (typeGlyph == 0 ) {
			glBegin(GL_LINES);                //draw velocities

			for (i = 0; i < DIM; i++) {
				for (j = 0; j < DIM; j++) {
					idx = (j * DIM) + i;
					direction_to_color(vx[idx], vy[idx], color_dir);
					//mapping the scalar value of the dataset with color.
					if (glyph ) {//User selects glyphs options
						set_colormap(dataset[idx]);
						if (vGlyph == 0)//fluid velocity
						{
							//(3.1415927 / 180.0) * angle;
							double magV = sqrt(pow(vx[idx], 2) + pow(vy[idx], 2));
							double angleV = atan2(vx[idx], vy[idx]);
							glVertex2f(wn + (fftw_real) i * wn, hn + (fftw_real) j * hn);
							glVertex2f((wn + (fftw_real) i * wn) + vec_scale * cos(angleV) * magV,
									   (hn + (fftw_real) j * hn) + vec_scale * sin(angleV) * magV);
						}

						if (vGlyph == 1)//force
						{
							double magF = sqrt(pow(fx[idx], 2) + pow(fy[idx], 2));
							double angleF = atan2(fx[idx], fy[idx]);
							glVertex2f(wn + (fftw_real) i * wn, hn + (fftw_real) j * hn);
							glVertex2f((wn + (fftw_real) i * wn) + vec_scale * cos(angleF) * magF,
									   (hn + (fftw_real) j * hn) + vec_scale * sin(angleF) * magF);
						}
					} else {
						glVertex2f(wn + (fftw_real) i * wn, hn + (fftw_real) j * hn);
						glVertex2f((wn + (fftw_real) i * wn) + vec_scale * vx[idx],
								   (hn + (fftw_real) j * hn) + vec_scale * vy[idx]);
					}
				}
			}
			glEnd();
        }
        else if (typeGlyph == 1) //Conical glyph section
        {
            for (j = 0; j < DIM; j++) {
                for (i = 0; i < DIM ; i++) {
                    idx = (j * DIM) + i;
                    set_colormap(dataset[idx]); //applying colourmapping
                    if (vGlyph == 0)//fluid velocity
                    {
                        double magV = sqrt(pow(vx[idx], 2) + pow(vy[idx], 2));
                        double angleV = atan2(vx[idx], vy[idx]);
                        double deg = angleV *(180/3.1415927);
                        glTranslatef(i*wn, j*hn, -5.0);
                        //todo:apply shading,scaling and direction correction
                        glRotatef(90,  0.0, 1.0, 0.0);
                        glRotatef(-deg, 1.0, 0.0, 0.0);
                        //glutSolidCone( GLdouble base, GLdouble height, GLint slices, GLint stacks );
                        glutSolidCone(5.0, magV * 500, 20, 20);
                        glLoadIdentity();
                    } else if (vGlyph == 1) // force
                    {
                        double magF = sqrt(pow(fx[idx], 2) + pow(fy[idx], 2));
                        double angleF = atan2(fx[idx], fy[idx]);
                        double deg = angleF *(180/3.1415927);
                        glutSolidCone(magF * 200, magF * 200, 20, 20);
                        glTranslatef(0.0,hn, 0.0);
                    }
                }
            }
        } else {                  //Arrow glyph section

        }
	}
    glLoadIdentity();
    draw_legend(min_v, max_v);

}

void create_textures()					//Create one 1D texture for each of the available colormaps.
{														//We will next use these textures to color map scalar data.

    glGenTextures(3,textureID);							//Generate 3 texture names, for the textures we will create
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);				//Make sure that OpenGL will understand our CPU-side texture storage format

    int selected_map = scalar_col;

    for(int i=COLOR_BLACKWHITE;i<=COLOR_HEATMAP;++i)
    {													//Generate all three textures:
        glBindTexture(GL_TEXTURE_1D,textureID[i]);		//Make i-th texture active (for setting it)
        const int size = 512;							//Allocate a texture-buffer large enough to store our colormaps with high resolution
        float textureImage[3*size];

        scalar_col = i;				//Activate the i-th colormap

        for(int j=0;j<size;++j)							//Generate all 'size' RGB texels for the current texture:
        {
            float v = float(j)/(size-1);				//Compute a scalar value in [0,1]
            float R,G,B;
            get_colormap(v, &R, &G, &B);						//Map this scalar value to a color, using the current colormap

            textureImage[3*j]   = R;					//Store the color for this scalar value in the texture
            textureImage[3*j+1] = G;
            textureImage[3*j+2] = B;
        }
        glTexImage1D(GL_TEXTURE_1D,0,GL_RGB,size,0,GL_RGB,GL_FLOAT,textureImage);
        //The texture is ready - pass it to OpenGL
    }

    scalar_col = selected_map;					//Reset the currently-active colormap to the default (first one)
}




//------ GLUI CODE STARTS HERE ------------------------------------------------------------------------

void radio_cb( int control )
{
    switch (control)
    {
        case RADIO_COLOR_MAP:   scalar_col = colormap_radio->get_int_val();         break;
        case RADIO_DATASET:     display_dataset = dataset_radio->get_int_val();     break;
        case RADIO_GLYPH:       vGlyph = (glyph_radio->get_int_val());              break;
    }
}

void color_bands_cb(int control)
{
    create_textures();
}

void glyph_button_cb(int control)
{
  printf( "callback: %d\n", control );
  glyph = true;
  if (glyph)
  {
   printf("Glyph view enabled");
  } else {
   printf ("Glyph view disabled");
  }
}
void glyphtype_cb(int control)
{
    typeGlyph = glyphtype->get_int_val();
}

void divergence_cb( int control )
{
    if (display_divergence)
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

//    glMatrixMode(GL_PROJECTION);
//    glLoadIdentity();
//    glFrustum(-1, 1, -1, 1, 1, 1000);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

//    gluLookAt ( 0.0, 0.0, 10.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0 );

    visualize();

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
    winWidth = w; winHeight = h;
    gridWidth = winWidth - legend_size - legend_text_len;
    gridHeight = winHeight;
    wn = (fftw_real)gridWidth  / (fftw_real)(DIM + 1);
    hn = (fftw_real)gridHeight / (fftw_real)(DIM + 1);
    glutPostRedisplay();
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
        case '[': clamp_min_spinner->set_float_val(max(0, clamp_min-0.1f)); break;
        case ']': clamp_min_spinner->set_float_val(clamp_min+0.1f); break;
        case '{': clamp_max_spinner->set_float_val(max(0, clamp_max-0.1f)); break;
        case '}': clamp_max_spinner->set_float_val(clamp_max+0.1f); break;
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
}

//add_matter: When the user drags with the left mouse button pressed, add a force that corresponds to the direction of
// the mouse cursor movement. Also inject some new matter into the field at the mouse location.
void add_matter(int mx, int my)
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

int orbit_view(int mx, int my)
{

}

// drag: select which action to map mouse drag to, depending on pressed button
void drag(int mx, int my)
{
	if (left_button == GLUT_DOWN && right_button == GLUT_DOWN) return;  //Do nothing if both buttons pressed.
    glutSetWindow(glui->get_glut_window_id());

//    mx -= 150;

    if (left_button == GLUT_DOWN)
        add_matter(mx, my);
    else if (right_button == GLUT_DOWN)
        orbit_view(mx, my);
}

//void myGlutMotion(int x, int y)
//{
//    // the change in mouse position
//    int dx = x-last_x;
//    int dy = y-last_y;
//
//    float scale, len, theta;
//    float neye[3], neye2[3];
//    float f[3], r[3], u[3];
//
//    switch(cur_button)
//    {
//        case GLUT_LEFT_BUTTON:
//            // translate
//            f[0] = lookat[0] - eye[0];
//            f[1] = lookat[1] - eye[1];
//            f[2] = lookat[2] - eye[2];
//            u[0] = 0;
//            u[1] = 1;
//            u[2] = 0;
//
//            // scale the change by how far away we are
//            scale = sqrt(length(f)) * 0.007;
//
//            crossproduct(f, u, r);
//            crossproduct(r, f, u);
//            normalize(r);
//            normalize(u);
//
//            eye[0] += -r[0]*dx*scale + u[0]*dy*scale;
//            eye[1] += -r[1]*dx*scale + u[1]*dy*scale;
//            eye[2] += -r[2]*dx*scale + u[2]*dy*scale;
//
//            lookat[0] += -r[0]*dx*scale + u[0]*dy*scale;
//            lookat[1] += -r[1]*dx*scale + u[1]*dy*scale;
//            lookat[2] += -r[2]*dx*scale + u[2]*dy*scale;
//
//            break;
//
//        case GLUT_MIDDLE_BUTTON:
//            // zoom
//            f[0] = lookat[0] - eye[0];
//            f[1] = lookat[1] - eye[1];
//            f[2] = lookat[2] - eye[2];
//
//            len = length(f);
//            normalize(f);
//
//            // scale the change by how far away we are
//            len -= sqrt(len)*dx*0.03;
//
//            eye[0] = lookat[0] - len*f[0];
//            eye[1] = lookat[1] - len*f[1];
//            eye[2] = lookat[2] - len*f[2];
//
//            // make sure the eye and lookat points are sufficiently far away
//            // push the lookat point forward if it is too close
//            if (len < 1)
//            {
//                printf("lookat move: %f\n", len);
//                lookat[0] = eye[0] + f[0];
//                lookat[1] = eye[1] + f[1];
//                lookat[2] = eye[2] + f[2];
//            }
//
//            break;
//
//        case GLUT_RIGHT_BUTTON:
//            // rotate
//
//            neye[0] = eye[0] - lookat[0];
//            neye[1] = eye[1] - lookat[1];
//            neye[2] = eye[2] - lookat[2];
//
//            // first rotate in the x/z plane
//            theta = -dx * 0.007;
//            neye2[0] = (float)cos(theta)*neye[0] + (float)sin(theta)*neye[2];
//            neye2[1] = neye[1];
//            neye2[2] =-(float)sin(theta)*neye[0] + (float)cos(theta)*neye[2];
//
//
//            // now rotate vertically
//            theta = -dy * 0.007;
//
//            f[0] = -neye2[0];
//            f[1] = -neye2[1];
//            f[2] = -neye2[2];
//            u[0] = 0;
//            u[1] = 1;
//            u[2] = 0;
//            crossproduct(f, u, r);
//            crossproduct(r, f, u);
//            len = length(f);
//            normalize(f);
//            normalize(u);
//
//            neye[0] = len * ((float)cos(theta)*f[0] + (float)sin(theta)*u[0]);
//            neye[1] = len * ((float)cos(theta)*f[1] + (float)sin(theta)*u[1]);
//            neye[2] = len * ((float)cos(theta)*f[2] + (float)sin(theta)*u[2]);
//
//            eye[0] = lookat[0] - neye[0];
//            eye[1] = lookat[1] - neye[1];
//            eye[2] = lookat[2] - neye[2];
//
//            break;
//    }
//
//
//    last_x = x;
//    last_y = y;
//
//    glutPostRedisplay();
//}

static void TimeEvent(int te)
{

    spin++;  // increase cube rotation by 1
    if (spin > 360) spin = 0; // if over 360 degress, start back at zero.
    glutPostRedisplay();  // Update screen with new rotation data
    glutTimerFunc( 100, TimeEvent, 1);  // Reset our timmer.
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
    winWidth = 900; winHeight = 900;
	glutInitWindowSize(winWidth,winHeight);
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
    colormap_radio = new GLUI_RadioGroup(colormap_panel, (&scalar_col), RADIO_COLOR_MAP, radio_cb);
    new GLUI_RadioButton( colormap_radio, "Greyscale" );
    new GLUI_RadioButton( colormap_radio, "Rainbow" );
    new GLUI_RadioButton( colormap_radio, "Heatmap" );

    GLUI_Panel *dataset_panel = new GLUI_Panel( glui, "Dataset to be Mapped" );
    dataset_radio = new GLUI_RadioGroup(dataset_panel, (&display_dataset), RADIO_DATASET, radio_cb);
    new GLUI_RadioButton( dataset_radio, "Density" );
    new GLUI_RadioButton( dataset_radio, "Velocity" );
    new GLUI_RadioButton( dataset_radio, "Force" );

    GLUI_Panel *glyphvector_panel = new GLUI_Panel( glui, "vector value for glyph" );
    glyph_radio = new GLUI_RadioGroup(glyphvector_panel , (&vGlyph), RADIO_GLYPH, radio_cb);
    new GLUI_RadioButton( glyph_radio, "fluid velocity" );
    new GLUI_RadioButton( glyph_radio, "force" );

	GLUI_Panel *glyphtype_panel = new GLUI_Panel( glui, "choose the type of glyph" );
	glyphtype = new GLUI_RadioGroup(glyphtype_panel , (&typeGlyph), 6, glyphtype_cb);
	new GLUI_RadioButton( glyphtype, "default" );
	new GLUI_RadioButton( glyphtype, "cones" );
	new GLUI_RadioButton( glyphtype, "arrows" );

    GLUI_Panel *clamping_panel = new GLUI_Panel( glui, "Clamping options" );
    glui->add_checkbox_to_panel(clamping_panel, "Apply clamping", &apply_mode);
    clamp_max_spinner = glui->add_spinner_to_panel(clamping_panel, "Clamp max", GLUI_SPINNER_FLOAT, &clamp_max);
    clamp_min_spinner = glui->add_spinner_to_panel(clamping_panel, "Clamp min", GLUI_SPINNER_FLOAT, &clamp_min);
    clamp_max_spinner->set_int_limits(0, 1);
    clamp_min_spinner->set_int_limits(-1, 1);
    clamp_max_spinner->set_float_val(1.0f);
    clamp_min_spinner->set_float_val(0.0f);

    glui->add_checkbox("Use texture mapping", &texture_mapping);
    glui->add_checkbox("Dynamic scaling", &dynamic_scalling);
	glui->add_checkbox("Show divergence", &display_divergence, -1, divergence_cb);
    glui->add_checkbox("Render smoke", &draw_smoke);
    glui->add_checkbox("Render glyphs", &draw_vecs);
    glui->add_checkbox("Height plot", &height_plot, -1, enable_height_plot);

    glui->add_checkbox("Render Isolines", &draw_iLines);
    printf("Clamp max initial value: %f",clamp_max);

//    (new GLUI_Spinner( glui, "Number of colours", &n_colors ))->set_int_limits( 3, 256 );
    GLUI_Spinner *color_bands_spinner = glui->add_spinner("Number of colours", GLUI_SPINNER_INT, &n_colors, -1, color_bands_cb);
    color_bands_spinner->set_int_limits(3, 256);

    glui->add_button("Reset", -1, (GLUI_Update_CB) reset_simulation);

    new GLUI_Button( glui, "QUIT", 0,(GLUI_Update_CB)exit );


    glui->set_main_gfx_window( main_window );
    GLUI_Master.set_glutIdleFunc(do_one_simulation_step);

//    glui->hide();

	init_simulation(DIM);	//initialize the simulation data structures
    create_textures();

	glutMainLoop();			//calls do_one_simulation_step, keyboard, display, drag, reshape
	return 0;
}
