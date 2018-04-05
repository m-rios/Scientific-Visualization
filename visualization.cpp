//
// Created by mario on 28-3-18.
//

#include "visualization.h"

Visualization::Visualization(int DIM)
{
    sim = new Simulation(DIM);

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
}

//rainbow: Implements a color palette, mapping the scalar 'value' to a rainbow color RGB
void Visualization::rainbow(float value,float* R,float* G,float* B)
{
	const float dx=0.8;
	if (value<0)
		value=0;
	if (value>1)
		value=1;
	value = (6-2*dx)*value+dx;
	*R = sim->max(0.0,(3-fabs(value-4)-fabs(value-5))/2);
	*G = sim->max(0.0,(4-fabs(value-2)-fabs(value-4))/2);
	*B = sim->max(0.0,(3-fabs(value-1)-fabs(value-2))/2);
}

//heatmap: Implements a heatmap color palette (Black-Red-Yellow-White).
void Visualization::heatmap(float value, float* R, float* G, float* B)
{
    //Clamp value between 0 and 1
    if (value<0)
        value=0;
    if (value>1)
        value=1;

    //For now we don't mess with S,V, only with Hue
    //Normalise value to [0,3] 0->Black, 1->Full red, 2-> Orange, 3->Full yellow.
    value *= 4;
    *R = sim->max(0, 1 - fabs(value/2-1)) + sim->max(0, 1 - fabs(value/2-2));
    *B = sim->max(0, 1 - fabs(value - 4));
    *G = sim->max(0, 1 - fabs(value-3)) + sim->max(0, 1 - fabs(value-4));
}

float Visualization::conform_to_bands(float vy)
{
    vy *= n_colors;
    vy = (int)(vy);
    vy/= n_colors-1;
    return vy;
}

fftw_real Visualization::scale(fftw_real min, fftw_real max, fftw_real value)
{
    return (value - min)/(max-min);
}

//set_colormap: Sets different types of colormaps
void Visualization::set_colormap(float vy)
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

void Visualization::get_colormap(float vy, float *R, float *G, float *B)
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
void Visualization::draw_text(const char* text, int x, int y)
{
    int len = strlen( text );

    glColor3f( 1.0f, 1.0f, 1.0f );
    glPushMatrix();
    glTranslatef(x, y, 0.0f);
    glScalef(0.15, 0.15, 0.15);
    for( int i = 0; i < len; i++ ) {
        glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, (int) text[i]);
    }
    glPopMatrix();
}

void Visualization::draw_legend(fftw_real min_v, fftw_real max_v)
{
    //Draw on the image plane
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, (GLdouble)winWidth, 0.0, (GLdouble)winHeight, -10.0, 10.0);

    float step = (float) (winHeight - 2 *hn) / (float) n_colors;

    glEnable(GL_TEXTURE_1D);

    for (int j = 0; j < n_colors; ++j)
    {
        //Normalise value (j) to [0,1]

        float v = (float) j/((float) (n_colors-1));

        float y0 = hn+step*j;
        float x0 = gridWidth; //do not hardcode legend size
        float y1 = hn+step*(j+1);
        float x1 = gridWidth + legend_size;

        glTexCoord1f(v);

        glRecti(x0,y0,x1,y1);

    }
    glDisable(GL_TEXTURE_1D);
    char string[48];
    snprintf (string, sizeof(string), "%1.3f", min_v);
    draw_text(string, winWidth-legend_text_len, hn);
    snprintf (string, sizeof(string), "%1.3f", max_v);
    draw_text(string, winWidth-legend_text_len, (sim->DIM-1)*hn);
}


//direction_to_color: Set the current color by mapping a direction vector (x,y), using
//                    the color mapping method 'method'. If method==1, map the vector direction
//                    using a rainbow colormap. If method==0, simply use the white color
void Visualization::direction_to_color(float x, float y, int method)
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
void Visualization::find_min_max(fftw_real* min_v, fftw_real* max_v, fftw_real* dataset)
{
    *max_v = FLT_MIN;
    *min_v = FLT_MAX;
    for (int i = 0; i < sim->DIM; ++i) {
        if (dataset[i] <= *min_v) *min_v = dataset[i];
        if (dataset[i] >= *max_v) *max_v = dataset[i];
    }

}

//compute_divergence: computes from the x,y vector field the divergence and assigns it to dataset
void Visualization::compute_divergence(fftw_real *x, fftw_real *y, fftw_real *dataset)
{
    for (int j = 0; j < sim->DIM - 1; j++)
    {
        for (int i = 0; i < sim->DIM - 1; i++)
		{
            int next_x = (j * sim->DIM) + (i + 1); //next in x
            int next_y = ((j + 1) * sim->DIM) + i; //next cell in y

            int current = (j * sim->DIM) + i;
			dataset[current] = ((x[next_x] - x[current])/wn + (y[next_y] - y[current])/hn)*1000;	//Divergence operator
        }
    }
}

//prepare_dataset: perform the required transformations in order to make dataset ready to display (clamping, scalling...)
void Visualization::prepare_dataset(fftw_real* dataset, fftw_real* min_v, fftw_real* max_v)
{
    size_t dim = sim->DIM * 2*(sim->DIM/2+1);

    assert(display_dataset == DATASET_DENSITY || display_dataset == DATASET_VELOCITY || display_dataset == DATASET_FORCE);

    //Copy proper dataset source
    if (display_dataset == DATASET_DENSITY)
    {
        memcpy(dataset, sim->rho, dim * sizeof(sim->rho));
    }
    else if (display_dataset == DATASET_VELOCITY)
    {
		if (display_divergence)
		{
			compute_divergence(sim->vx, sim->vy, dataset);
		}
		else
		{
			for (int i = 0; i < dim; ++i)
				dataset[i] = sqrt(pow(sim->vx[i],2) + pow(sim->vy[i],2));
		}
    }
    else if (display_dataset == DATASET_FORCE)
    {
		if (display_divergence)
		{
			compute_divergence(sim->fx, sim->fy, dataset);
		}
		else
		{
			for (int i = 0; i < dim; ++i)
				dataset[i] = sqrt(pow(sim->fx[i],2) + pow(sim->fy[i],2));
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
            dataset[i] = sim->min(clamp_max, dataset[i]);
            dataset[i] = sim->max(clamp_min, dataset[i]);
            dataset[i] = scale(clamp_min, clamp_max, dataset[i]);
            *min_v = (fftw_real) clamp_min;
            *max_v = (fftw_real) clamp_max;
        }
    }
}

void Visualization::draw_3d_grid()
{
    glBegin(GL_LINES);
    glColor3f(1,1,1);

    //Draw bottom grid
    // Draw vertical lines
    for (int i = 0; i < sim->DIM; ++i)
    {
        glVertex3f(wn + (fftw_real) i *wn, hn, 0.0);
        glVertex3f((wn + (fftw_real) i * wn), hn*sim->DIM, 0.0);
    }

    //Draw horizontal lines
    for (int j = 0; j < sim->DIM; ++j)
    {
        glVertex3f(wn, hn + (fftw_real) j*hn, 0.0);
        glVertex3f(wn*sim->DIM, hn + (fftw_real) j*hn, 0.0);
    }

    //Front
    glVertex3f(wn , hn, 0);
    glVertex3f(wn, hn, hp_height);

    glVertex3f(wn + (sim->DIM -1)*wn, hn, 0);
    glVertex3f(wn + (sim->DIM -1)*wn, hn, hp_height);

    //Left
    glVertex3f(wn, hn + (sim->DIM -1) * hn, 0);
    glVertex3f(wn, hn + (sim->DIM -1) * hn, hp_height);

    //Right
    glVertex3f(wn + (sim->DIM - 1)*wn, hn + (sim->DIM -1) * hn, 0);
    glVertex3f(wn + (sim->DIM - 1)*wn, hn + (sim->DIM -1) * hn, hp_height);

    //Top
    glVertex3f(wn, hn, hp_height);
    glVertex3f(wn + (sim->DIM -1) * wn, hn, hp_height);

    glVertex3f(wn + (sim->DIM -1) * wn, hn, hp_height);
    glVertex3f(wn + (sim->DIM -1) * wn, hn + (sim->DIM -1) * hn, hp_height);

    glVertex3f(wn + (sim->DIM -1) * wn, hn + (sim->DIM -1) * hn, hp_height);
    glVertex3f(wn, hn + (sim->DIM -1) * hn, hp_height);

    glVertex3f(wn, hn + (sim->DIM -1) * hn, hp_height);
    glVertex3f(wn, hn, hp_height);

    //Draw marks
    int nsteps = 4;
    int mark_size = 2;

    for (int i = 1; i < nsteps; ++i) //start at 1 because we don't want to draw at height 0
    {
        int height = (hp_height / nsteps)*i;
        //Front
        glVertex3d( wn, hn, height);
        glVertex3d( wn + wn * mark_size, hn, height);

        glVertex3d( wn + (sim->DIM - 1)*wn, hn, height);
        glVertex3d( wn + (sim->DIM - 1)*wn - wn * mark_size, hn, height);

        //Left
        glVertex3d( wn, hn, height);
        glVertex3d( wn, hn + hn * mark_size, height);

        glVertex3d( wn, hn + (sim->DIM - 1)*hn - hn * mark_size, height);
        glVertex3d( wn, hn + (sim->DIM - 1)*hn, height);

        //Right
        glVertex3d( wn + (sim->DIM - 1)*wn, hn, height);
        glVertex3d( wn + (sim->DIM - 1)*wn, hn + hn * mark_size, height);

        glVertex3d( wn + (sim->DIM - 1)*wn, hn + (sim->DIM - 1)*hn, height);
        glVertex3d( wn + (sim->DIM - 1)*wn, hn + (sim->DIM - 1)*hn - hn * mark_size, height);

        //Back
        glVertex3d( wn, hn + (sim->DIM - 1)*hn, height);
        glVertex3d( wn + wn * mark_size, hn + (sim->DIM - 1)*hn, height);

        glVertex3d( wn + (sim->DIM - 1)*wn, hn + (sim->DIM - 1)*hn, height);
        glVertex3d( wn + (sim->DIM - 1)*wn - wn * mark_size, hn + (sim->DIM - 1)*hn, height);
    }

    glEnd();
}

void Visualization::draw_smoke_surface(fftw_real *dataset, fftw_real min_v, fftw_real max_v)
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//    glDisable(GL_LIGHTING);
//    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glShadeModel(GL_SMOOTH);

    glEnable(GL_TEXTURE_1D);
    glBindTexture(GL_TEXTURE_1D,textureID[scalar_col]);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_DECAL);

    //Prepare dataset for height plot if enabled
    size_t dim = sim->DIM * 2*(sim->DIM/2+1);
    fftw_real* height_dataset = (fftw_real*) calloc(dim, sizeof(fftw_real));

    if (height_plot)
    {
        int old_display_dataset = display_dataset;
        int old_apply_mode = apply_mode;

        display_dataset = hp_display_dataset;
        apply_mode = APPLY_CLAMP;
        fftw_real dont_care;
        prepare_dataset(height_dataset, &dont_care, &dont_care); //scale, clamp or compute magnitude for the required dataset

        display_dataset = old_display_dataset;
        apply_mode = old_apply_mode;
    }

    int idx, idx0, idx1, idx2, idx3;
    double px0, py0, pz0, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3;
    glBegin(GL_TRIANGLES);
//    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE); //Enable color to modify diffuse material
    for (int j = 0; j < sim->DIM - 1; j++)
    {
        for (int i = 0; i < sim->DIM - 1; i++)
        {

            px0 = wn + (fftw_real)i * wn;
            py0 = hn + (fftw_real)j * hn;
            idx0 = (j * sim->DIM) + i;
            if (height_plot) pz0 = height_dataset[idx0]*hp_height;

            px1 = wn + (fftw_real)i * wn;
            py1 = hn + (fftw_real)(j + 1) * hn;
            idx1 = ((j + 1) * sim->DIM) + i;
            if (height_plot) pz1 = height_dataset[idx1]*hp_height;

            px2 = wn + (fftw_real)(i + 1) * wn;
            py2 = hn + (fftw_real)(j + 1) * hn;
            idx2 = ((j + 1) * sim->DIM) + (i + 1);
            if (height_plot) pz2 = height_dataset[idx2]*hp_height;

            px3 = wn + (fftw_real)(i + 1) * wn;
            py3 = hn + (fftw_real)j * hn;
            idx3 = (j * sim->DIM) + (i + 1);
            if (height_plot) pz3 = height_dataset[idx3]*hp_height;

            fftw_real v0, v1, v2, v3;

            v0 = dataset[idx0];
            v1 = dataset[idx1];
            v2 = dataset[idx2];
            v3 = dataset[idx3];

            glTexCoord1f(v0);    glVertex3f(px0,py0, pz0);
            glTexCoord1f(v1);    glVertex3f(px1,py1, pz1);
            glTexCoord1f(v2);    glVertex3f(px2,py2, pz2);

            glTexCoord1f(v0);    glVertex3f(px0,py0, pz0);
            glTexCoord1f(v3);    glVertex3f(px3,py3, pz3);
            glTexCoord1f(v2);    glVertex3f(px2,py2, pz2);

        }
    }
    glEnd();
    glDisable(GL_TEXTURE_1D);
//    glEnable(GL_COLOR_MATERIAL);
//    draw_legend(min_v, max_v);
}

//visualize: This is the main visualization function
void Visualization::visualize(void)
{
	int        i, j, idx;

    fftw_real min_v, max_v;
    size_t dim = sim->DIM * 2*(sim->DIM/2+1);
    fftw_real* dataset = (fftw_real*) calloc(dim, sizeof(fftw_real));
    prepare_dataset(dataset, &min_v, &max_v); //scale, clamp or compute magnitude for the required dataset

    if (height_plot)
        draw_3d_grid();

	if (draw_smoke)
        draw_smoke_surface(dataset, min_v, max_v);

	if (draw_vecs)
	{
		if (typeGlyph == 0 ) {
			glBegin(GL_LINES);                //draw velocities

			for (i = 0; i < sim->DIM; i++) {
				for (j = 0; j < sim->DIM; j++) {
					idx = (j * sim->DIM) + i;
					direction_to_color(sim->vx[idx], sim->vy[idx], color_dir);
					//mapping the scalar value of the dataset with color.
					if (glyph ) {//User selects glyphs options
						set_colormap(dataset[idx]);
						if (vGlyph == 0)//fluid velocity
						{
							//(3.1415927 / 180.0) * angle;
							double magV = sqrt(pow(sim->vx[idx], 2) + pow(sim->vy[idx], 2));
							double angleV = atan2(sim->vx[idx], sim->vy[idx]);
							glVertex2f(wn + (fftw_real) i * wn, hn + (fftw_real) j * hn);
							glVertex2f((wn + (fftw_real) i * wn) + vec_scale * cos(angleV) * magV,
									   (hn + (fftw_real) j * hn) + vec_scale * sin(angleV) * magV);
						}

						if (vGlyph == 1)//force
						{
							double magF = sqrt(pow(sim->fx[idx], 2) + pow(sim->fy[idx], 2));
							double angleF = atan2(sim->fx[idx], sim->fy[idx]);
							glVertex2f(wn + (fftw_real) i * wn, hn + (fftw_real) j * hn);
							glVertex2f((wn + (fftw_real) i * wn) + 500 * cos(angleF) * magF,
									   (hn + (fftw_real) j * hn) + 500 * sin(angleF) * magF);
						}
					} else {
						glVertex2f(wn + (fftw_real) i * wn, hn + (fftw_real) j * hn);
						glVertex2f((wn + (fftw_real) i * wn) + vec_scale * sim->vx[idx],
								   (hn + (fftw_real) j * hn) + vec_scale * sim->vy[idx]);
					}
				}
			}
			glEnd();
        }
        else if (typeGlyph == 1) //Conical glyph section
        {
            for (j = 0; j < sim->DIM; j++) {
                for (i = 0; i < sim->DIM ; i++) {
                    idx = (j * sim->DIM) + i;
                    set_colormap(dataset[idx]); //applying colourmapping
                    if (vGlyph == 0)//fluid velocity
                    {
                        double magV = sqrt(pow(sim->vx[idx], 2) + pow(sim->vy[idx], 2));
                        double angleV = atan2(sim->vx[idx], sim->vy[idx]);
                        double deg = angleV *(180/3.1415927);
                        glTranslatef(i*wn, j*hn, -5.0);
                        //todo:rotation based on the direction
                        glRotatef(90,  0.0, 1.0, 0.0);
                        glRotatef(-deg, 1.0, 0.0, 0.0);
                        //glutSolidCone( GLdouble base, GLdouble height, GLint slices, GLint stacks );
                        glutSolidCone(10.0, magV * 200, 20, 20);
                        glLoadIdentity();
                    } else if (vGlyph == 1) // force
                    {
                        double magF = sqrt(pow(sim->fx[idx], 2) + pow(sim->fy[idx], 2));
                        double angleF = atan2(sim->fx[idx], sim->fy[idx]);
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

void Visualization::create_textures()					//Create one 1D texture for each of the available colormaps.
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

void Visualization::do_one_simulation_step(void)
{
    sim->do_one_simulation_step();
    glutPostRedisplay();
}
