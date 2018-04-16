//
// Created by mario on 3-4-18.
//

#ifndef SCIENTIFIC_VISUALIZATION_VISUALIZATION_H
#define SCIENTIFIC_VISUALIZATION_VISUALIZATION_H

#include<rfftw.h>
#include <cmath>
#include "simulation.h"
#include <cstring>
#include <cfloat>
#include <cassert>
#include <vector>
#include <array>
#include <queue>

#ifdef LINUX
#include <GL/glut.h>            //the GLUT graphics library
#endif

#ifdef MACOS
#include <GLUT/glut.h>            //the GLUT graphics library
#endif


class Visualization
{
public:
    Simulation *sim;
    int   winWidth, winHeight;      //size of the graphics window, in pixels
    int   gridWidth, gridHeight;    //size of the simulation grid in pixels
    fftw_real  wn, hn;              //size of the grid cell in pixels
    int   color_dir = 0;            //use direction color-coding or not
    float vec_scale = 1000;			//scaling of hedgehogs
    int   draw_smoke = 1;           //draw the smoke or not
    int   draw_vecs = 0;            //draw the vector field or not
    const int COLOR_BLACKWHITE=0;   //different types of color mapping: black-and-white, rainbow, banded
    const int COLOR_RAINBOW=1;
    const int COLOR_HEATMAP=2;
    const int COLOR_CUSTOM=3;
    int   scalar_col = 1;           //method for scalar coloring
    int   frozen = 0;               //toggles on/off the animation

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
    int display_divergence = 0;
    int dynamic_scalling = 0;
    int height_plot = 0;
    int glyph = 0;
    int vGlyph = 0;
    int sGlyph = 0;
    int typeGlyph = 0;
    int hp_height = 250;
    int hp_display_dataset = 0;
    int stream_tubes = 0;
    std::vector<std::array<GLdouble, 3>> seeds; //Location of the streamtube seeds
    float min_hue=0, max_hue=0, min_sat=1, max_sat=1; //parameters for the user-defined colormap
    std::queue<std::array<fftw_real*, 2>> v_volume; //time as z 3D volume of velocity field

    GLfloat light_ambient[4] = { 1, 1, 1, 1.0 };
    GLfloat light_diffuse[4] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat light_specular[4] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat light_position[4] = { ((GLfloat)gridWidth)/2.0f, ((GLfloat)gridHeight)/2.0f, hp_height+50, 1.0 };

    Visualization(int DIM);
    void rainbow(float value,float* R,float* G,float* B);
    void heatmap(float value, float* R, float* G, float* B);
    void user_defined_map(float value, float* R, float* G, float* B);
    float conform_to_bands(float vy);
    fftw_real scale(fftw_real min, fftw_real max, fftw_real value);
    void set_colormap(float vy);
    void get_colormap(float vy, float *R, float *G, float *B);
    void draw_text(const char* text, int x, int y);
    void draw_legend(fftw_real min_v, fftw_real max_v);
    void direction_to_color(float x, float y, int method);
    void find_min_max(fftw_real* min_v, fftw_real* max_v, fftw_real* dataset);
    void compute_divergence(fftw_real *x, fftw_real *y, fftw_real *dataset);
    void prepare_dataset(fftw_real* dataset, fftw_real* min_v, fftw_real* max_v);
    void draw_smoke_surface(fftw_real *dataset, fftw_real min_v, fftw_real max_v);
    void visualize(void);
    void create_textures();
    void do_one_simulation_step();
    void draw_3d_grid();
    void add_seed(GLdouble x, GLdouble y, GLdouble z);
    void remove_seed();
    void draw_seeds();
    void move_seed(GLdouble x, GLdouble y, GLdouble z);
    void draw_tubes();
    void light();
};
#endif //SCIENTIFIC_VISUALIZATION_VISUALIZATION_H
