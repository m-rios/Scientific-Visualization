//
// Created by Mario Ríos Muñoz on 16/04/2018.
//

#include "util.h"

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
