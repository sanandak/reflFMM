#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import os
import pkg_resources
import pykonal
import scipy.ndimage

from scipy.optimize import differential_evolution

import bokeh.plotting as bp
from bokeh.io import output_notebook, curdoc, output_file
#output_notebook()

def calc_refl(velocity, shotloc_x, shotloc_z, layer_idxs):
    """calculate the reflection tt in velocity grid with given shotloc off layer_idxs
    velocity is a pykonal velocity obj
    shotloc_[xz] are positions relative to velocity.min_coords
    layer_idxs is an array of depth indices (z) for each x node that denotes the reflection surface
    """
    solver_dg = pykonal.EikonalSolver(coord_sys="cartesian")
    solver_dg.vv.min_coords = velocity.min_coords
    solver_dg.vv.node_intervals = velocity.node_intervals
    solver_dg.vv.npts = velocity.npts
    solver_dg.vv.values = velocity.values

    #shotloc = 2.56 # km
    src_idx = (int((shotloc_x - velocity.min_coords[0])/velocity.node_intervals[0]), int(shotloc_z/velocity.node_intervals[1]), 0)
    solver_dg.tt.values[src_idx] = 0
    solver_dg.unknown[src_idx] = False
    solver_dg.trial.push(*src_idx)
    solver_dg.solve()

    solver_ug = pykonal.EikonalSolver(coord_sys="cartesian")
    solver_ug.vv.min_coords = solver_dg.vv.min_coords
    solver_ug.vv.node_intervals = solver_dg.vv.node_intervals
    solver_ug.vv.npts = solver_dg.vv.npts
    solver_ug.vv.values = solver_dg.vv.values

    for ix in range(solver_ug.tt.npts[0]):
        #idx = (ix, solver_ug.tt.npts[1]-1, 0)
        idx = (ix, layer_idxs[ix], 0)
        solver_ug.tt.values[idx] = solver_dg.tt.values[idx]
        #print(idx, solver_dg.tt.values[idx])
        solver_ug.unknown[idx] = False
        solver_ug.trial.push(*idx)
    solver_ug.solve()
    ofs = np.linspace(solver_ug.tt.min_coords[0] - shotloc_x, solver_ug.tt.max_coords[0] - shotloc_x, num=solver_ug.tt.npts[0], endpoint=True)
    return ofs, solver_ug.tt.values[:,0,0]

def make_vel(vel, zi, alphai, zw, alphaw):
    "ice shelf geom with vi=3.8 0->zi  and vw=1.5  from zi->zw vsed=4.5 below"
    npts = vel.npts
    nx=npts[0]
    nz=npts[1]
    ny=npts[2]
    
    vel.values[0:nx, 0:nz] = 3.8
    icelay = make_nodes(vel, zi, alphai)
    waterlay = make_nodes(vel, zw, alphaw)
    
    for i in range(len(icelay)):
        if waterlay[i] <  icelay[i]:
            waterlay[i] =  icelay[i]

    for ix in range(nx):
        #idx = int(zi/dz) + int(ix*dx*np.tan(alphai)/dz)
        #print(ix, idx)
        ili = icelay[ix] #ice bottom index
        wli = waterlay[ix] #water bottom index
        # don't let water bottom rise above ice bottom
        if  wli>ili:
            vel.values[ix, ili:wli] = 1.5
            vel.values[ix, wli:nz] = 4.5
        else:
            vel.values[ix, ili:nz] = 4.5
    #vel.values[0:nx-1, 60:70] = 1.5
    #vel.values[0:nx-1,  71:nz-1] = 4.5
    return vel, icelay, waterlay


def tt_err(refl, velocity, shotloc_x, shotloc_z, obs_x, obs_t,  disp=False):
    "given a reflector (depth refl[0] at x=0, dip refl[1]), calculate tt  through velocity model  from  shotloc and  compare to obs"
    refl_z=refl[0]
    refl_ang=refl[1]
    layer_idxs = make_nodes(velocity, refl_z, refl_ang)
    modx, modt = calc_refl(velocity, shotloc_x, shotloc_z, layer_idxs)
    err = 0
    for x,t in zip(obs_x, obs_t):
        (ix, iz, iy) = get_idx(velocity, (x/1000, 0, 0))
        #print(ix, x)
        modti = modt[ix]
        diff = t-modti
        if disp:
            print(x, t, ix, modti, diff)
        err = err + diff**2
    print("z=", refl_z, "ang=", refl_ang, "err=", err)
    return err

def tt_err_waterbot(refl, zi, alphai, velocity, shotloc_x, shotloc_z, obs_x, obs_t,  disp=False):
    "given a water bott refl (depth refl[0] at x=0, dip refl[1]), calculate tt through velocity model from  shotloc and  compare to obs"
    refl_z=refl[0]
    refl_ang=refl[1]
    v, icelay, waterlay = make_vel(velocity, zi, alphai, refl_z, refl_ang) #make_nodes(velocity, refl_z, refl_ang)
    #print(v.min_coords)
    modx, modt = calc_refl(velocity, shotloc_x, shotloc_z, waterlay)
    err = 0
    for x,t in zip(obs_x, obs_t):
        (ix, iz, iy) = get_idx(v, (x/1000, 0, 0))
        #print(ix, x)
        modti = modt[ix]
        diff = t-modti
        if disp:
            print(x, t, ix, modti, diff)
        err = err + diff**2
    print("z=", refl_z, "ang=", refl_ang, "err=", err)
    return err

def ice_mod(vi = 3.8, x0=-2.56, dx=0.01, dz =0.001, nx=512, nz=1024):
    velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
    velocity.min_coords = x0, 0, 0
    velocity.node_intervals = dx, dz, 0.01
    velocity.npts = nx, nz, 1
    #velocity.values = 3.8*np.ones(velocity.npts)
    #velocity.values[0:511,60:70] = 1.5
    #velocity.values[0:511,71:127] = 4.5
    #npts = velocity.npts
    #nx=npts[0]
    #nz=npts[1]
    #ny=npts[2]
    
    velocity.values[0:nx, 0:nz] = vi

    return velocity

def water_mod(icemod, zi, alphai):
    velocity = icemod
    #pykonal.fields.ScalarField3D(coord_sys="cartesian")
    #elocity.min_coords = -2.56, 0, 0
    #velocity.node_intervals = 0.01, 0.001, 0.01
    #velocity.npts = 512, 1024, 1
    #velocity.values = 3.8*np.ones(velocity.npts)
    #velocity.values[0:511,60:70] = 1.5
    #velocity.values[0:511,71:127] = 4.5
    npts = velocity.npts
    nx=npts[0]
    nz=npts[1]
    ny=npts[2]
    
    velocity.values[0:nx, 0:nz] = 3.8
    icelay = make_nodes(velocity, zi, alphai)
    #waterlay = make_nodes(vel, zw, alphaw)

    for ix in range(nx):
        #idx = int(zi/dz) + int(ix*dx*np.tan(alphai)/dz)
        #print(ix, idx)
        ili = icelay[ix]
    #    wli = waterlay[ix]
        velocity.values[ix, ili:nz] = 1.5
    #    vel.values[ix, wli:nz] = 4.5
    #vel.values[0:nx-1, 60:70] = 1.5
    #vel.values[0:nx-1,  71:nz-1] = 4.5
    return velocity

def get_idx(velocity, coord):
    """given velocity, calculate idices for coord(3-tuple)"""
    d = velocity.node_intervals
    dx=d[0]
    dz=d[1]
    dy=d[2]
    mn = velocity.min_coords
    mnx=mn[0]
    mnz=mn[1]
    mny=mn[2]
    ix  = int((coord[0] - mnx)/dx)
    iz  = int((coord[1] - mnz)/dz)
    iy  = int((coord[2] - mny)/dy)
    return (ix, iz, iy)

def make_nodes(vel, z, alpha):
    """"for geom in vel, find nodes for a layer that has depth z at x=0, alpha in deg."""
    npts = vel.npts
    nx=npts[0]
    nz=npts[1]
    ny=npts[2]
    d = vel.node_intervals
    dx=d[0]
    dz=d[1]
    dy=d[2]
    mn = vel.min_coords
    mnx=mn[0]
    mnz=mn[1]
    mny=mn[2]
    #print(nx, nz, ny, dx, dz, dy, mnx, mnz, mny)
    lay = np.ndarray(nx, dtype=int)
    for ix in range(nx):
        idx = int(z/dz) + int((ix*dx+mnx)*np.tan(alpha * np.pi/180)/dz)
        if idx >= nz:
            idx = nz-1
        if idx < 0:
            idx = 0
        lay[ix] = int(idx)
        #print(ix, idx)
        #vel.values[ix, idx:nz] = 1.5
    return lay
