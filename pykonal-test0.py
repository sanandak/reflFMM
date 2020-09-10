#!/usr/bin/env python
# coding: utf-8

# # Figure 8
# Compatible with PyKonal Version 0.2.0

# In[3]:


get_ipython().run_line_magic('matplotlib', 'ipympl')

import matplotlib.pyplot as plt
import numpy as np
import os
import pkg_resources
import pykonal
import scipy.ndimage


# In[4]:


from scipy.optimize import differential_evolution


# In[5]:


import bokeh.plotting as bp
from bokeh.io import output_notebook, curdoc, output_file
output_notebook()


# In[6]:


import json
import pandas as pd
#from pandas.io.json import json_normalize


# In[7]:


with open("/Users/sak/Desktop/MELT/GZDS/12-JAN-2019/42_water.json") as f:
    j = json.load(f)
    df=pd.json_normalize(j['picks'], sep='_')
    df_water = df.loc[:,['offset', 'autopicks_picks10_peakTime', 'autopicks_picks_peakTime']]
    df_water = df_water.rename({'offset': 'x', 'autopicks_picks10_peakTime':'t10', 'autopicks_picks_peakTime': 't'}, axis='columns')
with open("/Users/sak/Desktop/MELT/GZDS/12-JAN-2019/42_ice.json") as f:
    j = json.load(f)
    df_ice=pd.json_normalize(j['picks'], sep='_')
    df_ice = df_ice.loc[:,['offset', 'autopicks_picks10_peakTime', 'autopicks_picks_peakTime']]
    df_ice = df_ice.rename({'offset': 'x', 'autopicks_picks10_peakTime':'t10', 'autopicks_picks_peakTime': 't'}, axis='columns')


# In[206]:


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


# In[221]:


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


# In[203]:


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
    
    return solver_ug.tt.values[:,0,0]


# In[257]:


def tt_err(refl, velocity, shotloc_x, shotloc_z, obs_x, obs_t,  disp=False):
    "given a reflector (depth refl[0] at x=0, dip refl[1]), calculate tt  through velocity model  from  shotloc and  compare to obs"
    refl_z=refl[0]
    refl_ang=refl[1]
    layer_idxs = make_nodes(velocity, refl_z, refl_ang)
    tt_mod = calc_refl(velocity, shotloc_x, shotloc_z, layer_idxs)
    err = 0
    for x,t in zip(obs_x, obs_t):
        (ix, iz, iy) = get_idx(velocity, (x/1000, 0, 0))
        #print(ix, x)
        modt = tt_mod[ix]
        diff = t-modt
        if disp:
            print(x, t, ix, modt, diff)
        err = err + diff**2
    print("z=", refl_z, "ang=", refl_ang, "err=", err)
    return err
def tt_err_waterbot(refl, zi, alphai, velocity, shotloc_x, shotloc_z, obs_x, obs_t,  disp=False):
    "given a water bott refl (depth refl[0] at x=0, dip refl[1]), calculate tt through velocity model from  shotloc and  compare to obs"
    refl_z=refl[0]
    refl_ang=refl[1]
    v, icelay, waterlay = make_vel(velocity, zi, alphai, refl_z, refl_ang) #make_nodes(velocity, refl_z, refl_ang)
    tt_mod = calc_refl(v, shotloc_x, shotloc_z, waterlay)
    err = 0
    for x,t in zip(obs_x, obs_t):
        (ix, iz, iy) = get_idx(v, (x/1000, 0, 0))
        #print(ix, x)
        modt = tt_mod[ix]
        diff = t-modt
        if disp:
            print(x, t, ix, modt, diff)
        err = err + diff**2
    print("z=", refl_z, "ang=", refl_ang, "err=", err)
    return err


# In[52]:


def ice_mod():
    velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
    velocity.min_coords = -2.56, 0, 0
    velocity.node_intervals = 0.01, 0.001, 0.01
    velocity.npts = 512, 1024, 1
    #velocity.values = 3.8*np.ones(velocity.npts)
    #velocity.values[0:511,60:70] = 1.5
    #velocity.values[0:511,71:127] = 4.5
    npts = velocity.npts
    nx=npts[0]
    nz=npts[1]
    ny=npts[2]
    
    velocity.values[0:nx, 0:nz] = 3.8
    #icelay = make_nodes(vel, zi, alphai)
    #waterlay = make_nodes(vel, zw, alphaw)

    #for ix in range(nx):
        #idx = int(zi/dz) + int(ix*dx*np.tan(alphai)/dz)
        #print(ix, idx)
    #    ili = icelay[ix]
    #    wli = waterlay[ix]
    #    vel.values[ix, ili:wli] = 1.5
    #    vel.values[ix, wli:nz] = 4.5
    #vel.values[0:nx-1, 60:70] = 1.5
    #vel.values[0:nx-1,  71:nz-1] = 4.5
    return velocity

def water_mod(zi, alphai):
    velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
    velocity.min_coords = -2.56, 0, 0
    velocity.node_intervals = 0.01, 0.001, 0.01
    velocity.npts = 512, 1024, 1
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
    
    #velocity = make_vel(velocity, 0.610, 0*np.pi/180,  1, 0*np.pi/180)
    #icelay = make_nodes(velocity, 0.61, 0*np.pi/180)
    #waterlay = make_nodes(velocity, 0.66, -4.2*np.pi/180)
#scipy.ndimage.gaussian_filter(20. * np.random.randn(*velocity.npts) + 6., 10)


# In[258]:


icemod = ice_mod()
#icelay = make_nodes(icemod, 0.61, 0.55)
watermod = water_mod(0.61, 0.55)
vel, icelay, waterlay = make_vel(icemod, 0.61, 0.55, 0.66, -3)
#print(watermod.values[256,600:700])


# In[228]:


print(waterlay)


# In[259]:


#waterlay = make_nodes(watermod, 0.655, -7)
xx=calc_refl(vel,  0, 0, waterlay)
for i in range(len(xx)):
    print(i, xx[i])


# In[260]:


tt_err_waterbot([0.66, -3], 0.61, 0.55, watermod, 0,  0, df_water.x,   df_water.t, True)


# In[137]:


differential_evolution(tt_err, bounds=[(0.4, 0.8),  (-8, 8)], mutation=(0.5,1.5), args=(icemod, 0, 0, df_ice.x, df_ice.t))


# In[261]:


differential_evolution(tt_err_waterbot, bounds=[(0.61, 0.8),  (-8, 8)], args=(0.61, 0.55, watermod, 0, 0, df_water.x, df_water.t))


# In[13]:


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


# In[262]:


velocity = pykonal.fields.ScalarField3D(coord_sys="cartesian")
velocity.min_coords = -2.56, 0, 0
velocity.node_intervals = 0.01, 0.001, 0.01
velocity.npts = 512, 1024, 1
#velocity.values = 3.8*np.ones(velocity.npts)
#velocity.values[0:511,60:70] = 1.5
#velocity.values[0:511,71:127] = 4.5
velocity,icelay,waterlay = make_vel(velocity, 0.610, 0.55,  0.661, -4.2)
#icelay = make_nodes(velocity, 0.61, 0)
#waterlay = make_nodes(velocity, 0.655, -7, water=True)
#scipy.ndimage.gaussian_filter(20. * np.random.randn(*velocity.npts) + 6., 10)
solver_dg = pykonal.EikonalSolver(coord_sys="cartesian")
solver_dg.vv.min_coords = velocity.min_coords
solver_dg.vv.node_intervals = velocity.node_intervals
solver_dg.vv.npts = velocity.npts
solver_dg.vv.values = velocity.values

shotloc_x = 2.56 # km
shotloc_z  = 0
src_idx = (int(shotloc_x/velocity.node_intervals[0]), int(shotloc_z/velocity.node_intervals[1]), 0)
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
    idx = (ix, waterlay[ix], 0)
    solver_ug.tt.values[idx] = solver_dg.tt.values[idx]
    solver_ug.unknown[idx] = False
    solver_ug.trial.push(*idx)
solver_ug.solve()


# In[263]:


plt.close("all")
fig = plt.figure(figsize=(6, 2.5))

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax = fig.add_subplot(1, 1, 1, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax.set_ylabel("Depth [km]")
ax2.set_xlabel("Horizontal offset [km]")
ax1.set_xticklabels([])

for solver, ax, panel in (
    (solver_dg, ax1, f"(a)"), 
    (solver_ug, ax2, f"(b)"),
):
    ax.text(-0.025, 1.05, panel, va="bottom", ha="right", transform=ax.transAxes)
    qmesh = ax.pcolormesh(
        solver.vv.nodes[:,:,0,0], 
        solver.vv.nodes[:,:,0,1], 
        solver.vv.values[:,:,0],
        cmap=plt.get_cmap("jet")
    )
    ax.contour(
        solver.tt.nodes[:,:,0,0], 
        solver.tt.nodes[:,:,0,1], 
        solver.tt.values[:,:,0],
        colors="k",
        linestyles="--",
        linewidths=1,
        levels=np.arange(0, solver.tt.values.max(), 0.25)
    )
    ax.scatter(
        solver.vv.nodes[src_idx + (0,)],
        solver.vv.nodes[src_idx + (1,)],
        marker="*",
        facecolor="w",
        edgecolor="k",
        s=256
    )
    ax.invert_yaxis()
cbar = fig.colorbar(qmesh, ax=(ax1, ax2))
cbar.set_label("Velocity [km/s]")


# In[264]:


p1=bp.figure()
p1.circle(df_ice.x, df_ice.t)
p1.circle(df_water.x, df_water.t)
p1.circle(np.linspace(-2560, 2560, num=512), solver_ug.tt.values[:,0,0])
#p1.circle(np.linspace(-2560, 2560, num=512), xx)

bp.show(p1)


# In[180]:


plt.close("all")
vel=watermod
fig = plt.figure(figsize=(6, 2.5))

ax1 = fig.add_subplot(1, 1, 1)
#ax2 = fig.add_subplot(2, 1, 2)

ax = fig.add_subplot(1, 1, 1, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax.set_ylabel("Depth [km]")
ax1.set_xlabel("Horizontal offset [km]")
ax1.set_xticklabels([])

ax.text(-0.025, 1.05, panel, va="bottom", ha="right", transform=ax.transAxes)
qmesh = ax.pcolormesh(
    vel.nodes[:,:,0,0], 
    vel.nodes[:,:,0,1], 
    vel.values[:,:,0],
    cmap=plt.get_cmap("jet"),
    shading="auto"
)
#ax1.scatter([-2.56, 2.56], [0.7, 0.8])

ax1.invert_yaxis()
cbar = fig.colorbar(qmesh, ax=(ax1,))
cbar.set_label("Velocity [km/s]")


# In[101]:


fig1 = plt.figure()
axx = fig1.add_subplot()
axx.scatter(np.arange(512), solver_ug.tt.values[:,0])
#axx.scatter(np.linspace(0,1,512),solver_ug.tt.values[:,0,0])


# In[149]:


np.arange(len(waterlay))


# In[73]:


solver_ug.tt.values[0]


# In[17]:


get_ipython().run_line_magic('pinfo', 'np.linspace')


# In[ ]:




