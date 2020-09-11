# reflFMM
using Fast Marching Method to calculate reflection travel times

## needed s/w
pykonal is an implementation of FMM
[PyKonal](https://github.com/malcolmw/pykonal)

scipy - differential_evolution is an implementation of a genetic algorithm 
![differential evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html)

## plotting
If you want to display plots, I use [bokeh](http://bokeh.org) and matplotlib

## Usage
1. Create vel  model: create an ice model (infinite layer of ice  with  vi=3.8).  The coord system is x=-2.56 to  +2.56km, with  10m grid and z=0  to 1.024  with 1m  grid. 
```python
icemod = ice_mod()
```
2. Create reflector: you can create an ice bottom layer within  icemod with `make_vel`
```python
v, icelayer, waterlay = make_vel(icemod, zi, alphai, zw, alphaw)
```
  - output `v`  and  `waterlay` are ignored
  - input `zi` and  `alphai`  are the layer depth and dip at `x=0` (center of model)
3. Forward model:  calculate travel times to that layer  through the ice
```
modx, modt = calc_refl(icemod, shot_x, shot_z,  icelay)
```
  - `modx` and `modt` are the offsets to geophones and traveltimes. there are `nx` outputs that match the input model x resolution (10m), so `modx` goes from  -2.56 to 2.56km at 10m

4. Compare modeled and observed
 - `tt_err` calculates the mean-square err between model and observed data
```
err = tt_err([zi, alphai], icemod, shot_x, shot_z, obsx, obst, disp=True)
```
 - `obsx` and `obst` are observed offsets (IN METERS - WARNING - FIXME) and times (in seconds)
 - if `disp=True`,  print out each geophone observed/modeled  and diff
 - return mean square error

5. Carry out inversion. I used `differential_evolution`
```
differential_evolution(tt_err, bounds=[(zimin, zimax),  (alphaimin, alphaimax)], mutation=(0.5,1.5), args=(icemod, shotx, shotz, obsx, obst))
```
 - this runs through a range of `zi` and `alphai` (between the bounds) and repeatedly runs `tt_err` until a min error is reached. 

 6.  Once the ice bottom is determined, make a water model: fixed ice `zi`, `alphai` and infinite water below
 ```
 watermod = water_mod(icemod, zi, alphai)
 differential_evolution(tt_err_waterbot, bounds=[(0.61, 0.8),  (-8, 8)], args=(0.61, 0.4937, watermod, 0, 0, waterobs[:,0],  waterobs[:,1]))
 ```
  -  `watermod` is derived  from `icemod` but  with an  ice bottom added and  infinite water  below
  -  `tt_err_waterbot` is used by `differential_evolution` to get a best water bottom  fit to the observations


## References
1. Sethian, J. A. (1996). A fast marching level set method for monotonically advancing fronts. *Proceedings of the National Academy of Sciences, 93*(4), 1591–1595. https://doi.org/10.1073/pnas.93.4.1591
2. White, M. C. A., Fang, H., Nakata, N., & Ben-Zion, Y. (2020). PyKonal: A Python Package for Solving the Eikonal Equation in Spherical and Cartesian Coordinates Using the Fast Marching Method. *Seismological Research Letters, 91*(4), 2378-2389. https://doi.org/10.1785/0220190318
3. Storn, R and Price, K, Differential Evolution - a Simple and Efficient Heuristic for Global Optimization over Continuous Spaces, Journal of Global Optimization, 1997, 11, 341 - 359.