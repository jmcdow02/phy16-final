# N-body System Interaction Simulation
#
# By: James McDowell
#     Michael Rosen
#
# Due Date: May 12, 2017
#
# This program implements an n-body simulation of two systems interacting via
# the gravitational force. There are two integration methods we have
# incorporated: Euler's Method and Runge-Kutta 4th order.
#
# Two classes have been defined:
#     class Body represents one body in the system (a star if we are simulating
#     galaxies). This body contains information about its host system's position
#     and velocity, its own position, velocity, and changing acceleration,
#     its mass, and name.
#
#     class Galaxy represents the system in which the bodies reside. It contains
#     information about its position, velocity, total mass, and the list of bodies
#     it holds and may interact with.
#
# We have implemented two forms of output: a 3D line graph of the history of
# locations for each body, and a 2D simulation over time. These outputs can
# be saved to external files.
#
# This problem is O(N^2) in nature, where N is the total number of bodies in
# the simulation. We have hard-coded initial conditions for two systems, each
# with bodies representing the Sun and planets within our solar system.
#
# ALL VALUES ARE IN SI UNITS (meters, seconds, kilograms, m/s)


########################### LIBRARIES TO IMPORT ################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

################################## CONSTANTS ###################################

G    = 6.674e-11     # m^3/kg/s^2  gravitational constant
Msun = 1.989e30      # kg          mass of sun
Rsun = 2.469e20      # m           distance to center of milky way
Vsun = 220000.0      # m/s         radial velocity of sun
M_BH = 4.3e11 * Msun # Msun        mass of galactic center
pluto_x = 3.7e12     # m           pluto's orbit radius: for plotting dimension

########################### CLASS DEFINITIONS ##################################

# defines a single body
class Body:
    def __init__(self, gal_x, gal_y, gal_vel, pos_x,   pos_y,   pos_z, mass,
                                              vel_x,   vel_y,   vel_z, name="star",
                                              acl_x=0, acl_y=0, acl_z=0):
        self.gal_center = gal_x
        self.pos_x = pos_x + gal_x
        self.pos_y = pos_y + gal_y
        self.pos_z = pos_z
        self.mass  = mass
        self.vel_x = vel_x + gal_vel
        self.vel_y = vel_y
        self.vel_z = vel_z
        self.name  = name
        self.acl_x = acl_x
        self.acl_y = acl_y
        self.acl_z = acl_z


    # updates the body's velocity based on its acceleration
    # takes:   self star, timestep
    # returns: nothing, updates the star's velocity
    def update_velocity(self, timestep):
        self.vel_x += self.acl_x * timestep
        self.vel_y += self.acl_y * timestep
        self.vel_z += self.acl_z * timestep


    # updates the body's position based on its velocity
    # takes:   self star, timestep
    # returns: nothing, updates star's position
    def update_position(self, timestep):
        self.pos_x += self.vel_x * timestep
        self.pos_y += self.vel_y * timestep
        self.pos_z += self.vel_z * timestep


# defines a system in which the bodies are contained
class Galaxy:
    def __init__(self, stars, outer_stars, pos_x, pos_y, pos_z, mass,
                              vel_x, vel_y, vel_z, name="MW"):
        self.stars = stars   # bodies within the system
        self.outer_stars = outer_stars # bodies not contained in this system
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0
        self.mass = sum(s.mass for s in stars)
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0
        self.name = name


    # updates the positions of all stars in the galaxy for one timestep
    # takes:   self galaxy, timestep
    # returns: nothing, updates each star's attributes
    def compute_rotation_step(self, timestep):
        for star_ind in range(len(self.stars)):
            # can replace with call to calc_body_acceleration for Euler's Method
            star = self.rk4_body_acceleration(star_ind, timestep)
            star.update_velocity(timestep=timestep)
            star.update_position(timestep=timestep)


    # computes the acceleration on a single body from all other bodies in sim
    # takes:   galaxy, index of target body in its galaxy's star list
    # returns: target body with updated acceleration
    def calc_body_acceleration(self, body_ind):
        # body is the current star we are calculating acceleration for
        body = self.stars[body_ind]
        body.acl_x, body.acl_y, body.acl_z = 0, 0, 0

        # creates list of stars other than current star in system
        other_stars = [s for i,s in enumerate(self.stars) if i != body_ind]
        # loops through rest of stars in simulation
        for star in (other_stars + self.outer_stars):
            dist = euc_dist(body,star)
            frac = (G * star.mass) / dist**3
            body.acl_x += frac * (star.pos_x - body.pos_x)
            body.acl_y += frac * (star.pos_y - body.pos_y)
            body.acl_z += frac * (star.pos_z - body.pos_z)
        return body


    # computes the acceleration on a single body from all other bodies in sim
    # using the Runge-Kutta 4th order method
    # takes:   galaxy, index of target body in its galaxy's star list
    # returns: target body with updated acceleration
    def rk4_body_acceleration(self, body_ind, timestep):
        # body is the current star we are calculating acceleration for
        body = self.stars[body_ind]
        k1_x, k1_y, k1_z = 0, 0, 0
        k2_x, k2_y, k2_z = 0, 0, 0
        k3_x, k3_y, k3_z = 0, 0, 0
        k4_x, k4_y, k4_z = 0, 0, 0
        body.acl_x, body.acl_y, body.acl_z = 0, 0, 0

        # creates list of stars other than current star in system
        other_stars = [s for i,s in enumerate(self.stars) if i != body_ind]
        # loops through rest of stars in simulation
        for star in (other_stars + self.outer_stars):
            dist = euc_dist(body,star)
            frac = (G * star.mass) / dist**3

            # disinclude stars with weaker gravitational pull than Sun-pluto
            # attraction (to skip some integrations for faster computation time)
            if frac < 6.3e-27:
                body.acl_x += 0
                body.acl_y += 0
                body.acl_z += 0
            else:
                k1_x = frac * (star.pos_x - body.pos_x)
                k1_y = frac * (star.pos_y - body.pos_y)
                k1_z = frac * (star.pos_z - body.pos_z)
                # calculate hypothetical positions based on k1 acceleration
                # 0.5 timestep in the future
                temp_x,temp_y,temp_z = integrator(body, k1_x, k1_y, k1_z,
                                                  0.5*timestep)

                k2_x = frac * (star.pos_x - temp_x)
                k2_y = frac * (star.pos_y - temp_y)
                k2_z = frac * (star.pos_z - temp_z)
                # calculates hypothetical positions based on k2 acceleration
                # 0.5 timestep in the future
                temp_x,temp_y,temp_z = integrator(body, k2_x, k2_y, k2_z,
                                                  0.5*timestep)

                k3_x = frac * (star.pos_x - temp_x)
                k3_y = frac * (star.pos_y - temp_y)
                k3_z = frac * (star.pos_z - temp_z)
                # calculates hypothetical positions based on k3 acceleration
                # a full timestep in the future
                temp_x,temp_y,temp_z = integrator(body, k3_x, k3_y, k3_z,
                                                  timestep)

                k4_x = frac * (star.pos_x - temp_x)
                k4_y = frac * (star.pos_y - temp_y)
                k4_z = frac * (star.pos_z - temp_z)

                # uses all 4 k values to calculate and update the actual
                # acceleration of the body
                body.acl_x += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
                body.acl_y += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
                body.acl_z += (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6
        return body


    # runs a simulation of the galaxy
    # takes:   galaxy, timestep, number of timesteps to execute
    # returns: list of dictionaries for each star, containing lists of
    #          coordinates for each timestep
    def simulate_galaxy(self, timestep, num_steps, store_freq):
        # stored locations of stars in simulation
        locations = []
        # give each star the outline for a list of locations to store
        for star in self.stars:
            locations.append({"x":[],"y":[],"z":[],"name":star.name})

        # computes a timestep of galaxy's rotation num_steps times
        for step in range(1, num_steps):
            self.compute_rotation_step(timestep=timestep)
            if step % store_freq == 0:
                for i,location in enumerate(locations):
                    location["x"].append(self.stars[i].pos_x)
                    location["y"].append(self.stars[i].pos_y)
                    location["z"].append(self.stars[i].pos_z)

        return locations


################### GENERAL FUNCTION DEFINITIONS ############################

# computes hypothetical position given acceleration
# for use by RK4 method above
# takes:   a Body, acceleration in x,y, and z directions, and the timestep
# returns: x,y,z positions given acceleration and initial positions and
#          veloicites of the body (doesn't update the Body object)
def integrator(body, accel_x, accel_y, accel_z, timestep):
    # copy values to temporary variables
    # (we don't want to change the actual values for the body, only use them)
    vx, vy, vz = body.vel_x, body.vel_y, body.vel_z
    px, py, pz = body.pos_x, body.pos_y, body.pos_z
    # calculates velocity based on acceleration and timestep
    vx += accel_x * timestep
    vy += accel_y * timestep
    vz += accel_z * timestep
    # calculates position based on velocity and timestep
    px += vx * timestep
    py += vy * timestep
    pz += vz * timestep

    return px,py,pz


# computes the euclidean distance between two bodies
# takes:   the target body, another body
# returns: euclidean distance
def euc_dist(curr, other):
    dist = (curr.pos_x - other.pos_x)**2 + (curr.pos_y - other.pos_y)**2 + \
           (curr.pos_z - other.pos_z)**2
    return np.sqrt(dist)


# plots the simulation of the galaxy
# takes:   list of location coordinates for each star, file to output plot
# returns: nothing, plots simulation
def show_galaxy(locations, file_out=None):
    colors = ['r','g','b','m','y','c']
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    max_dim = 0
    for i,loc in enumerate(locations):
        curr_max = max(max(loc["x"]),max(loc["y"]),max(loc["z"]))
        if max_dim < curr_max:
            max_dim = curr_max
        ax.plot(loc["x"], loc["y"], loc["z"], c = colors[i%6],label=loc["name"])

    ax.set_xlim([-pluto_x, pluto_x*3])
    ax.set_ylim([-pluto_x, pluto_x*3])
    ax.set_zlim([-pluto_x, pluto_x*3])
    ax.legend()

    if file_out:
        plt.savefig(file_out)
    else:
        plt.show()
    return max_dim


# creates a 2D animation
# takes:   dimension value, list of dictionaries containing data for every star,
#          and a file where the animation can be saved
# returns: nothing, it first shows the animation, then saves it to a file
def animate_galaxy(max_dim, locations, file_out=None):
    colors = ['r','g','b','m','y','c']

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-pluto_x, pluto_x*3),
                                                  ylim=(-pluto_x, pluto_x))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=0)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    # initializes objects for animation
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    # function for one iteration of animation
    def animate(i):
        thisx = []
        thisy = []
        # for every star in simulation, create x and y positions arrays
        for star in range(len(locations)):
            thisx.append(locations[star]["x"][i])
            thisy.append(locations[star]["y"][i])

        line.set_data(thisx, thisy)
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(locations[0]["y"])),
                                  interval=100, blit=True, init_func=init)

    plt.show()
    ani.save(file_out, fps=15)


######################### PROGRAM STARTS RUNNING HERE ##########################
if __name__ == '__main__':

    # constants to pass to simulation functions
    time_step = 1000      # defines the resolution of each integration (in seconds)
    num_steps = 500000  # defines the number of integrations we perform
    store_freq = 500     # defines the frequency at which we store a body's location
                         # for plotting

    # initial conditions for the systems:
    gal1_x = 0       # initial x position for first system
    gal1_y = 0       # initial y position for first system
    gal2_x = 3.0e12  # initial x position for second system
    gal2_y = 1e11    # initial y position for second system
    gal_vel = 30000  # initial velocity for each system

    # template for initializing the Body class:
    # Body(gal_x,gal_y,gal_vel,posx,posy,posz,mass,velx,vely,velz,name,aclx=0,acly=0,aclz=0)

    # initial hard-coded values for each Body in first system:
    sun     = Body(gal1_x,gal1_y,gal_vel, 0,0,0,       2e30,     0,0,0,        "sun")
    mercury = Body(gal1_x,gal1_y,gal_vel, 0,5.7e10,0,  3.285e23, 47000,0,0,   "merc")
    venus   = Body(gal1_x,gal1_y,gal_vel, 0,1.1e11,0,  4.8e24,   35000,0,0,    "ven")
    earth   = Body(gal1_x,gal1_y,gal_vel, 0,1.5e11,0,  6e24,     30000,0,0,  "earth")
    mars    = Body(gal1_x,gal1_y,gal_vel, 0,2.2e11,0,  2.4e24,   24000,0,0,   "mars")
    jupiter = Body(gal1_x,gal1_y,gal_vel, 0,-7.7e11,0, 1e28,    -13000,0,0,"jupiter")
    saturn  = Body(gal1_x,gal1_y,gal_vel, 0,1.4e12,0,  5.7e26,   9000,0,0,  "saturn")
    uranus  = Body(gal1_x,gal1_y,gal_vel,-2.8e12,0,0,  8.7e25,   0,6835,0,  "uranus")
    neptune = Body(gal1_x,gal1_y,gal_vel, 0,4.5e12,0,  1e26,     5477,0,0,  "netune")
    pluto   = Body(gal1_x,gal1_y,gal_vel, 0,3.7e12,0,  1e22,     4748,0,0,   "pluto")
    stars1  = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus]

    # initial hard-coded values for each Body in second system:
    sun     = Body(gal2_x,gal2_y,-gal_vel, 0,0,0,       2e30,     0,0,0,       "sun")
    mercury = Body(gal2_x,gal2_y,-gal_vel, 0, 5.7e10,0, 3.285e23, 47000,0,0,  "merc")
    venus   = Body(gal2_x,gal2_y,-gal_vel, 0, 1.1e11,0, 4.8e24,   35000,0,0,   "ven")
    earth   = Body(gal2_x,gal2_y,-gal_vel, 0,1.5e11,0,  6e24,     30000,0,0, "earth")
    mars    = Body(gal2_x,gal2_y,-gal_vel, 0,2.2e11,0,  2.4e24,   24000,0,0,  "mars")
    jupiter = Body(gal2_x,gal2_y,-gal_vel, 7.7e11,0,0,  1e28,0,  -13000,0, "jupiter")
    saturn  = Body(gal2_x,gal2_y,-gal_vel, 0,-1.4e12,0, 5.7e26,  -9000,0,0, "saturn")
    uranus  = Body(gal2_x,gal2_y,-gal_vel, 0,2.8e12,0,  8.7e25,   6835,0,0, "uranus")
    neptune = Body(gal2_x,gal2_y,-gal_vel, 0,4.5e12,0,  1e26,     5477,0,0, "netune")
    pluto   = Body(gal2_x,gal2_y,-gal_vel, 0,3.7e12,0,  1e22,     4748,0,0,  "pluto")
    stars2 = [sun, mercury, venus, earth, mars, jupiter,saturn, uranus]

    # template for initializing the Galaxy class:
    #      Galaxy(stars,outer_stars, posx,posy,posz,  mass,  velx,vely,velz,    name)
    MW   = Galaxy(stars1, stars2,    gal1_x,gal1_y,0, 2e30,   gal_vel,0,0,     "MW1")
    MW2  = Galaxy(stars2, stars1,    gal2_x,gal2_y,0, 2e30,  -gal_vel,0,0,     "MW2")

    # simulate the first system, get a list of past positions for each body
    locations = MW.simulate_galaxy(time_step, num_steps, store_freq)
    # simulate the second system, get a list of past positions for each body
    locations2 = MW2.simulate_galaxy(time_step, num_steps, store_freq)

    # concatenates the lists of positions
    locs = locations + locations2
    # makes 3D line graph of simulation
    max_dim = show_galaxy(locs)
    # animates the simulation
    animate_galaxy(max_dim, locs, "sim7.mp4")



