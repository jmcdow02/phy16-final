# simulation of galaxy interaction
#
# By: James McDowell
#     Michael Rosen
#
# Due Date: May 12, 2017
#
#

########################### LIBRARIES TO IMPORT ################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##################### CONSTANTS/INITIAL CONDITIONS #############################

G    = 6.674e-11    # m^3/kg/s^2   gravitational constant
Msun = 1.989e30     # kg   mass of sun
Rsun = 2.469e20     # m    distance to center of milky way
Vsun = 220000.0     # m/s  radial velocity of sun
M_BH = 4.3e11 * Msun # kg   mass of galactic center

black_hole = {"pos_x":0, "pos_y":0, "pos_z":0, "mass":M_BH,
              "vel_x":0, "vel_y":0, "vel_z":0, "name":"SM Black Hole"}
sun_star = {"pos_x":0,  "pos_y":Rsun, "pos_z":0, "mass":Msun,
            "vel_x":Vsun, "vel_y": 0, "vel_z":0, "name":"Sun"}
#big_star = {"pos_x":0,  "pos_y":Rsun*, "pos_z":0, "mass":Msun*50,
#            "vel_x":Vsun*, "vel_y": 0, "vel_z":0, "name":"Big"}

########################### CLASS DEFINITIONS ##################################

# defines a single body (star)
class Body:
    def __init__(self, pos_x, pos_y, pos_z, mass,
                       vel_x, vel_y, vel_z, name="star",
                       acl_x=0, acl_y=0, acl_z=0):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.mass  = mass
        self.vel_x = vel_x
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


# defines a galaxy
class Galaxy:
    def __init__(self, stars, pos_x, pos_y, pos_z, mass,
                              vel_x, vel_y, vel_z, name="MW"):
        self.stars = stars
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
            star = self.calc_body_acceleration(star_ind)
            star.update_velocity(timestep=timestep)
            star.update_position(timestep=timestep)


    # computes the acceleration on a single body from all other bodies in galaxy
    # takes:   galaxy, index of target body in its galaxy's star list
    # returns: target body with updated acceleration
    def calc_body_acceleration(self, body_ind):
        # body is the current star we are calculating acceleration for
        body = self.stars[body_ind]
        body.acl_x, body.acl_y, body.acl_z = 0, 0, 0

        # creates list of stars other than current star
        other_stars = [s for i,s in enumerate(self.stars) if i != body_ind]
        # loops through rest of stars in galaxy
        for star in other_stars:
            dist = euc_dist(body,star)
            frac = (G * star.mass) / dist**3
            body.acl_x += frac * (star.pos_x - body.pos_x)
            body.acl_y += frac * (star.pos_y - body.pos_y)
            body.acl_z += frac * (star.pos_z - body.pos_z)
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

# computes the euclidean distance between two bodies
# takes:   the target star, another star
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
    #max_x = 0
    #max_y = 0
    #max_z = 0
    max_dim = 0
    for i,loc in enumerate(locations):
        #currmax_x = max(loc["x"])
        #currmax_y = max(loc["y"])
        #currmax_z = max(loc["z"])
        #if max_x < currmax_x:
        #    max_x = currmax_x
        #if max_y < currmax_y:
        #    max_y = currmax_y
        #if max_z < currmax_z:
        #    max_z = currmax_z
        curr_max = max(max(loc["x"]),max(loc["y"]),max(loc["z"]))
        if max_dim < curr_max:
            max_dim = curr_max
        ax.plot(loc["x"], loc["y"], loc["z"], c = colors[i%7],label=loc["name"])

    ax.set_xlim([-max_dim, max_dim])
    ax.set_ylim([-max_dim, max_dim])
    ax.set_zlim([-max_dim, max_dim])
    #ax.set_xlim([-max_x, max_x])
    #ax.set_ylim([-max_y, max_y])
    #ax.set_zlim([-max_z, max_z])
    ax.legend()

    if file_out:
        plt.savefig(file_out)
    else:
        plt.show()


######################### PROGRAM STARTS RUNNING HERE ##########################
if __name__ == '__main__':
    BH  = Body(black_hole["pos_x"], black_hole["pos_y"], black_hole["pos_z"],
               black_hole["mass"] , black_hole["vel_x"], black_hole["vel_y"],
               black_hole["vel_z"], black_hole["name"])
    sun = Body(sun_star["pos_x"], sun_star["pos_y"], sun_star["pos_z"],
               sun_star["mass"] , sun_star["vel_x"], sun_star["vel_y"],
               sun_star["vel_z"], sun_star["name"])
    stars = []
    stars.append(BH)
    stars.append(sun)


    #sun = {"location":point(0,0,0), "mass":2e30, "velocity":point(0,0,0)}
    #earth = {"location":point(0,1.5e11,0), "mass":6e24,
    #        "velocity":point(30000,0,0)}
    #jupiter = {"location":point(0,7.7e11,0), "mass":1e28,
    #        "velocity":point(13000,0,0)}
    #pluto = {"location":point(0,3.7e12,0), "mass":1.3e22,
    #"velocity":point(4748,0,0)}
    #sun = Body(0,0,0,2e30,0,0,0,"sun")
    #earth = Body(0,1.5e11,0,6e24,30000,0,0,"earth")
    #jupiter = Body(0,7.7e11,0,1e28,13000,0,0,"jupiter")
    #pluto = Body(0,3.7e12,0,1e22,4748,0,0,"pluto")

    #stars = [sun,earth,jupiter,pluto]
    MW  = Galaxy(stars, black_hole["pos_x"], black_hole["pos_y"],
                 black_hole["pos_z"], black_hole["mass"] , black_hole["vel_x"],
                 black_hole["vel_y"], black_hole["vel_z"], "Milky Way")

    time_step = 10000
    num_steps = 100000000
    report_freq = 1000
    locations = MW.simulate_galaxy(time_step, num_steps, report_freq)
    #print(locations[0]["x"])
    #print(locations[0]["y"])
    show_galaxy(locations)




