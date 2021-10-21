# Module that generates synthetic data (both, synthetic 'true positions' and synthetic 'reader data')

## Assumption: People only move in 2D, physical space can be described as coordinate system with only integer coordinates



# Idea 1: Generate random movement (brownian motion) of true position based on object speed within boundaries

# Idea 2: Generate random movement by taking a random end point and letting the true position move to that point in a linear (or other) fashion

# Idea 3: Combine idea 1 & 2 and have a random chance for both types at every timestep (Idea: people in dense environments move similar to brownian motion, in empty environments they can directly move to their target)

# Idea 4: NEW APPROACH: People move based on grid with A* Algorithm, if space before them is already occupied they wait for one timestep before executing their next step (1. Generate path with A* based on room map, 2. Set goal tile for everyone, 
#                                                                                                                                                   3. Set next step tile,  4. Everyone take one step (one after another) if tile not occupied, 5. Repeat 3.&4.)

#Import to generate unique identifiers according to RFC 4122
import uuid

class Person(object):

    def __init__(self, position_x, position_y, speed_x, speed_y, identifier = str(uuid.uuid4())):
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.position_x = position_x
        self.position_y = position_y
        self.identifier = str(uuid.uuid4())

    def __str__(self):
        return(str('Person with ID: ' + self.identifier + 
        '\nand position (x,y): ' + str(self.position_x) + ', ' + str(self.position_y) +
        '\nand speed (x,y): ' + str(self.speed_x) + ', ' + str(self.speed_y)))

    def set_position_x(self, position_x):
        self.position_x = position_x

    def set_position_y(self, position_y):
        self.position_y = position_y

    def get_position_x(self):
        return(self.position_x)

    def get_position_y(self):
        return(self.position_y)

    def get_position(self):
        position = (self.get_position_x(), self.get_position_y())

        return(position) 

    # Simulate straight movement from starting point (self.position_x,self.position_y) to target point (end_position_x,end_position_y)
    # Returns movement dict 
    ## TODO: include speed and sampling rate
    def get_straight_line(self, end_position_x, end_position_y, speed=1, sampling_rate = 1):

            points = []
            issteep = abs(end_position_y-self.position_y) > abs(end_position_x-self.position_x)
            if issteep:
                self.position_x, self.position_y = self.position_y, self.position_x
                end_position_x, end_position_y = end_position_y, end_position_x
            rev = False
            if self.position_x > end_position_x:
                self.position_x, end_position_x = end_position_x, self.position_x
                self.position_y, end_position_y = end_position_y, self.position_y
                rev = True
            deltax = end_position_x - self.position_x
            deltay = abs(end_position_y-self.position_y)
            error = int(deltax / 2)
            y = self.position_y
            ystep = None
            if self.position_y < end_position_y:
                ystep = 1
            else:
                ystep = -1
            for x in range(self.position_x, end_position_x + 1):
                if issteep:
                    points.append((y, x))
                else:
                    points.append((x, y))
                error -= deltay
                if error < 0:
                    y += ystep
                    error += deltax
            # Reverse the list if the coordinates were reversed
            if rev:
                points.reverse()

            return points

    def walk_straight_line(self, end_position_x, end_position_y, speed = 1, sampling_rate = 1):
        new_position = self.get_straight_line(end_position_x, end_position_y, speed, sampling_rate)[-1]

        (new_position_x, new_position_y) = new_position

        self.set_position_x(new_position_x)
        self.set_position_y(new_position_y)

        return(self.get_position())



## Function generates random movement based on provided speed and starting position, but will never extend beyond set boundaries 
### TODO: Think about the fact, that in reality the movement will not be truly random, but rather quite linear (people move in a planned direction and keep their trajectory most of the time except when they need to avoid obstacles)
def generate_random_movement(object_speed_x, object_speed_y, starting_position, boundary_x, boundary_y, ):



    return None



#print(get_line(1,5,10,10))

ueli = Person(1, 1, 4, 4)

print(ueli.get_straight_line(8, 7)) 

print(ueli.get_position())

print(ueli.walk_straight_line(8, 7)) 
print(ueli.get_position())

print(ueli)


















#Backup code snippet

# import random
# def brownianMotion(timePoints):
#     brownianTrajectory = [0]
#     for t in range(len(timePoints)-1):
#         randomNumber = random.gauss(0, timePoints[t+1]-timePoints[t])
#         brownianTrajectory.append(brownianTrajectory[t] + randomNumber)
#     return brownianTrajectory



# tp = [0,1,2,3,4,5,6,7,8,9,10]

# print(brownianMotion(tp))



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation

# # size of the crowd
# N = 100

# def gen_data():
#     """ init position and speed of each people """
#     x = y = np.zeros(N)
#     theta = np.random.random(N) * 360 / (2 * np.pi)
#     v0 = 0.1
#     vx, vy = v0 * np.cos(theta), v0 * np.sin(theta)
#     return np.column_stack([x, y, vx, vy])

# def init():
#     pathcol.set_offsets([[], []])
#     return pathcol,

# def update(i, pathcol, data):
#     data[:, 0:2] += data[:, 2:4]
#     data[:, 2] = np.where(np.abs(data[:, 0]) > 5, -data[:, 2], data[:, 2])
#     data[:, 3] = np.where(np.abs(data[:, 1]) > 5, -data[:, 3], data[:, 3])
#     pathcol.set_offsets(data[:, 0:2])
#     return [pathcol]

# fig = plt.figure()
# ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
# pathcol = plt.scatter([], [])
# data = gen_data()
# anim = animation.FuncAnimation(fig, update, init_func=init,
#                                fargs=(pathcol, data), interval=0, blit=True)
# plt.show()