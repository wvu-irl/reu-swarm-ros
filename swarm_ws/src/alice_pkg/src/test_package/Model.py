
from Rules import Rules
import math

'''
This class holds the robot's model of the world
This includes objects, food, and other robots
'''

class Model:

    speed = 0
    name = 0.0
    obstacles = [] #Saved in form (Direction, distance)
    robots = [] #Saved in form (Direction, distance, speed, angle, id)
    targets = [] #Not implemented, but will probably be stored as (Direction, distance)
    rules = Rules()

    def __init__(self, speed):
        self.speed = speed
    

    def clear(self):
        self.obstacles = []
        self.robots = []
        self.targets = []

    def generateIdeal(self):
        ideal = None #Will eventually be of the form 
        tolerance = 1
        while ideal == None:
            ideal_tuple = (
                           #self.rules.avoidObstacles(self.obstacles, tolerance),
                           #self.rules.avoidRobots(self.robots, tolerance),
                           #self.rules.maintainSpacing(self.robots, tolerance),
                           self.rules.goToTarget(self.targets, tolerance),
                           #self.rules.matchAngle(self.robots, tolerance),
                           #self.rules.matchSpeed(self.robots, self.speed, tolerance),
                           None
                           )
            #This will call each rule function in order of priority to create a list
            if ideal_tuple != None:
                for rule in ideal_tuple:
                    if rule != None:
                        ideal = (rule[0], rule[1], 0, (float(len(ideal_tuple) - ideal_tuple.index(rule) + 1))/(tolerance + 1), float(self.name)) #The +1 is so that priority is never zero, which would mess up VectorQueue
                        #If this method of setting priority ends up being problematic, we could also set priority as follows:
                        #ideal = (rule[0],  0, rule[1])
                        break;
            tolerance += 1
        #print(ideal)
        return ideal

    def addRobot(self, to_add): #Should be of the form (direction, distance, heading, ID)
        self.robots.append(to_add)

    def addObstacle(self, to_add): #Should be of the form (direction, distance)
        self.obstacles.append(to_add)

    def addTarget(self, to_add): #Should be of the form (direction, distance)
        self.targets.append(to_add)
    
    def setName(self, name):
        self.name = name

    def modelUpdate(self, data):
        for bot in data.neighborMail:
            self.addRobot((bot.theta, bot.distance, bot.heading, bot.id))
        for obs in data.obsPointMail:
            self.addObstacle((obs.theta, obs.radius))
        self.addTarget((data.targetMail.theta, data.targetMail.radius))

    def updateSpeed(self, new_speed):
        self.speed = new_speed

    def updateAngle(self, new_angle):
        self.angle = new_angle
