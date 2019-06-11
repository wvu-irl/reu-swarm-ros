
from Rules import Rules
from SelfPortrait import SelfPortrait
import math
import rospy

'''
This class holds the robot's model of the world
This includes objects, food, and other robots
'''

class Model:

    speed = 0
    obstacles = [] #Saved in form (Direction, distance)
    robots = [] #Saved in form (Direction, distance, speed, angle)
    food = [] #Not implemented, but will probably be stored as (Direction, distance)
    rules = Rules()
    tolerance = 0

    def __init__(self, speed):
        self.speed = speed
    
    def generateIdeal(self):
        ideal = None #Will eventually be of the form 
        tolerance = 1
        while ideal == None and not rospy.is_shutdown():
            ideal_tuple = (self.rules.avoidObstacles(self.obstacles, tolerance),
                           self.rules.avoidRobots(self.robots, tolerance),
                           self.rules.maintainSpacing(self.robots, tolerance),
                           #self.rules.matchAngle(self.robots, tolerance),
                           self.rules.matchSpeed(self.robots, self.speed, tolerance))
            #This will call each rule function in order of priority to create a list
            for rule in ideal_tuple:
                if rule != None:
                    ideal = (rule[0], rule[1], 0, (float(len(ideal_tuple) - ideal_tuple.index(rule) + 1))/(tolerance + 1)) #The +1 is so that priority is never zero, which would mess up VectorQueue
                    #If this method of setting priority ends up being problematic, we could also set priority as follows:
                    #ideal = (rule[0],  0, rule[1])
                    break;
            tolerance += 1
        return ideal

    def addRobot(self, to_add): #Should be of the form (direction, distance, heading)
        self.robots.append(to_add)

    def addObstacle(self, to_add): #Should be of the form (distance, direction)
        self.obstacles.append(to_add)

    def addFood(self, to_add):
        self.food.append(to_add)

    def modelUpdate(self, data):
        for bot in data.neighborMail:
            self.addRobot((bot.theta, bot.distance, bot.heading))

    def updateSpeed(self, new_speed):
        self.speed = new_speed

    def updateAngle(self, new_angle):
        self.angle = new_angle
