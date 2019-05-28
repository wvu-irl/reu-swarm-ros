
from Rules import Rules

'''
This class holds the robot's model of the world
This includes objects, food, and other robots
'''

class Model:

    speed = 0
    obstacles = [] #Saved in form (Direction, distance)
    robots = [] #Saved in form (Direction, distance, speed)
    food = [] #Not implemented, but will probably be stored as (Direction, distance)
    rules = Rules()

    def __init__(self, speed):
        self.speed = speed
    
    def generateIdeal(self):
        ideal = None #Will eventually be of the form 
        tolerance = 0
        while ideal == None:
            ideal_tuple = (rule.avoidObjects(obstacles, tolerance), rules.avoidRobots(robots, tolerance),
                           rules.maintainSpacing(robots, tolerance), rules.matchAngle(robots, tolerance), rules.matchSpeed(robots, speed, tolerance))
            #This will call each rule function in order of priority to create a list
            for rule in ideal_tuple:
                if rule != None:
                    ideal = (rule[0], rule[1], 0, (len(ideal_tuple) - ideal_tuple.index(rule) + 1)/(tolerance + 1)) #The +1 is so that priority is never zero, which would mess up VectorQueue
                    #If this method of setting priority ends up being problematic, we could also set priority as follows:
                    #ideal = (rule[0],  0, rule[1])
                    break
            tolerance += 1
        return ideal

    def addRobot(self, to_add): #Should be of the form (direction, distance, speed)
        self.robots.assign(toAdd)

    def addObstacles(self, to_add): #Should be of the form (distance, direction)
        self.robots.assign(obstacles)

    def addFood(self, to_add):
        self.food.assign(to_add)

    def updateSpeed(self, new_speed):
        speed = new_speed

    def updateAngle(self, new_angle):
        angle = new_angle
