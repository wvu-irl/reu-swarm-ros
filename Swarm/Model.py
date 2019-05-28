'''
This class holds the robot's model of the world
This includes objects, food, and other robots
'''

class Model:

    speed = 0
    angle = 0
    obstacles = [] #Saved in form (Direction, distance)
    robots = [] #Saved in form (Direction, distance, speed, ID)
    food = [] #Not implemented, but will probably be stored as (Direction, distance)
    #rules = Rules()
    
    def generateIdeal(self):
        ideal = None
        tolerance = 0
        while ideal == None:
            ideal_list = ('''rules.getRuleOne(Some info), rules.getRuleTwo(Some info)...''')
            #This will call each rule function in order of priority to create a list
            for rule in ideal_list:
                if rule != None:
                    ideal = (rule, 0, (len(ideal_list) - ideal_list.index(rule) + 1)/(tolerance + 1)) #The +1 is so that priority is never zero, which would mess up VectorQueue
                    #If this method of setting priority ends up being problematic, we could also set priority as follows:
                    #ideal = (rule[0],  0, rule[1])
                    break
            tolerance += 1
        return ideal

    def addRobot(self, to_add): #Should be of the form (direction, distance, speed, ID)
        self.robots.assign(toAdd)

    def addObstacles(self, to_add): #Should be of the form (distance, direction)
        self.robots.assign(obstacles)

    def addFood(self, to_add):
        self.food.assign(to_add)

    def updateSpeed(self, new_speed):
        speed = new_speed

    def updateAngle(self, new_angle):
        angle = new_angle
