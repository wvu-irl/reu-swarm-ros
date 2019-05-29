

from VectorQueue import VectorQueue
from Model import Model
import math

class Robot:
    running = True
    speed = 0.8
    angle = 0

    def convertCartesian(x, y)
        theta = math.atan(y/x)
        magnitude = (x ** 2 + y ** 2) ** 0.5
        return (theta, magnitude)

    while running:
        model = Model(speed)
        #run loops to update model
        model.addObstacle((math.pi/2, 15))
        model.addObstacle((4 * math.pi/7 , 30))
        model.addRobot((math.pi/2, 10, 0.9, math.pi/2))
        model.addRobot((3*math.pi/2, 10, 0.7, math.pi/2))
        vector_queue = VectorQueue() #create queue
        #accept sensor input to model
        vector_queue.addVector(model.generateIdeal())
        compromise = vector_queue.createCompromise()
        print (compromise)
        #pass compromise off
        #update speed and angle
        running = False #breaks loop in tests
