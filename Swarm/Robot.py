

from VectorQueue import VectorQueue
from Model import Model

class Robot:
    running = True

    while running:
        model = Model()
        #run loops to update model
        vector_queue = VectorQueue() #create queue
        #accept sensor input to model
        vector_queue.addVector(model.generateIdeal())
        compromise = vector_queue.createCompromise()
        print (compromise)
        #pass compromise off
        #update speed and angle
        running = False #breaks loop in tests
