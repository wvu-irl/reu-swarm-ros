
#import Model
from VectorQueue import VectorQueue

class Robot:
    speed = 0
    angle = 0
    position = (0,0)
    running = True

    '''
    def execute(to_run):
        Talk to hardware team about this function
    '''
    while running:
        vector_queue = VectorQueue() #create queue
        #accept sensor input
        ideal_vector = (1, 0, 0) #find ideal vector using model and rules
        vector_queue.addVector(ideal_vector)
        #send ideal vector
        compromise = vector_queue.createCompromise()
        print (compromise)
        #execute
        #update positon, speed, angle
        running = False #breaks loop in tests
