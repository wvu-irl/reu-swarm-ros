"""
This class accepts ideal vectors, adds them to a list, and gives a compromise vector
The code that is commented out is for testing VectorQueue locally, and can be uncommented
for that purpose. However, doing so means that it will not run properly in Robot, so be warned.
"""
class VectorQueue:
    #accepts vectors of the form (Direction, speed, distance, priority)
    vector_queue = []
    '''
    def __init__(self, to_add): #for testing only
        self.vector_queue += to_add
    '''
    def addVector(self, to_add):
        self.vector_queue.append(to_add)

    def queueUpdate(self, data):
        self.addVector(data.data)
        
    def returnFirst(self):
        return self.vector_queue.pop()

    def createCompromise(self):
        compromise_angle = 0 #angle to pass on
        compromise_speed = 0 #speed to pass on
        priority = 0 #keeps track of sum, so that new vectors affect it the correct amount
        
        while len(self.vector_queue) != 0:
            current = self.returnFirst()
            current_priority = (current[3] / (current[2] + 1)) ** 2 #factoring in distance
            compromise_angle = (compromise_angle * priority + current[0] * current_priority)/(priority + current_priority)
            compromise_speed = (compromise_speed * priority + current[1] * current_priority)/(priority + current_priority)
            priority += current_priority
            
        return (compromise_angle, compromise_speed)
'''
tester = VectorQueue([(0, 0, 0, 1), (1, 1, 0, 1), (2, 2, 0, 1), (3, 3, 0, 1)])
print (tester.createCompromise())
'''
