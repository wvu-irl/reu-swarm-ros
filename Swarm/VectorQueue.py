"""
This class accepts ideal vectors, adds them to a list, and gives a compromise vector
The code that is commented out is for testing VectorQueue locally, and can be uncommented
for that purpose. However, doing so means that it will not run properly in Robot, so be warned.
"""

class VectorQueue:
    #accepts vectors of the form (Direction, distance, priority)
    vector_queue = []
    '''
    def __init__(self, to_add): #for testing only
        self.vector_queue += to_add
    '''
    def addVector(self, to_add):
        self.vector_queue.append(to_add)
        
    def returnFirst(self):
        return self.vector_queue.pop()

    def createCompromise(self):
        compromise = 0 #angle to pass on
        priority = 0 #keeps track of sum, so that new vectors affect it the correct amount
        
        while len(self.vector_queue) != 0:
            current = self.returnFirst()
            current_priority = current[2] / (current[1] + 1) ** 2 #factoring in distance
            compromise = (compromise * priority + current[0] * current[2])/(priority + current[2])
            priority += current[2]
            
        return compromise
'''
tester = VectorQueue([(1, 1.45, 0), (0.2, 3.02, 5), (0.4, 3.03, 3), (0.6, 1.23, 4), (0.6, 3.10, 3)])
print (tester.createCompromise())
#This should print 0.42666666666666664. If it doesn't, come find Casey
'''
