
#!/usr/bin/env python

from VectorQueue import VectorQueue
from Model import Model
import math
import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String

def callback(data):
	rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
	#Alice.queueUpdate(data)

class Robot:
	running = True
	speed = 0.8
	angle = 0
	name = 'Alice'
	vector_queue = VectorQueue() #create queue
	MAILBOX = 7 #Sets the number of messeges to hold
	ideal_pub = rospy.Publisher('ideals', Float64MultiArray, queue_size = MAILBOX)
	execute_pub = rospy.Publisher('execute', String, queue_size = MAILBOX)
	rospy.Subscriber("ideals", Float64MultiArray, callback)

	def __init__(self, name):
		self.name = name
		rospy.init_node(name, anonymous=True)
		self.rate = rospy.Rate(10)
		while self.running == True:
			model = Model(self.speed)
			#run loops to update model
			model.addObstacle((math.pi/2, 15))
			model.addObstacle((4 * math.pi/7 , 30))
			model.addRobot((math.pi/2, 10, 0.9, math.pi/2))
			model.addRobot((3*math.pi/2, 10, 0.7, math.pi/2))
	
			#accept sensor input to model
			self.vector_queue.addVector(model.generateIdeal())
			compromise = self.vector_queue.createCompromise()
			print (compromise)
	
			ideal = Float64MultiArray(data=model.generateIdeal())
			self.ideal_pub.publish(ideal)
			compromise = String(self.name + "," + 
				str(compromise[1]) + "," + 
				str(compromise[0]))
			execute = String(data=compromise)
			self.execute_pub.publish(execute)
			self.rate.sleep()

			#pass compromise off
			#update speed and angle
			self.running = False #breaks loop in tests

	def convertCartesian(self, x, y):
		theta = math.atan(y/x)
		magnitude = (x ** 2 + y ** 2) ** 0.5
		return (theta, magnitude)

#Alice = Robot('Alice')

