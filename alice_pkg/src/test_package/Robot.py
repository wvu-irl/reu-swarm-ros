
#!/usr/bin/env python

from VectorQueue import VectorQueue
from Model import Model
import math
import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from wvu_swarm_std_msgs.msg import executeVector
from wvu_swarm_std_msgs.msg import aliceMailArray
#alice_mail/3

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

	def callToVector(self, data):
		vector_queue.queueUpdate(data)

	def callToModel(self, data):
 		model.modelUpdate(data)

	def __init__(self, name):
		self.name = rospy.get_param("name")
		ideal_pub = rospy.Publisher('ideals', Float64MultiArray, queue_size = self.MAILBOX)
		execute_pub = rospy.Publisher('execute', executeVector, queue_size = self.MAILBOX)
		rospy.Subscriber("ideals", Float64MultiArray, callback)
		sub_string = "alice_mail_" + self.name[-2:]
		rospy.Subscriber(sub_string, aliceMailArray, self.callToModel)
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
			compromise_vector = self.vector_queue.createCompromise()
			print (compromise_vector)
	
			ideal = Float64MultiArray(data=model.generateIdeal())
			ideal_pub.publish(ideal)
			compromise = executeVector()
			compromise.ID = self.name
			compromise.heading = compromise_vector[0]
			compromise.magnitude = compromise_vector[1]
			execute_pub.publish(compromise)
			rospy.spin()
			self.rate.sleep()

			self.running = False #breaks loop in tests

