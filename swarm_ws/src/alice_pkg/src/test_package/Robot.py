
#!/usr/bin/env python

from VectorQueue import VectorQueue
from Model import Model
import math
import rospy
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
from wvu_swarm_std_msgs.msg import robot_command
from wvu_swarm_std_msgs.msg import alice_mail_array
#alice_mail/3

def callback(data):
	rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
	#Alice.queueUpdate(data)

class Robot:
	speed = 0.8
	angle = 0
	name = 'Alice'
	vector_queue = VectorQueue() #create queue
        model = Model(speed)
	MAILBOX = 7 #Sets the number of messeges to hold

	def callToVector(self, data):
		self.vector_queue.queueUpdate(data)

	def callToModel(self, data):
 		self.model.modelUpdate(data)

	def __init__(self, name):
		rospy.init_node(name, anonymous=False)	
		self.name = '%02d' % int(rospy.get_param('~id'))
		print(self.name)
		ideal_pub = rospy.Publisher('ideals', Float64MultiArray, queue_size = self.MAILBOX)
		ex_string = "execute_" + self.name
		execute_pub = rospy.Publisher(ex_string, robot_command, queue_size = self.MAILBOX)
		rospy.Subscriber("ideals", Float64MultiArray, callback)
		sub_string = "alice_mail_" + self.name
		rospy.Subscriber(sub_string, alice_mail_array, self.callToModel)
		
		self.rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			self.model = Model(self.speed)
			#run loops to update model
			#model.addObstacle((math.pi/2, 15))
			#model.addObstacle((4 * math.pi/7 , 30))
			self.model.addRobot((math.pi/2, 10, 0.9, math.pi/2))
			self.model.addRobot((3*math.pi/2, 10, 0.7, math.pi/2))
	
			#accept sensor input to model
			self.vector_queue.addVector(self.model.generateIdeal())
			compromise_vector = self.vector_queue.createCompromise()
	
			ideal = Float64MultiArray(data=self.model.generateIdeal())
			ideal_pub.publish(ideal)
			compromise = robot_command()
			compromise.rid = list(self.name)
			compromise.theta = float(compromise_vector[0])
			compromise.r = float(compromise_vector[1])
			execute_pub.publish(compromise)
			self.rate.sleep()

