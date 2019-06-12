
#!/usr/bin/env python

from VectorQueue import VectorQueue
from Model import Model
from SelfPortrait import SelfPortrait
import math
import rospy
import Tkinter
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
	compromise_vector = None
	ideal_vector = None
	neighbors = []
	MAILBOX = 7 #Sets the number of messeges to hold

	def is_neighbor(self, to_check):
		to_return = 0
		for neighbor in self.neighbors:
			if neighbor[0] == to_check:
				to_return = neighbor[1]
		return to_return

	def callToVector(self, data):
		if self.is_neighbor(data.data[4]) != 0 and len(self.vector_queue.vector_queue) < 8:
			self.vector_queue.queueUpdate((data.data[0], data.data[1], self.is_neighbor(data.data[4]), data.data[3], data.data[4]))

	def callToModel(self, data):
		for neighbor in data.neighborMail:
			if self.is_neighbor(neighbor.id) == 0:
 				self.model.addRobot(neighbor)
 				self.neighbors.append((neighbor.id, neighbor.distance))

	def setUpNode(self):
		rospy.init_node(self.name, anonymous=False)	
		self.name = int(rospy.get_param('~id'))
		self.model = Model(self.name)
		self.ideal_pub = rospy.Publisher('ideals', Float64MultiArray, queue_size = self.MAILBOX)
		ex_string = "execute_" + str(self.name)
		self.execute_pub = rospy.Publisher(ex_string, robot_command, queue_size = self.MAILBOX)
		rospy.Subscriber("ideals", Float64MultiArray, self.callToVector)
		sub_string = "alice_mail_" + str(self.name)
		rospy.Subscriber(sub_string, alice_mail_array, self.callToModel)
		self.rate = rospy.Rate(10)
	'''
	def tester(self):
		self.model.addObstacle((math.pi/2, 15))
		self.model.addObstacle((4 * math.pi/7 , 30))
		self.model.addRobot((math.pi/2, 10, 0.9, math.pi/2))
		self.model.addRobot((3*math.pi/2, 10, 0.7, math.pi/2))
	'''
	def publishData(self):
		ideal = Float64MultiArray(data=self.model.generateIdeal())
		self.ideal_pub.publish(ideal)
		compromise = robot_command()
		compromise.rid = self.name
		compromise.theta = float(self.compromise_vector[0])
		compromise.r = float(self.compromise_vector[1])
		self.speed = compromise.r #Should eventually be updated
		self.execute_pub.publish(compromise)
		

	def __init__(self, name):
		
		self.setUpNode()
		self_portrait = SelfPortrait(self)
		should_update = True
		while not rospy.is_shutdown():
			if should_update:
				try:
					self_portrait.root.update()
				except Tkinter.TclError:
					should_update = False
			
			self.neighbors = []
			self.model.clear()
	
			#self.tester() #Uncomment to include test inputs
			self.ideal_vector = self.model.generateIdeal()
			self.vector_queue.addVector(self.ideal_vector)
			self.compromise_vector = self.vector_queue.createCompromise()
			
			self.vector_queue.vector_queue = []
	
			self.publishData()

			self.rate.sleep()

