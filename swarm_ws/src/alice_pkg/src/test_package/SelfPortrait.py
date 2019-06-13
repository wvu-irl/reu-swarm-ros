from Tkinter import *
import Tkinter
import sys
import Robot
		
class SelfPortrait:
	tolerance = 0
	speed = 0
	name = "Alice"
	compromise = ()
	ideal = ()
	robot = None
	root = Tk()
	root.title("Alice Self-Portrait")
	
	def list_to_string(self, my_list):
		to_return = ""
		for item in my_list:
			to_return += str(item) + "\n"
		return to_return
	
	def call_refresh(self):
		self.root.destroy()
		self.root = Tk()
		self.tolerance = self.robot.model.tolerance
		self.name = self.robot.name
		self.speed = self.robot.speed
		self.compromise = self.robot.compromise_vector
		self.ideal = self.robot.ideal_vector
		self.vector_queue = []
		self.vector_queue = self.robot.vector_queue.vector_queue
		self.rule = self.robot.model.current
		self.robots = self.robot.model.robots
		
		refresh = Button(self.root, text = "Click to refresh", command = self.call_refresh)
		refresh.grid(column = 1, row = 1)
		
		self._compromise = Label(self.root, text = "Compromise vector: " + str(self.compromise))
		self._compromise.grid(column = 1, row = 2)
		
		self._ideal = Label(self.root, text = "Ideal vector: " + str(self.ideal))
		self._ideal.grid(column = 1, row = 3)
		
		self._vector_queue = Label(self.root, text = "Vector queue: " + self.list_to_string(self.vector_queue))
		self._vector_queue.grid(column = 2, row = 1)
		
		self._rule = Label(self.root, text = "Rule: " + self.rule)
		self._rule.grid(column = 0, row = 4)
		
		self._robots = Label(self.root, text = "Robots: " + self.list_to_string(self.robots))
		self._robots.grid(column = 2, row = 2)
		
	
	def __init__(self, robot):
		self.robot = robot
		self.tolerance = self.robot.model.tolerance
		self.name = self.robot.name
		self.speed = self.robot.speed
		self.compromise = self.robot.compromise_vector
		self.ideal = self.robot.ideal_vector
		self.vector_queue = self.robot.vector_queue.vector_queue
		self.rule = self.robot.model.current
		self.robots = self.robot.model.robots
		
		refresh = Button(self.root, text = "Click to refresh", command = self.call_refresh)
		refresh.grid(column = 1, row = 1)
	
		_tolerance = Label(self.root, text = "Tolerance is: " + str(self.tolerance))
		_tolerance.grid(column = 0, row = 2)
	
		_speed = Label(self.root, text = "Speed is: " + str(self.speed))
		_speed.grid(column = 0, row = 3)
	
		_name = Label(self.root, text = self.name)
		_name.grid(column = 1, row = 0)

		self._compromise = Label(self.root, text = "Compromise vector: " + str(self.compromise))
		self._compromise.grid(column = 2, row = 1)
		
		self._ideal = Label(self.root, text = "Ideal vector: " + str(self.ideal))
		self._ideal.grid(column = 1, row = 3)
		
		self._vector_queue = Label(self.root, text = "Vector queue: " + self.list_to_string(self.vector_queue))
		self._vector_queue.grid(column = 2, row = 1)
		
		self._rule = Label(self.root, text = "Rule: " + self.rule)
		self._rule.grid(column = 0, row = 4)
		
		self._robots = Label(self.root, text = "Robots: " + self.list_to_string(self.robots))
		self._robots.grid(column = 2, row = 2)
		
		
	
		#_compromise = Label(root, text = 
