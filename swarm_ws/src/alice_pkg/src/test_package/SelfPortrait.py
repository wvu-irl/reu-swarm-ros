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
	
	def call_refresh(self):
		self.tolerance = self.robot.model.tolerance
		self.name = self.robot.name
		self.speed = self.robot.speed
	
	def __init__(self, robot):
		self.robot = robot
		self.tolerance = self.robot.model.tolerance
		self.name = self.robot.name
		self.speed = self.robot.speed
		self.compromise = self.robot.compromise_vector
		
		refresh = Button(self.root, text = "Click to refresh", command = self.call_refresh)
		refresh.grid(column = 1, row = 1)
	
		_tolerance = Label(self.root, text = "Tolerance is: " + str(self.tolerance))
		_tolerance.grid(column = 0, row = 2)
	
		_speed = Label(self.root, text = "Speed is: " + str(self.speed))
		_speed.grid(column = 0, row = 3)
	
		_name = Label(self.root, text = self.name)
		_name.grid(column = 1, row = 0)

		_compromise = Label(self.root, text = str(self.compromise))
		_compromise.grid(column = 1, row = 2)
	
		#_compromise = Label(root, text = 
