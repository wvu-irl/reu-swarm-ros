#from graphics import *
import tkinter
from tkinter  import *
import sys
from Robot import update

class SelfPortrait:
	tolerance = 0
	speed = 0
	name = "Alice"
	compromise = ()
	ideal = ()
	root = Tk()
	root.title("Alice Self-Portrait")
	
	def refresh():
		Robot.refresh()
	
	def SelfPortrait():
		refresh = Button(root, text = "Click to refresh", command = call_refresh)
		refresh.grid(column = 1, row = 1)
	
		_tolerance = Label(root, text = "Tolerance is: " + str(tolerance))
		_tolerance.grid(column = 0, row = 2)
	
		_speed = Label(root, text = "Speed is: " + str(speed))
		_speed.grid(column = 0, row = 3)
	
		_name = Label(root, text = name)
		_name.grid(column = 1, row = 0)
	
		#_compromise = Label(root, text = 
	

Alice = SelfPortrait()
