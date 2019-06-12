#from graphics import *
import tkinter
from tkinter import *
import sys

class SelfPortrait:
	tolerance = 0
	speed = 0
	root = Tk()
	root.title("Alice Self-Portrait")
	
	refresh = Button(root, text = "Click to refresh")
	Button.grid(column = 1, row = 1)
	
	_tolerance = Label(root, text = "Tolerance is: " + str(tolerance))
	_tolerance.grid(column = 0, row = 2)
	
	_speed = Label(root, text = "Speed is: " + str(speed))
	_speed.grid(column = 0, row = 3)
	

	root.mainloop()	

Alice = SelfPortrait()
