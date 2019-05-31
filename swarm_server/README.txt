This is the location for all the server related code

Server allows sending commands to individual robots/clients
  - this requires robots to "register" using an internal command

How to use register command
  - Immediately after connecting to the server the client should send the command
              register <id>
  - where the id is the robot id or a debugging listener
  - robot ids range from 0-49 (ideally, though 49 is not a max)
  - using an id of -2 registers the client as a logging terminal
  
Currently server communicates with strings
