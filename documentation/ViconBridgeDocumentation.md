# Vicon Bridge Usage
[- Overview](#overview) 

[- Running the bridge](#running)

[- Launch parameters](#launch-parameters)

[- Internals](#internals) 

## Overview
This package was originally provided by Markus Achtelik (markus.achtelik@mavt.ethz.ch) via the BSD license. Modifications have been made by researchers at the WVU Interactive Robotics Lab to tailor this package towards our specific swarm project.

The ``vicon_bridge`` package is used exactly as a bridge. It takes data from the Tracker software using the Vicon Data Stream SDK and outputs a ROS topic to connect this data to any ROS node requiring bot positions and orientations. The majority of this code is difficult to parse and is highly dependent on the Data Stream SDK, which itself is a complex codebase. Only a small portion (the ``vicon_bridge.cpp`` file) is directly relevant to our project and contains our only modifications to the package thus far.

## Running
The ``vicon_bridge`` package pulls data from Tracker software using the Vicon Data Stream SDK. As such, Tracker must be running on another computer (in our lab, the Windows computer plugged into the Vicon network).

The Tracker computer's IP must be recorded, and the computer running ``vicon_bridge`` must share a network with it. The parameter ``datastream_hostport`` in the file ``vicon.launch`` must match the Tracker computer's IP address with port 801.

To run the bridge, use ``vicon.launch`` in package ``vicon_bridge``. Tracked robots will be output as a ``vicon_bot_array`` message type in a topic called ``vicon_array``. A tf2 transform tree will be constructed in the ``tf`` topic.

## Launch Parameters
The file ``vicon.launch`` can be modified to change what items are tracked and how they are output.
- stream_mode: This parameter should always be set to ClientPull. Its functionality is beyond the necessary scope of this project.
- datastream_hostport: This specifies the path that the Vicon Data Stream will be hosted at, i.e. the address of the computer running Tracker. Default port for Data Stream is 801.
- tf_ref_frame_id: The name of the transform frame that should be used as the reference frame by the tf library.
- publish_markers: When true, an additional topic ``vicon_markers`` will be published, holding an array of every individual point that the Vicon system detects.
- publish_transforms: When true, an individual topic will be published for each _object_ that the vicon is tracking as transform messages.
- swarmbot_prefix: The string that must be matched in each Vicon object for it to be processed as a bot.

## Internals
TODO: explain the .cpp file somewhat
