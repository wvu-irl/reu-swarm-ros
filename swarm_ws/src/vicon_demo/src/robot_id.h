/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2019, WVU Interactive Robotics Laboratory
*                       https://web.statler.wvu.edu/~irl/
* All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#ifndef RID_H
#define RID_H
#include <stdlib.h>
#include <map>
#include <string>

// generic dataset for all the states
std::string rid_indexing[] =
		{ "DE", "PA", "NJ", "GA", "CT", "MA", "MD", "SC", "NH", "VA", "NY", "NC",
				"RI", "VT", "KY", "TN", "OH", "LA", "IN", "MS", "IL", "AL", "ME", "MO",
				"AR", "MI", "FL", "TX", "IA", "WI", "CA", "MN", "OR", "KA", "WV", "NV",
				"NE", "CO", "ND", "SD", "MT", "WA", "ID", "WY", "UT", "OK", "NM", "AZ",
				"AK", "HI" };

std::map<std::string, int> initializeRIDs()
{
	std::map<std::string, int> map; // map containing all the conversions
	for (size_t i = 0; i < sizeof(rid_indexing) / sizeof(rid_indexing[0]); i++)
	{ // adding all states in order from 0-49
		map.insert(std::pair<std::string, int>(rid_indexing[i], i));
	}
	map.insert(std::pair<std::string, int>(std::string("XX"), -1)); // sending to all case
	map.insert(std::pair<std::string, int>(std::string("YY"), -2)); // logging only
	return map;
}

std::map<std::string, int> rid_map = initializeRIDs(); // making default mapp

#endif
