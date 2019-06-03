#ifndef RID_H
#define RID_H
#include <stdlib.h>
#include <map>
#include <string>

// generic dataset for all the states
std::string rid_indexing[] = { "DE", "PA", "NJ", "GA", "CT", "MA", "MD", "SC",
		"NH", "VA", "NY", "NC", "RI", "VT", "KY", "TN", "OH", "LA", "IN", "MS",
		"IL", "AL", "ME", "MO", "AR", "MI", "FL", "TX", "IA", "WI", "CA", "MN",
		"OR", "KA", "WV", "NV", "NE", "CO", "ND", "SD", "MT", "WA", "ID", "WY",
		"UT", "OK", "NM", "AZ", "AK", "HI" };

std::map<std::string, int> initializeRIDs() {
	std::map<std::string, int> map; // map containing all the conversions
	for (size_t i = 0; i < sizeof(rid_indexing) / sizeof(rid_indexing[0]);
			i++) { // adding all states in order from 0-49
		map.insert(std::pair<std::string, int>(rid_indexing[i], i));
	}
	map.insert(std::pair<std::string, int>(std::string("XX"), -1)); // sending to all case
	map.insert(std::pair<std::string, int>(std::string("YY"), -2)); // logging only
	return map;
}

std::map<std::string, int> rid_map = initializeRIDs(); // making default mapp

#endif
