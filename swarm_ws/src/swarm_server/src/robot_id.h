#ifndef RID_H
#define RID_H
#include <stdlib.h>
#include <map>

char *rid_indexing[] = {"DE", "PA", "NJ", "GA", "CT", "MA", "MD", "SC", "NH", "VA", "NY", "NC", 
                                "RI", "VT", "KY", "TN", "OH", "LA", "IN", "MS", "IL", "AL", "ME", "MO", 
                                "AR", "MI", "FL", "TX", "IA", "WI", "CA", "MN", "OR", "KA", "WV", "NV", 
                                "NE", "CO", "ND", "SD", "MT", "WA", "ID", "WY", "UT", "OK", "NM", "AZ", 
                                "AK", "HI"};

// creating a map for all the string to integer values
std::map<char*, int> inst()
{
  std::map<char*, int> map;
  for (size_t i = 0; i < sizeof(rid_indexing) / sizeof(rid_indexing[0]); i++)
  {
    map.insert(std::pair<char*, int>(rid_indexing[i], i));
  }
  return map;
}

std::map<char*, int> rid_map = inst(); // running before main to make sure this structure exists
#endif