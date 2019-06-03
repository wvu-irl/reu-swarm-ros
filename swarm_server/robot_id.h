#ifndef RID_H
#define RID_H
#include <stdlib.h>
#include <map>
#include <string>

std::string rid_indexing[] = {"DE", "PA", "NJ", "GA", "CT", "MA", "MD", "SC", "NH", "VA", "NY", "NC",
                                "RI", "VT", "KY", "TN", "OH", "LA", "IN", "MS", "IL", "AL", "ME", "MO",
                                "AR", "MI", "FL", "TX", "IA", "WI", "CA", "MN", "OR", "KA", "WV", "NV",
                                "NE", "CO", "ND", "SD", "MT", "WA", "ID", "WY", "UT", "OK", "NM", "AZ",
                                "AK", "HI"};

std::map<std::string, int> initializeRIDs()
{
  std::map<std::string, int> map;
  for (size_t i = 0; i < sizeof(rid_indexing) / sizeof(rid_indexing[0]); i++) {
    map.insert(std::pair<std::string, int>(rid_indexing[i], i));
  }
  return map;
}

std::map<std::string, int> rid_map = initializeRIDs();

#endif
