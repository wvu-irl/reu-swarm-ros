#include <string>

class AliceStructs {

	typedef struct {
		float dir;
		float spd;
		float dis;
		float pri;
	} ideal;

	typedef struct {
		float dir;
		float mag;
	} vel;

	typedef struct {
		float dir;
		float dis;
		float ang;
		int name;
	} neighbor;

	typedef struct {
		float dir;
		float dis;
	}  obj;

	typedef struct {
		float dir;
		float dis;
	}  tar;

	typedef struct {
		std::vector<neighbor> neighbors;
		std::vector<obj> objs;
		std::vector<tar> tars;
		int name;
	} mail;
};
