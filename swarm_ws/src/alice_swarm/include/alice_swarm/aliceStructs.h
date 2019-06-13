#include <string>

class aliceStructs {

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
		float spd;
		float ang;
		std::string name;
	} neighbor;

	typedef struct {
		float dir;
		float dis;
	}  obj;

	typedef struct {
		float dir;
		float dis;
	}  tar;
};
