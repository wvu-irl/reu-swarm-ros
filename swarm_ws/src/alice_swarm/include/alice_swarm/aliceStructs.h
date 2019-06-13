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
		std::vector<neighbor> neighbors;
		std::vector<obj> obstacless;
		std::vector<obj> targets;
		int name;
	} mail;
};
