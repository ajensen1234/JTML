#include "sym_trap.h"
#include "point6d.h"
#include <iostream>


int main() {

	// lets run a few tests to see what we can make happen


	Point6D random_pose(25, 65, -1500, 45, 78, 68);
	sym_trap sym;

	Point6D mirror_pose = sym.compute_mirror_pose(random_pose);

	


	std::cout << mirror_pose.x << " " <<
		mirror_pose.y << " " <<
		mirror_pose.z << " " <<
		mirror_pose.za << " " <<
		mirror_pose.xa << " " <<
		mirror_pose.ya << "\n" <<
		random_pose.x << " " <<
		random_pose.y << " " <<
		random_pose.z << " " <<
		random_pose.za << " " <<
		random_pose.xa << " " <<
		random_pose.ya << " ";


}