#include <string>
#include <omp.h>

int main(int argc, char *argv[]) {

	int x=5;
	int y=1;
#pragma omp parallel reduction(*: x) reduction(+:y) num_threads(3)
	{
		x+=3;
		y+=3;
	}
	std::printf("x=%d, y=%d\n", x, y);
}

