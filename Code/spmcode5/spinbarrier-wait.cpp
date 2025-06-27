#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>

#include <barrier>

class mybarrier {
    uint64_t const count;
    std::atomic<uint64_t> spaces;
    std::atomic<uint64_t> generation;
public:
    mybarrier(unsigned count):
        count(count),spaces(count),generation(0)
    {}
    void arrive_and_wait() {
        const uint64_t my_generation=generation;
        if(!--spaces) { // I'm the last one
            spaces=count;            // reset barrier
			++generation;            // change generation
			generation.notify_all(); // notify all waiting
        } else  {
			generation.wait(my_generation);
        }
    }
};

// 
// https://en.cppreference.com/w/cpp/thread/barrier
//
int main() {
	using namespace std::chrono_literals;
	const int nthreads = 20;

	auto on_completion = []() {
		
		std::printf("ALL have reached the barrier\n");

		std::this_thread::sleep_for(2s);
	};

	//mybarrier bar(nthreads);
	std::barrier bar(nthreads,  on_completion);
	
	auto F=[&](int id) {
		std::this_thread::sleep_for(id*100ms);
		std::printf("Thread %d waiting for barrier\n", id);
		bar.arrive_and_wait();
		std::printf("Thread %d crossed the barrier\n", id);
	};

	std::vector<std::thread> V;
	
	for(int i=0;i<(nthreads-1);++i)
		V.emplace_back(F, i);

	std::this_thread::sleep_for(1000ms);
	std::printf("Thread MAIN waiting for barrier\n");
	bar.arrive_and_wait();
	std::printf("Thread MAIN crossed the barrier\n");
	
	for(auto& v: V) v.join();
	
}

