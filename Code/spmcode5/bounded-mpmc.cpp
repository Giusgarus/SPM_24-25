#include <new>
#include <random>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstddef>

constexpr int CACHE_LINE_SIZE=64;

// ----- auxiliary functions
constexpr unsigned long BACKOFF_MIN = 4;
constexpr unsigned long BACKOFF_MAX = 32;

inline bool isPowerOf2(unsigned x) {
	return (x != 0 && (x & (x-1)) == 0);
}

inline unsigned long nextPowerOf2(unsigned long x) {
    unsigned long p=1;
    while (x>p) p <<= 1;
    return p;
}

inline int random(const int min, const int max) {
		// better to have different per-thread seeds....
		thread_local std::mt19937   
			generator(std::hash<std::thread::id>()(std::this_thread::get_id()));
		std::uniform_int_distribution<int> distribution(min,max);
		return distribution(generator);
}
// emulate some work
inline void random_work(const int min, const int max) {
	auto start = std::chrono::high_resolution_clock::now();
	auto end   = start + std::chrono::nanoseconds(random(min, max));
    while(std::chrono::high_resolution_clock::now() < end) ;	
}

// -------------------
template<typename T>
class MPMC_Ptr_Queue {
private:
    struct element_t {
        std::atomic<unsigned long> seq;
        T*                         data;
    };    
public:
    MPMC_Ptr_Queue() {}
    ~MPMC_Ptr_Queue() { delete [] buf;  }

    inline bool init(size_t size) {
        if (size<2) size=2;
        // we need a size that is a power 2 in order to set the mask 
        if (!isPowerOf2(size)) size = nextPowerOf2(size);
        mask = size-1;

        buf = new element_t[size];
        if (!buf) return false;
        for(size_t i=0;i<size;++i) {
            buf[i].data = nullptr;
            buf[i].seq.store(i,std::memory_order_relaxed);          
        }
        pwrite.store(0,std::memory_order_relaxed);        
        pread.store(0,std::memory_order_relaxed);
        return true;
    }
    inline bool push(T *const data) {
        unsigned long pw, seq;
        element_t * node;
        unsigned long bk = BACKOFF_MIN;
        do { // CAS loop
            pw    = pwrite.load(std::memory_order_relaxed);
            node  = &buf[pw & mask];
            seq   = node->seq.load(std::memory_order_acquire);
            
            if (pw == seq) { 
                if (pwrite.compare_exchange_weak(pw, pw+1,std::memory_order_relaxed))
                    break;
                // exponential delay with max value
				for (unsigned i = 0; i < bk; ++i) {
					std::this_thread::yield();
				}
				bk = (bk < BACKOFF_MAX) ? (bk << 1) : BACKOFF_MAX;
            } else 
                if (pw > seq) return false; // queue full
        } while(1);
        node->data = data;
        node->seq.store(seq+1,std::memory_order_release);
        return true;
    }	
    inline bool pop(T*& data) {
        unsigned long pr, seq;
        element_t * node;
        unsigned long bk = BACKOFF_MIN;

        do { // CAS loop
            pr    = pread.load(std::memory_order_relaxed);
            node  = &buf[pr & mask];
            seq   = node->seq.load(std::memory_order_acquire);

            long diff = seq - (pr+1);
            if (diff == 0) {
                if (pread.compare_exchange_weak(pr, (pr+1), std::memory_order_relaxed))
                    break;
                // exponential delay with max value
				for (unsigned i = 0; i < bk; ++i) {
					std::this_thread::yield();
				}
				bk = (bk < BACKOFF_MAX) ? (bk << 1) : BACKOFF_MAX;
            } else { 
                if (diff < 0) return false; // queue empty
            }
        } while(1);
        data = node->data;
        node->seq.store((pr+mask+1), std::memory_order_release);
        return true;
    }
private:
	/// Pointer to the location where to write to
	alignas(CACHE_LINE_SIZE)
	std::atomic<unsigned long> pwrite;
	/// Pointer to the location where to read from
	alignas(CACHE_LINE_SIZE)
	std::atomic<unsigned long> pread;

    element_t *                 buf;
    unsigned long               mask;
};



template<typename T>
class BlockingQueue {
public:
    BlockingQueue(size_t capacity) : capacity(capacity) { }

    void push(const T& value) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_full.wait(lock, [this](){ return queue.size() < capacity; });
        queue.push_back(value);
        cv_not_empty.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_empty.wait(lock, [this](){ return !queue.empty(); });
        T value = queue.front();
        queue.pop_front();
        cv_not_full.notify_one();
        return value;
    }
    
private:
    std::deque<T> queue;
    size_t capacity;
    std::mutex mtx;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
};


void test_lockfree_queue(const int nprod, const int ncons,
						 const int size, const int numiterations,
						 const int min, const int max) {
    MPMC_Ptr_Queue<int> queue;
    queue.init(size);

    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};

    auto producer = [&]() {
        for (int i = 0; i < numiterations; ++i) {
			random_work(min, max);
            int* item = new int(i); 
            while (!queue.push(item)) ;
            produced.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto consumer = [&]() {
        while (1) {
            int* data = nullptr;
            if (queue.pop(data)) {
				if (*data==-1) { delete data; break; }
                delete data;
                consumed.fetch_add(1, std::memory_order_relaxed);
				random_work(min, max);
            } 
        }
    };

    std::vector<std::thread> consumers;
	std::vector<std::thread> producers;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < ncons; ++i)
        consumers.emplace_back(consumer);
    for (int i = 0; i < nprod; ++i)
        producers.emplace_back(producer);
    for (auto& t : producers) t.join();
	for (int i = 0; i < ncons; ++i)
		queue.push(new int(-1));	
	for (auto& t : consumers) t.join();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Lock-free queue: Produced " << produced.load() 
              << ", Consumed " << consumed.load() 
              << " in " << elapsed.count() << " seconds.\n";
}

void test_blocking_queue(const int nprod, const int ncons,
						 const int size, const int numiterations,
						 const int min, const int max) {
    BlockingQueue<int*> queue(size);

    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};

    auto producer = [&]() {
        for (int i = 0; i < numiterations; ++i) {
			random_work(min,max);           
            queue.push(new int(i));
            produced.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto consumer = [&]() {
        while (1) {
            int* item = queue.pop();
			if (*item == -1) { delete item; break;}
            delete item;
            consumed.fetch_add(1, std::memory_order_relaxed);
			random_work(min,max);
        }
    };

    std::vector<std::thread> producers;
	std::vector<std::thread> consumers;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < ncons; ++i)
        consumers.emplace_back(consumer);
    for (int i = 0; i < nprod; ++i)
        producers.emplace_back(producer);
    for (auto& t : producers) t.join();
	for (int i = 0; i < ncons; ++i)
		queue.push(new int(-1));	
    for (auto& t : consumers) t.join();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Lock-based queue: Produced " << produced.load() 
              << ", Consumed " << consumed.load() 
              << " Time " << elapsed.count() << " s\n";
}

int main(int argc, char *argv[]) {
	int random_work_min=10;
	int random_work_max=100;
	int numiterations=100000;
	if (argc<4) {
		std::printf("Usage: %s size #prod #cons [numiter=%d] [workmin=%dns] [workmax=%dns]\n", argv[0],
					numiterations,
					random_work_min,
					random_work_max);
		return -1;
	}
	int size =std::stol(argv[1]);
	int nprod=std::stol(argv[2]);
	int ncons=std::stol(argv[3]);
	if (size<=0) {
		std::printf("Invalid size. Should be >0\n");
		return -1;
	}
	if (argc>4) {
		numiterations=std::stol(argv[4]);
		if (numiterations<=0) {
			std::printf("Invalid numiterations. Should be >0\n");
			return -1;
		}
	}
	if (argc>5) {
		random_work_min=std::stol(argv[5]);
	}
	if (argc>6) {
		random_work_max=std::stol(argv[6]);
	}
	if (random_work_min>random_work_max) {
		std::printf("Invalid random work, %d>%d\n", random_work_min, random_work_max);
		return -1;
	}
	
    test_lockfree_queue(nprod,ncons,size,numiterations,random_work_min,random_work_max);
    test_blocking_queue(nprod,ncons,size,numiterations,random_work_min,random_work_max);
}

