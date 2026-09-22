#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

/*A persistent pool of worker threads reused across many parallel calls, 
in contrast to Stage 4's matmul_threaded, which spawns and joins a fresh 
std::thread on every single call. Creating an OS thread typically costs 
tens of microseconds; fine for one call on a huge problem, but wasteful 
if the same parallel kernel is invoked thousands of times (a training 
loop calling matmul repeatedly, say). This pool starts its worker threads 
once in the constructor, parks them between jobs, and wakes them via a 
lock-free generation counter, so repeated calls only pay for the actual 
work, not thread creation/teardown.*/
class ThreadPool {
public:
    explicit ThreadPool(unsigned num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /* Runs fn(thread_index, num_threads) on every worker thread and 
    blocks until all of them have returned. fn is expected to derive 
    its own slice of work from thread_index/num_threads -- this is a 
    fork-join barrier parallel-for, not a work-stealing task queue, 
    which is the  right fit here because every call partitions the 
    same n x n problem  evenly rather than handing out variable-sized 
    independent tasks.*/
    void run(const std::function<void(unsigned, unsigned)>& fn);

    unsigned size() const { return num_threads_; }

private:
    void worker_loop(unsigned thread_index);

    unsigned num_threads_;
    std::vector<std::thread> workers_;
    std::function<void(unsigned, unsigned)> task_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<bool> stop_{false};
    std::atomic<unsigned> completed_{0};
};
