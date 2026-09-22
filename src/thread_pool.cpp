#include "thread_pool.h"

ThreadPool::ThreadPool(unsigned num_threads)
    : num_threads_(num_threads == 0 ? 1 : num_threads) {
    workers_.reserve(num_threads_);
    for (unsigned t = 0; t < num_threads_; ++t) {
        workers_.emplace_back([this, t]() { worker_loop(t); });
    }
}

ThreadPool::~ThreadPool() {
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    for (auto& th : workers_) th.join();
}

void ThreadPool::worker_loop(unsigned thread_index) {
    std::uint64_t seen_generation = 0;
    while (true) {
        std::uint64_t gen;
        while ((gen = generation_.load(std::memory_order_acquire)) == seen_generation) {
            if (stop_.load(std::memory_order_acquire)) return;
            /* Yield rather than pure-spin so idle worker threads don't peg a core at 100% 
            between benchmark calls in this demo. A latency-critical production pool might 
            spin without yielding to shave off scheduler wake-up latency instead.*/
            std::this_thread::yield();
        }
        seen_generation = gen;
        if (stop_.load(std::memory_order_acquire)) return;
        task_(thread_index, num_threads_);
        completed_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void ThreadPool::run(const std::function<void(unsigned, unsigned)>& fn) {
    /*Safe to overwrite task_ here without a lock: the previous round only 
    returns to the caller once every worker's completed_ increment is 
    visible (acquire, below), which happens-after that worker finished 
    reading the previous task_ -- so no worker is touching task_ right now.*/
    task_ = fn;
    completed_.store(0, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    while (completed_.load(std::memory_order_acquire) < num_threads_) {
        std::this_thread::yield();
    }
}
