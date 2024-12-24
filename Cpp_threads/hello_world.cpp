#include <iostream>
#include <thread>
#include <mutex>
#include <cmath>
#include <vector> 
#include <atomic>

void accumulate_sin_sum(std::atomic<int> &total){
    for (size_t i = 0; i < 1'000'000; i++)
    {
        total+=1;
    }
}


int main()
{
    std::atomic<int> total=0;
    int num_threads =std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads/2);
    //create threads
    for (int i=0;i<threads.size();i++){
        threads[i] = std::thread(accumulate_sin_sum, std::ref(total));
    }
    for (auto &t : threads) {
        t.join();
    }
    std::cout<<"Total: "<<total<<std::endl;
}