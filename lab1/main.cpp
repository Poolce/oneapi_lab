#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

int main()
{
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    for (size_t i = 0; i < platforms.size(); i++)
    {
        std::cout << "Platform #" << i << ": " << platforms[i].get_info<sycl::info::platform::name>() << std::endl;
        std::vector<sycl::device> devices = platforms[i].get_devices();
        for (size_t j = 0; j < devices.size(); j++)
        {
            std::cout << "\tDevice #" << j << ": " << devices[j].get_info<sycl::info::device::name>() << std::endl;
        }
    }

    constexpr int globalSize = 5;
    for (size_t i = 0; i < platforms.size(); i++)
    {
        std::vector<sycl::device> devices = platforms[i].get_devices();
        for (size_t j = 0; j < devices.size(); j++)
        {
            std::cout << devices[j].get_info<sycl::info::device::name>() << std::endl;
            sycl::queue q(devices[j]);
            q.submit([&](sycl::handler &h)
                     {
                sycl::stream out(1024, 80, h);
                h.parallel_for(sycl::range<1>(globalSize), [=](sycl::id<1> item) {
                    out << '[' << item.get(0) << "] Hello from platform " << i << " and device " << j << '\n';
                }); });
            q.wait();
        }
    }
}
