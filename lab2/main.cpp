#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

enum class RUN_MODE
{
    GPU,
    CPU
};

int kernel_execution(RUN_MODE mode, std::size_t iteration_count)
{
    constexpr std::size_t GROUP_SIZE = 16;
    // SIN(x^2) + COS(Y) dx dy [0, 5] [0, 5] expected value (-2.155004)
    constexpr double expected_val = -2.155004;

    double dx = 5.f / iteration_count;
    double dy = 5.f / iteration_count;

    std::vector<double> result(256, 0.f);

    uint64_t dev_time;
    auto dev_selector = [=](sycl::device device)
    {
        if (mode == RUN_MODE::GPU)
        {
            if (device.is_gpu())
                return 1;
            else
                return -1;
        }
        if (mode == RUN_MODE::CPU)
        {
            if (device.is_cpu())
                return 1;
            else
                return -1;
        }
    };

    try
    {
        uint64_t dev_time;
        uint64_t host_time;
        sycl::queue queue(dev_selector, sycl::property_list{sycl::property::queue::enable_profiling{}});
        std::cout << "Running on device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
        std::cout << "Size: " << iteration_count << " x " << iteration_count << "\n\n";
        {
            sycl::buffer<double> buffer(result.data(), result.size());
            sycl::event dev_event = queue.submit([&](sycl::handler &h)
                                                 {
                auto accessor = buffer.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<2>(sycl::range<2>(iteration_count, iteration_count),
                               sycl::range<2>(GROUP_SIZE, GROUP_SIZE)), [=](sycl::nd_item<2> item){
                        double x = dx * (item.get_global_id(0) + 0.5);
                        double y = dy * (item.get_global_id(1) + 0.5);
                        double val = (sycl::sin(x*x) + sycl::cos(y))*dx*dy;
                        double sum = sycl::reduce_over_group(item.get_group(), val, std::plus<double>());

                        if (item.get_local_id(0) == 0 && item.get_local_id(1) == 0) {
                            accessor[item.get_group(0) * item.get_group_range(0) + item.get_group(1)] = sum;
                        }
                    }); });
            queue.wait();

            sycl::event host_event = queue.submit([&](sycl::handler &h) {});
            queue.wait();
            uint64_t dev_start = dev_event.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t dev_end = dev_event.get_profiling_info<sycl::info::event_profiling::command_end>();
            std::cout << "Execution time: " << dev_end - dev_start << std::endl;
        }

        double res = 0;
#pragma omp parallel for reduction(+ : res)
        for (std::size_t i = 0; i < GROUP_SIZE * GROUP_SIZE; i++)
            res += result[i];

        std::cout << "Calculated value: " << res << "\n";
        std::cout << "Expected value: " << expected_val << "\n";
        std::cout << "Error value: " << std::abs(res - expected_val) << "\n";
    }
    catch (sycl::exception const &e)
    {
        std::cout << e.what();
    }
    return 0;
}

int main(int argc, char const *argv[])
{
    std::size_t iteration_count = std::stoi(argv[1]);
    std::string str_mode(argv[2]);

    RUN_MODE mode;

    if (str_mode == "GPU")
    {
        mode = RUN_MODE::GPU;
    }
    else if (str_mode == "CPU")
    {
        mode = RUN_MODE::CPU;
    }
    else
    {
        std::cout << "ERROR: Run mode is not supported yet. Select from GPU and CPU." << std::endl;
        return 0;
    }

    return (kernel_execution(mode, iteration_count));
}