#include "./lab3.h"
#include <CL/sycl.hpp>

lab3Utils::ComputedResult compute_device(sycl::queue &queue, const std::vector<double> &A, const std::vector<double> &b)
{
    int iter = 0;
    uint64_t kernel_time_nano = 0;

    double *device_A = sycl::malloc_device<double>(A.size(), queue);
    double *device_b = sycl::malloc_device<double>(b.size(), queue);
    double *device_x_prev = sycl::malloc_device<double>(lab3Mode::CUR_N, queue);
    double *device_x_cur = sycl::malloc_device<double>(lab3Mode::CUR_N, queue);

    std::vector<double> x_prev(lab3Mode::CUR_N);
    for (std::size_t i = 0; i < lab3Mode::CUR_N; i++)
        x_prev[i] = b[i] / A[i * lab3Mode::CUR_N + i];

    std::vector<double> x_cur(lab3Mode::CUR_N);

    queue.memcpy(device_A, A.data(), A.size() * sizeof(double)).wait();
    queue.memcpy(device_b, b.data(), b.size() * sizeof(double)).wait();
    queue.memcpy(device_x_prev, x_prev.data(), x_prev.size() * sizeof(double)).wait();
    do
    {
        sycl::event event = queue.submit([&](sycl::handler &h)
                                         {
			sycl::stream s(1024, 80, h);
			h.parallel_for(sycl::range<1>(lab3Mode::CUR_N), [=](sycl::item<1> item) {
				int i = item.get_id(0);
				int n = item.get_range(0);
				double sum = 0.0;

				for (int j = 0; j < n; j++)
					if (j != i) sum += device_A[j * n + i] * device_x_prev[j];

				device_x_cur[i] = (device_b[i] - sum) / device_A[i * n + i];
				std::swap(device_x_prev[i], device_x_cur[i]);
				}); });
        queue.wait();
        queue.memcpy(x_prev.data(), device_x_prev, lab3Mode::CUR_N * sizeof(double)).wait();
        queue.memcpy(x_cur.data(), device_x_cur, lab3Mode::CUR_N * sizeof(double)).wait();
        uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        kernel_time_nano += (end - start);
        iter++;
    } while (iter < lab3Mode::MAX_ITERATIONS &&
             (lab3Utils::norm_of_difference(x_cur.data(), x_prev.data(), lab3Mode::CUR_N) / lab3Utils::norm(x_cur.data(), lab3Mode::CUR_N)) >= lab3Mode::TARGET_ACCURACY);

    sycl::free(device_A, queue);
    sycl::free(device_b, queue);
    sycl::free(device_x_prev, queue);
    sycl::free(device_x_cur, queue);

    auto res = lab3Utils::ComputedResult();
    res.name = "[DEVICE]";
    res.duration = kernel_time_nano / 1e6;
    res.accuracy = lab3Utils::calc_final_accuracy(A, x_prev, b);
    res.iteration_cur = iter;

    return res;
}

lab3Utils::ComputedResult compute_shared(sycl::queue &queue, const std::vector<double> &A, const std::vector<double> &b)
{
    int iter = 0;
    uint64_t kernel_time_nano = 0;

    double *shared_A = sycl::malloc_shared<double>(A.size(), queue);
    double *shared_b = sycl::malloc_shared<double>(b.size(), queue);
    double *shared_x_prev = sycl::malloc_shared<double>(lab3Mode::CUR_N, queue);
    double *shared_x_cur = sycl::malloc_shared<double>(lab3Mode::CUR_N, queue);

    queue.memcpy(shared_A, A.data(), A.size() * sizeof(double)).wait();
    queue.memcpy(shared_b, b.data(), b.size() * sizeof(double)).wait();

    for (std::size_t i = 0; i < lab3Mode::CUR_N; i++)
        shared_x_prev[i] = shared_b[i] / shared_A[i * lab3Mode::CUR_N + i];
    do
    {
        sycl::event event = queue.submit([&](sycl::handler &h)
                                         {
			sycl::stream s(1024, 80, h);
			h.parallel_for(sycl::range<1>(lab3Mode::CUR_N), [=](sycl::item<1> item) {
				int i = item.get_id(0);
				int n = item.get_range(0);
				double sum = 0.0;

				for (int j = 0; j < n; j++)
					if (j != i) sum += shared_A[j * n + i] * shared_x_prev[j];

				shared_x_cur[i] = (shared_b[i] - sum) / shared_A[i * n + i];
				std::swap(shared_x_prev[i], shared_x_cur[i]);
				}); });
        queue.wait();
        uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        kernel_time_nano += (end - start);
        iter++;
    } while (iter < lab3Mode::MAX_ITERATIONS &&
             (lab3Utils::norm_of_difference(shared_x_cur, shared_x_prev, lab3Mode::CUR_N) / lab3Utils::norm(shared_x_cur, lab3Mode::CUR_N)) >= lab3Mode::TARGET_ACCURACY);
    std::vector<double> x_prev(shared_x_prev, shared_x_prev + lab3Mode::CUR_N);

    sycl::free(shared_A, queue);
    sycl::free(shared_b, queue);
    sycl::free(shared_x_prev, queue);
    sycl::free(shared_x_cur, queue);

    auto res = lab3Utils::ComputedResult();
    res.name = "[SHARED]";
    res.duration = kernel_time_nano / 1e6;
    res.accuracy = lab3Utils::calc_final_accuracy(A, x_prev, b);
    res.iteration_cur = iter;

    return res;
}

lab3Utils::ComputedResult compute_accessors(sycl::queue &queue, const std::vector<double> &A, const std::vector<double> &b)
{
    int iter = 0;
    uint64_t kernel_time_nano = 0;

    std::vector<double> x_prev(lab3Mode::CUR_N);
    for (std::size_t i = 0; i < lab3Mode::CUR_N; i++)
        x_prev[i] = b[i] / A[i * lab3Mode::CUR_N + i];

    std::vector<double> x_cur(lab3Mode::CUR_N);

    sycl::buffer<double> buf_A(A.data(), A.size());
    sycl::buffer<double> buf_b(b.data(), b.size());

    do
    {
        sycl::buffer<double> buf_x_prev(x_prev.data(), x_prev.size());
        sycl::buffer<double> buf_x_cur(x_cur.data(), x_cur.size());

        sycl::event event = queue.submit([&](sycl::handler &h)
                                         {
			sycl::stream s(1024, 80, h);
			auto acc_A = buf_A.get_access<sycl::access::mode::read, sycl::access::target::device>(h);
			auto acc_b = buf_b.get_access<sycl::access::mode::read, sycl::access::target::device>(h);
			auto acc_x_prev = buf_x_prev.get_access<sycl::access::mode::read_write>(h);
			auto acc_x_cur = buf_x_cur.get_access<sycl::access::mode::read_write>(h);

			h.parallel_for(sycl::range<1>(lab3Mode::CUR_N), [=](sycl::item<1> item) {
				int i = item.get_id(0);
				int n = item.get_range(0);
				double sum = 0.0;

				for (int j = 0; j < n; j++)
					if (j != i) sum += acc_A[j * n + i] * acc_x_prev[j];

				acc_x_cur[i] = (acc_b[i] - sum) / acc_A[i * n + i];
				std::swap(acc_x_prev[i], acc_x_cur[i]);
				}); });
        queue.wait();
        uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        kernel_time_nano += (end - start);
        iter++;
    } while (iter < lab3Mode::MAX_ITERATIONS &&
             (lab3Utils::norm_of_difference(x_cur.data(), x_prev.data(), lab3Mode::CUR_N) / lab3Utils::norm(x_cur.data(), lab3Mode::CUR_N)) >= lab3Mode::TARGET_ACCURACY);

    auto res = lab3Utils::ComputedResult();
    res.name = "[ACCESSORS]";
    res.duration = kernel_time_nano / 1e6;
    res.accuracy = lab3Utils::calc_final_accuracy(A, x_prev, b);
    res.iteration_cur = iter;

    return res;
}

int compute()
{
    std::size_t N = lab3Mode::CUR_N;
    std::vector<double> A(N * N);
    std::vector<double> B(N);

    lab3Utils::A_randomize(A);
    lab3Utils::B_randomize(B);

    auto dev_selector = [](sycl::device device)
    {
        if (lab3Mode::CUR_MODE == lab3Mode::RUN_MODE::GPU && device.is_gpu())
            return 1;
        if (lab3Mode::CUR_MODE == lab3Mode::RUN_MODE::CPU && device.is_cpu())
            return 1;
        return -1;
    };

    sycl::queue queue(dev_selector, sycl::property_list{sycl::property::queue::enable_profiling{}});
    std::cout << "Running on device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

    lab3Utils::ComputedResult res;

    res = compute_device(queue, A, B); // COLD
    res = compute_device(queue, A, B); // HOT
    lab3Utils::out_result(res);

    res = compute_shared(queue, A, B); // COLD
    res = compute_shared(queue, A, B); // HOT
    lab3Utils::out_result(res);

    res = compute_accessors(queue, A, B); // COLD
    res = compute_accessors(queue, A, B); // HOT
    lab3Utils::out_result(res);

    return 0;
}

int main(int argc, char const *argv[])
{
    if (argc != 5)
    {
        std::cout << "Please determine the values.\n";
        std::cout << "1. N (matrix dimention)\n";
        std::cout << "2. TARGET ACCURACY\n";
        std::cout << "3. MAX ITERATION\n";
        std::cout << "4. DEVICE MODE (CPU or GPU)\n";
        return 0;
    }
    lab3Mode::CUR_N = std::stoi(argv[1]);
    lab3Mode::TARGET_ACCURACY = std::stod(argv[2]);
    lab3Mode::MAX_ITERATIONS = std::stoi(argv[3]);
    std::string str_mode(argv[4]);
    if (str_mode == "GPU")
    {
        lab3Mode::CUR_MODE = lab3Mode::RUN_MODE::GPU;
    }
    else if (str_mode == "CPU")
    {
        lab3Mode::CUR_MODE = lab3Mode::RUN_MODE::CPU;
    }
    else
    {
        std::cout << "ERROR: Run mode is not supported yet. Select from GPU and CPU." << std::endl;
        return 0;
    }

    return (compute());
}