#pragma once

#include <random>
#include <string>
#include <vector>
#include <iostream>

double get_random_val(double a, double b)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> urd(a, b);
    return urd(gen);
}

namespace lab3Mode
{
    enum class RUN_MODE
    {
        GPU,
        CPU
    };

    RUN_MODE CUR_MODE;
    std::size_t MAX_ITERATIONS;
    std::size_t CUR_N;
    double TARGET_ACCURACY;
}

namespace lab3Utils
{
    struct ComputedResult
    {
        std::string name;
        float duration;
        float accuracy;
        std::size_t iteration_cur;
    };

    void A_randomize(std::vector<double> &vec)
    {
        for (auto &i : vec)
            i = get_random_val(-10, 10);

        for (std::size_t i = 0; i < lab3Mode::CUR_N; i++)
        {
            double sum = 0.0;
            for (std::size_t j = 0; j < lab3Mode::CUR_N; j++)
            {
                if (j != i)
                    sum += fabs(vec[i * lab3Mode::CUR_N + j]);
            }
            vec[i * lab3Mode::CUR_N + i] = get_random_val(sum + 1.0, sum + 10.0);
        }
    }

    void B_randomize(std::vector<double> &vec)
    {
        for (auto &i : vec)
            i = get_random_val(-10, 10);
    }

    void out_result(ComputedResult res)
    {
        std::cout << "Iteration count: " << res.iteration_cur << std::endl;
        std::cout << res.name << " Time: " << res.duration << " ms   Accuracy: " << res.accuracy << std::endl;
    }

    double norm(double *vec, int n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
            sum += pow(vec[i], 2);
        return sqrt(sum);
    }

    double norm_of_difference(double *vec1, double *vec2, int n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
            sum += pow(vec1[i] - vec2[i], 2);
        return sqrt(sum);
    }
    double calc_final_accuracy(
        const std::vector<double> &A,
        const std::vector<double> &x,
        const std::vector<double> &b)
    {

        std::vector<double> resvec(lab3Mode::CUR_N);
        for (int i = 0; i < lab3Mode::CUR_N; i++)
        {
            for (int j = 0; j < lab3Mode::CUR_N; j++)
                resvec[i] += A[j * lab3Mode::CUR_N + i] * x[j];
            resvec[i] -= b[i];
        }
        return norm(resvec.data(), lab3Mode::CUR_N);
    }
}
