#include <chrono>
#include <cmath>
#include <cstddef>
#include <fcntl.h>
#include <future>
#include <immintrin.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <xmmintrin.h>
// horizontal_reduction
static inline float hsum256_ps(__m256 acc) {
  __m128 xmm_high = _mm256_extractf128_ps(acc, 1);
  __m128 xmm_low = _mm256_castps256_ps128(acc);

  __m128 xmm_sum = _mm_add_ps(xmm_low, xmm_high);

  // one more fold.

  __m128 upper_2 = _mm_movehl_ps(xmm_sum, xmm_sum);

  xmm_sum = _mm_add_ps(upper_2, xmm_sum);
  __m128 final_reduce = _mm_shuffle_ps(xmm_sum, xmm_sum, 1);
  xmm_sum = _mm_add_ps(final_reduce, xmm_sum);

  return _mm_cvtss_f32(xmm_sum);
}

// single avx2

double calculate_vector_dot_product(float *a, float *b, size_t num_element) {
  double total = 0.0;

  const size_t BLOCK = 100000; // must be a multiple of 32
  size_t nblocks = num_element / BLOCK;

  for (size_t c = 0; c < nblocks; c++) {
    __m256 ymm_1 = _mm256_setzero_ps();
    __m256 ymm_2 = _mm256_setzero_ps();
    __m256 ymm_3 = _mm256_setzero_ps();
    __m256 ymm_4 = _mm256_setzero_ps();

    for (size_t j = 0; j + 32 <= BLOCK; j += 32) {
      __m256 a1 = _mm256_loadu_ps(&a[c * BLOCK + j]);
      __m256 a2 = _mm256_loadu_ps(&a[c * BLOCK + j + 8]);
      __m256 a3 = _mm256_loadu_ps(&a[c * BLOCK + j + 16]);
      __m256 a4 = _mm256_loadu_ps(&a[c * BLOCK + j + 24]);

      __m256 b1 = _mm256_loadu_ps(&b[c * BLOCK + j]);
      __m256 b2 = _mm256_loadu_ps(&b[c * BLOCK + j + 8]);
      __m256 b3 = _mm256_loadu_ps(&b[c * BLOCK + j + 16]);
      __m256 b4 = _mm256_loadu_ps(&b[c * BLOCK + j + 24]);

      ymm_1 = _mm256_fmadd_ps(a1, b1, ymm_1);
      ymm_2 = _mm256_fmadd_ps(a2, b2, ymm_2);
      ymm_3 = _mm256_fmadd_ps(a3, b3, ymm_3);
      ymm_4 = _mm256_fmadd_ps(a4, b4, ymm_4);
    }

    __m256 acc_ymm1_ymm2 = _mm256_add_ps(ymm_1, ymm_2);
    __m256 acc_ymm3_ymm4 = _mm256_add_ps(ymm_3, ymm_4);
    __m256 acc = _mm256_add_ps(acc_ymm1_ymm2, acc_ymm3_ymm4);

    total += hsum256_ps(acc);
  }

  // scalar tail

  for (size_t i = BLOCK * nblocks; i < num_element; i++) {
    total += (double)a[i] * (double)b[i];
  }

  return total;
}

// multithreaded avx2

double threaded_avx2_fnct(float *a, float *b, size_t start_index,
                          size_t end_index) {

  double total = 0.0;

  const size_t BLOCK = 100000;
  size_t j = start_index;

  while (j + 32 <= end_index) {
    size_t block_end = j + BLOCK;
    if (block_end > end_index)
      block_end = end_index;

    __m256 ymm_1 = _mm256_setzero_ps();
    __m256 ymm_2 = _mm256_setzero_ps();
    __m256 ymm_3 = _mm256_setzero_ps();
    __m256 ymm_4 = _mm256_setzero_ps();

    for (; j + 32 <= block_end; j += 32) {
      __m256 a1 = _mm256_loadu_ps(&a[j]);
      __m256 a2 = _mm256_loadu_ps(&a[j + 8]);
      __m256 a3 = _mm256_loadu_ps(&a[j + 16]);
      __m256 a4 = _mm256_loadu_ps(&a[j + 24]);

      __m256 b1 = _mm256_loadu_ps(&b[j]);
      __m256 b2 = _mm256_loadu_ps(&b[j + 8]);
      __m256 b3 = _mm256_loadu_ps(&b[j + 16]);
      __m256 b4 = _mm256_loadu_ps(&b[j + 24]);

      ymm_1 = _mm256_fmadd_ps(a1, b1, ymm_1);
      ymm_2 = _mm256_fmadd_ps(a2, b2, ymm_2);
      ymm_3 = _mm256_fmadd_ps(a3, b3, ymm_3);
      ymm_4 = _mm256_fmadd_ps(a4, b4, ymm_4);
    }

    __m256 acc_ymm1_ymm2 = _mm256_add_ps(ymm_1, ymm_2);
    __m256 acc_ymm3_ymm4 = _mm256_add_ps(ymm_3, ymm_4);
    __m256 acc = _mm256_add_ps(acc_ymm1_ymm2, acc_ymm3_ymm4);

    total += hsum256_ps(acc);
  }

  // scalar tail

  for (; j < end_index; j++) {
    total += (double)a[j] * (double)b[j];
  }

  return total;
}
// mmap-binary file / scalar

int main() {

  std::cout << std::scientific << std::setprecision(15);

  size_t num_element = 100000000;

  int fd_1 = open("a.f32", O_RDONLY);
  int fd_2 = open("b.f32", O_RDONLY);

  if (fd_1 == -1) {
    std::cerr << "failed to open file: a.f32" << std::endl;
    close(fd_2);
    return 1;
  }

  if (fd_2 == -1) {
    std::cerr << "failed to open file: b.f32" << std::endl;
    close(fd_1);
    return 1;
  }

  struct stat file_info_a;
  struct stat file_info_b;

  if (fstat(fd_1, &file_info_a) == -1) {
    std::cerr << "failed to get a.f32 file stat";
    close(fd_1);
    close(fd_2);

    return 1;
  }
  if (fstat(fd_2, &file_info_b) == -1) {
    std::cerr << "failed to get b.f32 file stat";
    close(fd_1);
    close(fd_2);
    return 1;
  }

  size_t file_size_a = file_info_a.st_size;
  size_t file_size_b = file_info_b.st_size;

  if (file_size_a != num_element * sizeof(float)) {
    std::cerr << "a.f32 is " << file_size_a
              << " what is expected: " << num_element * sizeof(float) << "\n";
    close(fd_1);
    close(fd_2);
    return 1;
  }

  if (file_size_b != num_element * sizeof(float)) {
    std::cerr << "b.f32 is " << file_size_b
              << " what is expected: " << num_element * sizeof(float) << "\n";
    close(fd_1);
    close(fd_2);
    return 1;
  }

  void *mapped_a = mmap(NULL, file_size_a, PROT_READ, MAP_PRIVATE, fd_1, 0);

  if (mapped_a == MAP_FAILED) {
    std::cerr << "mmap a failed" << std::endl;
    close(fd_1);
    close(fd_2);
    return 1;
  }

  void *mapped_b = mmap(NULL, file_size_b, PROT_READ, MAP_PRIVATE, fd_2, 0);

  if (mapped_b == MAP_FAILED) {
    std::cerr << "mmap b failed" << std::endl;
    close(fd_1);
    close(fd_2);

    munmap(mapped_a, file_size_a);
    return 1;
  }

  float *a = static_cast<float *>(mapped_a);
  float *b = static_cast<float *>(mapped_b);

  // scalar - warm loop
  float warm = 0.0f;

  for (size_t i = 0; i < num_element; i += 32) {
    warm += a[i] * b[i];
  }

  if (warm == 1234.0f) {
    std::cout << "";
  }

  double accum = 0.0;

  // time harness

  const auto start_time = std::chrono::steady_clock::now();

  for (size_t i = 0; i < num_element; i++) {
    float prod = a[i] * b[i];
    accum += prod;
  }

  const auto end_time = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed_seconds(end_time - start_time);

  std::cout << "scalarf32 result: " << accum << std::endl;
  std::cout << elapsed_seconds.count() << "s\n";

  double bytes = 2.0 * num_element * sizeof(float);
  double bandwidth = bytes / 1e9 / elapsed_seconds.count();

  std::cout << "scalarf32_bandwidth: " << bandwidth << "gb/s\n";

  // float64 reference

  double reference_accum = 0.0;

  for (size_t i = 0; i < num_element; i++) {
    double prod = static_cast<double>(a[i]) * static_cast<double>(b[i]);
    reference_accum += prod;
  }

  std::cout << "scalar reference dot product: " << reference_accum << std::endl;

  const auto start_time_avx2_single = std::chrono::steady_clock::now();
  double avx2_result = calculate_vector_dot_product(a, b, num_element);
  const auto end_time_avx2_single = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed_seconds_avx2_single(
      end_time_avx2_single - start_time_avx2_single);

  std::cout << elapsed_seconds_avx2_single.count() << "s\n";
  double bandwidth_avx2_single =
      bytes / 1e9 / elapsed_seconds_avx2_single.count();
  std::cout << "avx2 bandwidth: " << bandwidth_avx2_single << "gb/s\n";

  std::cout << "avx2: " << avx2_result << std::endl;

  double err_scalar = std::abs(accum - reference_accum) / reference_accum;
  double err_avx2 = std::abs(avx2_result - reference_accum) / reference_accum;

  std::cout << "scalar rel err: " << err_scalar
            << (err_scalar < 1e-3 ? " -> passed" : " -> failed") << "\n";
  std::cout << "avx2 rel err: " << err_avx2
            << (err_avx2 < 1e-3 ? " -> passed" : " -> failed") << "\n";

  size_t total_cores = std::thread::hardware_concurrency();

  if (total_cores == 0) {
    std::cerr << "warning: hardware_concurrency() returned 0, falling back to "
                 "1 core, single thread avx2"
              << "\n";

    total_cores = 1;
  }

  std::cout << "hardware_concurrency: " << total_cores << "\n";

  size_t c_size = (num_element / total_cores) & ~size_t(31);

  std::vector<std::future<double>> futures;

  const auto start_t = std::chrono::steady_clock::now();
  for (size_t workers_id = 0; workers_id < total_cores; workers_id++) {
    size_t start_index = workers_id * c_size;
    size_t end_index =
        (workers_id == total_cores - 1) ? num_element : start_index + c_size;

    futures.push_back(std::async(std::launch::async, threaded_avx2_fnct, a, b,
                                 start_index, end_index));
  }

  double total_result = 0.0;

  for (auto &future : futures) {
    total_result += future.get();
  }
  const auto end_t = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed_s(end_t - start_t);

  std::cout << elapsed_s.count() << "s\n";
  double threaded_bandw = bytes / 1e9 / elapsed_s.count();

  std::cout << "threaded_avx2_bandwidth: " << threaded_bandw << "gb/s\n";

  std::cout << "threaded_avx2: " << total_result << std::endl;

  double err_threaded_avx2 =
      std::abs(total_result - reference_accum) / reference_accum;

  std::cout << "threaded_avx2 rel err: " << err_threaded_avx2
            << (err_threaded_avx2 < 1e-3 ? " -> passed" : " -> failed") << "\n";

  munmap(mapped_a, file_size_a);
  munmap(mapped_b, file_size_b);
  close(fd_1);
  close(fd_2);
  return 0;
}
