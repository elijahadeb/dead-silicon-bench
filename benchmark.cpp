#include <chrono>
#include <cstddef>
#include <fcntl.h>
#include <immintrin.h>
#include <iomanip>
#include <ios>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xmmintrin.h>

float calculate_vector_dot_product(float *a, float *b) {
  double total = 0.0;

  const auto start_time = std::chrono::steady_clock::now();

  for (size_t c = 0; c < 1000; c++) {
    __m256 ymm_1 = _mm256_setzero_ps();
    __m256 ymm_2 = _mm256_setzero_ps();
    __m256 ymm_3 = _mm256_setzero_ps();
    __m256 ymm_4 = _mm256_setzero_ps();

    for (size_t j = 0; j < 100000; j += 32) {
      __m256 a1 = _mm256_loadu_ps(&a[c * 100000 + j]);
      __m256 a2 = _mm256_loadu_ps(&a[c * 100000 + j + 8]);
      __m256 a3 = _mm256_loadu_ps(&a[c * 100000 + j + 16]);
      __m256 a4 = _mm256_loadu_ps(&a[c * 100000 + j + 24]);

      __m256 b1 = _mm256_loadu_ps(&b[c * 100000 + j]);
      __m256 b2 = _mm256_loadu_ps(&b[c * 100000 + j + 8]);
      __m256 b3 = _mm256_loadu_ps(&b[c * 100000 + j + 16]);
      __m256 b4 = _mm256_loadu_ps(&b[c * 100000 + j + 24]);

      ymm_1 = _mm256_fmadd_ps(a1, b1, ymm_1);
      ymm_2 = _mm256_fmadd_ps(a2, b2, ymm_2);
      ymm_3 = _mm256_fmadd_ps(a3, b3, ymm_3);
      ymm_4 = _mm256_fmadd_ps(a4, b4, ymm_4);
    }

    __m256 acc_ymm1_ymm2 = _mm256_add_ps(ymm_1, ymm_2);
    __m256 acc_ymm3_ymm4 = _mm256_add_ps(ymm_3, ymm_4);
    __m256 acc = _mm256_add_ps(acc_ymm1_ymm2, acc_ymm3_ymm4);

    __m128 xmm_high = _mm256_extractf128_ps(acc, 1);
    __m128 xmm_low = _mm256_castps256_ps128(acc);

    __m128 xmm_sum = _mm_add_ps(xmm_low, xmm_high);

    // one more fold.

    __m128 upper_2 = _mm_movehl_ps(xmm_sum, xmm_sum);

    xmm_sum = _mm_add_ps(upper_2, xmm_sum);
    __m128 final_reduce = _mm_shuffle_ps(xmm_sum, xmm_sum, 1);
    xmm_sum = _mm_add_ps(final_reduce, xmm_sum);

    float scalar_conv = _mm_cvtss_f32(xmm_sum);

    total += scalar_conv;
  }
  const auto end_time = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed_seconds(end_time - start_time);

  std::cout << elapsed_seconds.count() << "s\n";
  double bandwidth = 0.8 / elapsed_seconds.count();
  std::cout << "avx2 bandwidth: " << bandwidth << "gb/s\n";

  return total;
}

int main() {

  std::cout << std::scientific << std::setprecision(15);

  int num_element = 100000000;

  int fd_1 = open("a.f32", O_RDONLY);
  int fd_2 = open("b.f32", O_RDONLY);

  if (fd_1 == -1) {
    std::cout << "failed to open file: a.f32" << std::endl;
    close(fd_1);
    return 1;
  }

  if (fd_2 == -1) {
    std::cout << "failed to open file: b.f32" << std::endl;
    close(fd_2);
    return 1;
  }

  struct stat file_info_a;
  struct stat file_info_b;

  size_t size_a = fstat(fd_1, &file_info_a);
  size_t size_b = fstat(fd_2, &file_info_b);

  size_t file_size_a = file_info_a.st_size;
  size_t file_size_b = file_info_b.st_size;

  if (file_size_a == -1) {
    std::cout << "failed to fetch size a" << std::endl;
    close(fd_1);
    return 1;
  } else {
    std::cout << file_size_a << "\n";
  }

  if (file_size_b == -1) {
    std::cout << "failed to fetch size" << std::endl;
    close(fd_2);
    return 1;
  } else {
    std::cout << file_size_b << "\n";
  }
  void *mapped_a = mmap(NULL, file_size_a, PROT_READ, MAP_PRIVATE, fd_1, 0);
  void *mapped_b = mmap(NULL, file_size_b, PROT_READ, MAP_PRIVATE, fd_2, 0);

  if (mapped_a == MAP_FAILED) {
    std::cout << "mmap a failed" << std::endl;
    close(fd_1);
    return 1;
  }

  if (mapped_b == MAP_FAILED) {
    std::cout << "mmap b failed" << std::endl;
    close(fd_2);
    return 1;
  }

  float *a = static_cast<float *>(mapped_a);
  float *b = static_cast<float *>(mapped_b);

  std::cout << "a[0]: " << a[0] << "\n";
  std::cout << "a[1]: " << a[1] << "\n";
  std::cout << "b[0]: " << b[0] << "\n";
  std::cout << "b[1]: " << b[1] << "\n";

  // scalar

  double accum = 0.0;

  float warm = 0.0f;

  for (size_t i = 0; i < num_element; i += 32) {
    warm += a[i] * b[i];
  }

  if (warm == 1234.0f) {
    std::cout << "";
  };

  // time harness|

  const auto start_time = std::chrono::steady_clock::now();

  for (size_t i = 0; i < num_element; i++) {
    float prod = a[i] * b[i];
    accum += prod;
  }

  // float64 reference

  double reference_accum = 0.0;

  for (size_t i = 0; i < num_element; i++) {
    double prod = static_cast<double>(a[i]) * static_cast<double>(b[i]);
    reference_accum += prod;
  }

  std::cout << "scalar reference dot product: " << reference_accum << std::endl;

  const auto end_time = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed_seconds(end_time - start_time);

  std::cout << "scalar dot product: " << accum << std::endl;
  std::cout << elapsed_seconds.count() << "s\n";

  double bandwidth = 0.8 / elapsed_seconds.count();

  std::cout << "bandwidth: " << bandwidth << "gb/s\n";

  double avx2_result = calculate_vector_dot_product(a, b);

  std::cout << "avx2: " << avx2_result << std::endl;

  double err_scalar = std::abs(accum - reference_accum) / reference_accum;
  double err_avx2 = std::abs(avx2_result - reference_accum) / reference_accum;

  std::cout << "scalar rel err: " << err_scalar
            << (err_scalar < 1e-3 ? " -> passed" : " -> failed") << "\n";
  std::cout << "avx2 rel err: " << err_avx2
            << (err_avx2 < 1e-3 ? " -> passed" : " -> failed") << "\n";

  return 0;
}
