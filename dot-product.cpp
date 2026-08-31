#include <chrono>
#include <cstddef>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

int main() {

  std::mt19937 gen1(0);
  std::mt19937 gen2(2);

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::vector<float> values_a;
  std::vector<float> values_b;

  for (float i = 0; i < 100; i++) {

    for (float i = 0; i < 1000000; i++) {
      float rand_val_a = dist(gen1);
      float rand_val_b = dist(gen2);

      // std::cout << rand_val_a << "\n";
      // std::cout << rand_val_b << "\n";

      values_a.push_back(rand_val_a);
      values_b.push_back(rand_val_b);
    }
  }

  float *val_ptr_a = values_a.data();
  float *val_ptr_b = values_b.data();

  std::ofstream ostrm1("a.f32", std::ios::binary);
  ostrm1.write(reinterpret_cast<char *>(val_ptr_a),
               values_a.size() * sizeof(float));
  std::ofstream ostrm2("b.f32", std::ios::binary);
  ostrm2.write(reinterpret_cast<char *>(val_ptr_b),
               values_b.size() * sizeof(float));

  values_a.clear();
  values_b.clear();

  int fd_1 = open("a.f32", O_RDONLY);
  int fd_2 = open("b.f32", O_RDONLY);

  if (fd_1 == -1) {
    std::cout << "failed to open file: a.f32" << std::endl;
    close(fd_1);
    return 1;
  }

  if (fd_2 == -1) {
    std::cout << "failed to open file: a.f32" << std::endl;
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

  double accum = 0;

  for (int i = 0; i < 100000000; i++) {
    float prod = a[i] * b[i];
    accum += prod;
  }

  std::cout << "scalar dot product: " << accum << std::endl;
  std::chrono::steady_clock::now();

  return 0;
}
