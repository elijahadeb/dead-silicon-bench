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

  return 0;
}
