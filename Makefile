CXXFLAGS = -O3 -mavx2 -mfma -Wall -Wextra -std=c++17

all: run

generator: generator.cpp
	g++ $(CXXFLAGS) -o generator generator.cpp

gen: benchmark.cpp
	g++ $(CXXFLAGS) -o gen benchmark.cpp

a.f32 b.f32: generator
	./generator

run: gen a.f32 b.f32
	./gen

clean:
	rm -f gen generator a.f32 b.f32
