# Compiler
CXX = clang++

# Compiler flags
CXXFLAGS = -std=c++17 $(shell pkg-config --cflags opencv4)
CXXFLAGS += -I$(HOME)/onnxruntime/include

# Linker flags
LDFLAGS = $(shell pkg-config --libs opencv4)
LDFLAGS += -L$(HOME)/onnxruntime/lib -lonnxruntime

# Make methods
buildFeatures:
	$(CXX) $(CXXFLAGS) buildFeatures.cpp csv_util.cpp featureMethods.cpp -o buildFeatures $(LDFLAGS)

matchImage:
	$(CXX) $(CXXFLAGS) matchImage.cpp csv_util.cpp featureMethods.cpp -o matchImage $(LDFLAGS)

# Clean
clean:
	rm -f buildFeatures feature.csv matchImage