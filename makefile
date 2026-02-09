# Detect OS
ifeq ($(OS),Windows_NT)
    # Windows settings
    CXX = g++
    OPENCV_DIR = C:/msys64/ucrt64
    ONNX_DIR = C:/onnxruntime
    CXXFLAGS = -std=c++17 -I$(OPENCV_DIR)/include/opencv4 -I$(ONNX_DIR)/include
    LDFLAGS = -L$(OPENCV_DIR)/lib -L$(ONNX_DIR)/lib
    LDFLAGS += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_dnn
    LDFLAGS += -lonnxruntime
    RM = del /Q
    EXE = .exe
else
    # macOS settings
    CXX = clang++
    CXXFLAGS = -std=c++17 $(shell pkg-config --cflags opencv4)
    CXXFLAGS += -I$(HOME)/onnxruntime/include
    LDFLAGS = $(shell pkg-config --libs opencv4)
    LDFLAGS += -L$(HOME)/onnxruntime/lib -lonnxruntime
    RM = rm -f
    EXE =
endif

# Source files
COMMON_SRC = csv_util.cpp featureMethods.cpp distanceFunctions.cpp filters.cpp faceDetect.cpp

# Make methods
buildFeatures: buildFeatures.cpp $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) buildFeatures.cpp $(COMMON_SRC) -o buildFeatures$(EXE) $(LDFLAGS)

matchImage: matchImage.cpp $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) matchImage.cpp $(COMMON_SRC) -o matchImage$(EXE) $(LDFLAGS)

readfiles: readfiles.cpp
	$(CXX) $(CXXFLAGS) readfiles.cpp -o readfiles$(EXE) $(LDFLAGS)

all: buildFeatures matchImage

clean:
	$(RM) buildFeatures$(EXE) matchImage$(EXE) readfiles$(EXE) feature.csv

.PHONY: all clean