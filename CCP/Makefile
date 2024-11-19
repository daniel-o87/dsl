CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
INCLUDES = -I. -I/usr/include -I/usr/local/include

OBJDIR = objs
TARGET = run

$(shell mkdir -p $(OBJDIR))

SOURCES = source.cpp operations.cpp
OBJECTS = $(SOURCES:%.cpp=$(OBJDIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET)

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJDIR)/*.o $(TARGET)

.PHONY: all clean
