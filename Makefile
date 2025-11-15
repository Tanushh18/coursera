NVCC = nvcc
SRC = ./src
OUT = ./output

all: main

main:
	$(NVCC) $(SRC)/main.cu $(SRC)/kernels.cu $(SRC)/utils.cu -o process.exe

run:
	./process.exe

clean:
	rm -f process.exe
