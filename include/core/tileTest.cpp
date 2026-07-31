#include "TileGenerator.h"
#include "Image.h"
#include <iostream>

int main() {
    Image<uint8_t> src(2048, 2048, 3);
    uint8_t* data = src.data();
    TileGenerator generator(data, 0, 2048, 2048, 128, 128, 16);
    TileView tile;
    bool res = generator.next(tile);
    if (res) {
        for (size_t i = 0; i < 2048*2048*3; i++) {
            src[i] = static_cast<uint8_t>(42);
        }
        const uint8_t* a = tile.pixel(0,1,1);
        if (*a == 42) {
            std::cout << "test passed";
        }
    }
}