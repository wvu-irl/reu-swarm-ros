# Adafruit SSD1351 Color OLED library

![Particle Photon with OLED Display](http://nfriedly.github.io/Adafruit_SSD1351_Photon/photon-oled.jpg)

Confirmed working with the [1.5 inch OLED](https://www.adafruit.com/products/1431) breakout board. 

Should also work with the [1.27 inch OLED](https://www.adafruit.com/products/684), and possibly the [0.96 inch OLED](https://www.adafruit.com/products/684) breakout boards (although you'll probably have to fork the library and edit the [height and width DEFINE's](https://github.com/nfriedly/Adafruit_SSD1351_Photon/blob/master/src/Adafruit_SSD1351_Photon.h#L29).)

This is based on a collection of work from other tallented developers, 99.9% of the credit goes to them. 

* I started with https://github.com/mikeseeh/photon-demo-oled-sdcard
* Which I think is based on https://github.com/pkourany/Adafruit_SSD1351_Library/
* And then both of those are based on https://github.com/adafruit/Adafruit-SSD1351-library, which is also where the examples came from

## microSD Card is not currently supported

I fiddled with the code for a little while but couldn't convince it to work. 
Some folks apparently have, though, see https://community.particle.io/t/particle-photon-oled-screen-and-sd-card-demo/18145 and https://github.com/mikeseeh/photon-demo-oled-sdcard


## Wiring

For hardware SPI (considerably faster), connect the pins as follows:
    
    Photon  -> OLED
    ----------------------
    3V3     -> 3Vo (3V)
    GND     -> GND (G)
    D6      -> DC
    D5      -> RESET (R)
    
    A5      -> MOSI (SI)
    A3      -> SCK (CL)
    A2      -> OLEDCS (OC)

DC and RESET (R) may be set to any pin, but the example defaults to D5 & D6.

Any combination of pins may be used for software SPI, but the display will update more slowly.

## API documentation

See https://learn.adafruit.com/adafruit-gfx-graphics-library/
