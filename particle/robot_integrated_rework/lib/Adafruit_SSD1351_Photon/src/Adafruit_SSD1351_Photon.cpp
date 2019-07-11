/***************************************************
  This is a library for the 1.5" & 1.27" 16-bit Color OLEDs
  with SSD1331 driver chip

  Pick one up today in the adafruit shop!
  ------> http://www.adafruit.com/products/1431
  ------> http://www.adafruit.com/products/1673

  These displays use SPI to communicate, 4 or 5 pins are required to
  interface
  Adafruit invests time and resources providing this open source code,
  please support Adafruit and open-source hardware by purchasing
  products from Adafruit!

  Written by Limor Fried/Ladyada for Adafruit Industries.
  BSD license, all text above must be included in any redistribution
  
  ----------------

  Modified by Mike Heininger and Nathan Friedly for compatibility with 
  the Particle Photon & Particle Build IDE.
 ****************************************************/

#include "Adafruit_mfGFX/Adafruit_mfGFX.h"
#include "Adafruit_SSD1351_Photon.h"


#if !defined(PLATFORM_ID)		// Core v0.3.4
#warning "CORE 0.3.4"
  #define pinSetFast(_pin)	PIN_MAP[_pin].gpio_peripheral->BSRR = PIN_MAP[_pin].gpio_pin
  #define pinResetFast(_pin)	PIN_MAP[_pin].gpio_peripheral->BRR = PIN_MAP[_pin].gpio_pin
  #define digitalWriteFast(pin, value)	(value) ? pinSetFast(pin) : pinResetFast(pin)
#endif


/********************************** low level pin interface */

inline void Adafruit_SSD1351::spiwrite(uint8_t c) {
    //Serial.println(c, HEX);

    if (!_sid) {
        SPI.transfer(c);
		// might be able to make this even faster but
		// a delay -is- required
		//delayMicroseconds(1);
        return;
    }

	//Software SPI, MSB first
	for (uint8_t bit = 0; bit < 8; bit++)  {
		//PIN_MAP[_sclk].gpio_peripheral->BRR = PIN_MAP[_sclk].gpio_pin; // Clock Low
		pinResetFast(_sclk);

		if (c & (1 << (7-bit)))		// walk down mask from bit 7 to bit 0
			//PIN_MAP[_sid].gpio_peripheral->BSRR = PIN_MAP[_sid].gpio_pin; // Data High
			pinSetFast(_sid);
		else
			//PIN_MAP[_sid].gpio_peripheral->BRR = PIN_MAP[_sid].gpio_pin; // Data Low
			pinResetFast(_sid);

		//PIN_MAP[_sclk].gpio_peripheral->BSRR = PIN_MAP[_sclk].gpio_pin; // Clock High
		pinSetFast(_sclk);
	}

}


void Adafruit_SSD1351::writeCommand(uint8_t c) {

    digitalWriteFast(_rs, LOW);
    digitalWriteFast(_cs, LOW);

    //Serial.print("C ");
    spiwrite(c);

    digitalWriteFast(_cs, HIGH);
}


void Adafruit_SSD1351::writeData(uint8_t c) {
    digitalWriteFast(_rs, HIGH);
    digitalWriteFast(_cs, LOW);

//    Serial.print("D ");
    spiwrite(c);

    digitalWriteFast(_cs, HIGH);
}

/***********************************/

void Adafruit_SSD1351::goTo(int x, int y) {
  if ((x >= SSD1351WIDTH) || (y >= SSD1351HEIGHT)) return;

  // set x and y coordinate
  writeCommand(SSD1351_CMD_SETCOLUMN);
  writeData(x);
  writeData(SSD1351WIDTH-1);

  writeCommand(SSD1351_CMD_SETROW);
  writeData(y);
  writeData(SSD1351HEIGHT-1);

  writeCommand(SSD1351_CMD_WRITERAM);
}

uint16_t Adafruit_SSD1351::Color565(uint8_t r, uint8_t g, uint8_t b) {
  uint16_t c;
  c = r >> 3;
  c <<= 6;
  c |= g >> 2;
  c <<= 5;
  c |= b >> 3;

  return c;
}

void Adafruit_SSD1351::fillScreen(uint16_t fillcolor) {
  fillRect(0, 0, SSD1351WIDTH, SSD1351HEIGHT, fillcolor);
}

// Draw a filled rectangle with no rotation.
void Adafruit_SSD1351::rawFillRect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t fillcolor) {
  // Bounds check
  if ((x >= SSD1351WIDTH) || (y >= SSD1351HEIGHT))
	return;

  // Y bounds check
  if (y+h > SSD1351HEIGHT)
  {
    h = SSD1351HEIGHT - y - 1;
  }

  // X bounds check
  if (x+w > SSD1351WIDTH)
  {
    w = SSD1351WIDTH - x - 1;
  }

  /*
  Serial.print(x); Serial.print(", ");
  Serial.print(y); Serial.print(", ");
  Serial.print(w); Serial.print(", ");
  Serial.print(h); Serial.println(", ");
*/

  // set location
  writeCommand(SSD1351_CMD_SETCOLUMN);
  writeData(x);
  writeData(x+w-1);
  writeCommand(SSD1351_CMD_SETROW);
  writeData(y);
  writeData(y+h-1);
  // fill!
  writeCommand(SSD1351_CMD_WRITERAM);

  for (uint16_t i=0; i < w*h; i++) {
    writeData(fillcolor >> 8);
    writeData(fillcolor);
  }
}

/**************************************************************************/
/*!
    @brief  Draws a filled rectangle using HW acceleration
*/
/**************************************************************************/
void Adafruit_SSD1351::fillRect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t fillcolor) {
  // Transform x and y based on current rotation.
  switch (getRotation()) {
  case 0:  // No rotation
    rawFillRect(x, y, w, h, fillcolor);
    break;
  case 1:  // Rotated 90 degrees clockwise.
    swap(x, y);
    x = WIDTH - x - h;
    rawFillRect(x, y, h, w, fillcolor);
    break;
  case 2:  // Rotated 180 degrees clockwise.
    x = WIDTH - x - w;
    y = HEIGHT - y - h;
    rawFillRect(x, y, w, h, fillcolor);
    break;
  case 3:  // Rotated 270 degrees clockwise.
    swap(x, y);
    y = HEIGHT - y - w;
    rawFillRect(x, y, h, w, fillcolor);
    break;
  }
}

// Draw a horizontal line ignoring any screen rotation.
void Adafruit_SSD1351::rawFastHLine(int16_t x, int16_t y, int16_t w, uint16_t color) {
  // Bounds check
  if ((x >= SSD1351WIDTH) || (y >= SSD1351HEIGHT))
	return;

  // X bounds check
  if (x+w > SSD1351WIDTH)
  {
    w = SSD1351WIDTH - x - 1;
  }

  if (w < 0) return;

  // set location
  writeCommand(SSD1351_CMD_SETCOLUMN);
  writeData(x);
  writeData(x+w-1);
  writeCommand(SSD1351_CMD_SETROW);
  writeData(y);
  writeData(y);
  // fill!
  writeCommand(SSD1351_CMD_WRITERAM);

  for (uint16_t i=0; i < w; i++) {
    writeData(color >> 8);
    writeData(color);
  }
}

// Draw a vertical line ignoring any screen rotation.
void Adafruit_SSD1351::rawFastVLine(int16_t x, int16_t y, int16_t h, uint16_t color) {
  // Bounds check
  if ((x >= SSD1351WIDTH) || (y >= SSD1351HEIGHT))
  return;

  // X bounds check
  if (y+h > SSD1351HEIGHT)
  {
    h = SSD1351HEIGHT - y - 1;
  }

  if (h < 0) return;

  // set location
  writeCommand(SSD1351_CMD_SETCOLUMN);
  writeData(x);
  writeData(x);
  writeCommand(SSD1351_CMD_SETROW);
  writeData(y);
  writeData(y+h-1);
  // fill!
  writeCommand(SSD1351_CMD_WRITERAM);

  for (uint16_t i=0; i < h; i++) {
    writeData(color >> 8);
    writeData(color);
  }
}

void Adafruit_SSD1351::drawFastVLine(int16_t x, int16_t y, int16_t h, uint16_t color) {
  // Transform x and y based on current rotation.
  switch (getRotation()) {
  case 0:  // No rotation
    rawFastVLine(x, y, h, color);
    break;
  case 1:  // Rotated 90 degrees clockwise.
    swap(x, y);
    x = WIDTH - x - h;
    rawFastHLine(x, y, h, color);
    break;
  case 2:  // Rotated 180 degrees clockwise.
    x = WIDTH - x - 1;
    y = HEIGHT - y - h;
    rawFastVLine(x, y, h, color);
    break;
  case 3:  // Rotated 270 degrees clockwise.
    swap(x, y);
    y = HEIGHT - y - 1;
    rawFastHLine(x, y, h, color);
    break;
  }
}

void Adafruit_SSD1351::drawFastHLine(int16_t x, int16_t y, int16_t w, uint16_t color) {
  // Transform x and y based on current rotation.
  switch (getRotation()) {
  case 0:  // No rotation.
    rawFastHLine(x, y, w, color);
    break;
  case 1:  // Rotated 90 degrees clockwise.
    swap(x, y);
    x = WIDTH - x - 1;
    rawFastVLine(x, y, w, color);
    break;
  case 2:  // Rotated 180 degrees clockwise.
    x = WIDTH - x - w;
    y = HEIGHT - y - 1;
    rawFastHLine(x, y, w, color);
    break;
  case 3:  // Rotated 270 degrees clockwise.
    swap(x, y);
    y = HEIGHT - y - w;
    rawFastVLine(x, y, w, color);
    break;
  }
}

void Adafruit_SSD1351::drawPixel(int16_t x, int16_t y, uint16_t color)
{
  // Transform x and y based on current rotation.
  switch (getRotation()) {
  // Case 0: No rotation
  case 1:  // Rotated 90 degrees clockwise.
    swap(x, y);
    x = WIDTH - x - 1;
    break;
  case 2:  // Rotated 180 degrees clockwise.
    x = WIDTH - x - 1;
    y = HEIGHT - y - 1;
    break;
  case 3:  // Rotated 270 degrees clockwise.
    swap(x, y);
    y = HEIGHT - y - 1;
    break;
  }

  // Bounds check.
  if ((x >= SSD1351WIDTH) || (y >= SSD1351HEIGHT)) return;
  if ((x < 0) || (y < 0)) return;

  goTo(x, y);

  // setup for data
  digitalWriteFast(_rs, HIGH);
  digitalWriteFast(_cs, LOW);

  spiwrite(color >> 8);
  spiwrite(color);

  digitalWriteFast(_cs, HIGH);
}

void Adafruit_SSD1351::begin(void) {
    // set pin directions
    pinMode(_rs, OUTPUT);

    if (_sclk) {
        pinMode(_sclk, OUTPUT);
        pinMode(_sid, OUTPUT);
    } else {
        // using the hardware SPI
        SPI.begin();
		SPI.setClockDivider(SPI_CLOCK_DIV8);	// 72MHz / 8 = 9Mhz
        SPI.setDataMode(SPI_MODE3);
    }

    // Toggle RST low to reset; CS low so it'll listen to us
    pinMode(_cs, OUTPUT);
    digitalWrite(_cs, LOW);

    if (_rst) {
        pinMode(_rst, OUTPUT);
        digitalWrite(_rst, HIGH);
        delay(500);
        digitalWrite(_rst, LOW);
        delay(500);
        digitalWrite(_rst, HIGH);
        delay(500);
    }

    // Initialization Sequence
    writeCommand(SSD1351_CMD_COMMANDLOCK);  // set command lock
    writeData(0x12);
    writeCommand(SSD1351_CMD_COMMANDLOCK);  // set command lock
    writeData(0xB1);

    writeCommand(SSD1351_CMD_DISPLAYOFF);  		// 0xAE

    writeCommand(SSD1351_CMD_CLOCKDIV);  		// 0xB3
    writeCommand(0xF1);  						// 7:4 = Oscillator Frequency, 3:0 = CLK Div Ratio (A[3:0]+1 = 1..16)

    writeCommand(SSD1351_CMD_MUXRATIO);
    writeData(127);

    writeCommand(SSD1351_CMD_SETREMAP);
    writeData(0x74);

    writeCommand(SSD1351_CMD_SETCOLUMN);
    writeData(0x00);
    writeData(0x7F);
    writeCommand(SSD1351_CMD_SETROW);
    writeData(0x00);
    writeData(0x7F);

    writeCommand(SSD1351_CMD_STARTLINE); 		// 0xA1
    if (SSD1351HEIGHT == 96) {
      writeData(96);
    } else {
      writeData(0);
    }


    writeCommand(SSD1351_CMD_DISPLAYOFFSET); 	// 0xA2
    writeData(0x0);

    writeCommand(SSD1351_CMD_SETGPIO);
    writeData(0x00);

    writeCommand(SSD1351_CMD_FUNCTIONSELECT);
    writeData(0x01); // internal (diode drop)
    //writeData(0x01); // external bias

//    writeCommand(SSSD1351_CMD_SETPHASELENGTH);
//    writeData(0x32);

    writeCommand(SSD1351_CMD_PRECHARGE);  		// 0xB1
    writeCommand(0x32);

    writeCommand(SSD1351_CMD_VCOMH);  			// 0xBE
    writeCommand(0x05);

    writeCommand(SSD1351_CMD_NORMALDISPLAY);  	// 0xA6

    writeCommand(SSD1351_CMD_CONTRASTABC);
    writeData(0xC8);
    writeData(0x80);
    writeData(0xC8);

    writeCommand(SSD1351_CMD_CONTRASTMASTER);
    writeData(0x0F);

    writeCommand(SSD1351_CMD_SETVSL );
    writeData(0xA0);
    writeData(0xB5);
    writeData(0x55);

    writeCommand(SSD1351_CMD_PRECHARGE2);
    writeData(0x01);

    writeCommand(SSD1351_CMD_DISPLAYON);		//--turn on oled panel
}

void  Adafruit_SSD1351::invert(boolean v) {
   if (v) {
     writeCommand(SSD1351_CMD_INVERTDISPLAY);
   } else {
     	writeCommand(SSD1351_CMD_NORMALDISPLAY);
   }
 }

/********************************* low level pin initialization */

// Software SPI
Adafruit_SSD1351::Adafruit_SSD1351(uint8_t cs, uint8_t rs, uint8_t sid, uint8_t sclk, uint8_t rst) : Adafruit_GFX(SSD1351WIDTH, SSD1351HEIGHT) {
    _cs = cs;
    _rs = rs;
    _sid = sid;
    _sclk = sclk;
    _rst = rst;
}

// Hardware SPI
Adafruit_SSD1351::Adafruit_SSD1351(uint8_t cs, uint8_t rs,  uint8_t rst) : Adafruit_GFX(SSD1351WIDTH, SSD1351HEIGHT) {
    _cs = cs;
    _rs = rs;
    _sid = 0;
    _sclk = 0;
    _rst = rst;
}
