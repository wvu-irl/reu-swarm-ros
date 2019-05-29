#pragma once
#include <SFML/Graphics.hpp>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace menuGraphics
{

static int renderLevel = 0;

class Button
{
  private:
	int renderlevel = 0;

	sf::Rect<int> location;
	char *text;

	sf::Color foreNormal, backNormal;
	sf::Color foreHover, backHover;
	sf::Color foreClick, backClick;

	bool hover, click;

	sf::RectangleShape shape;
	sf::Text drawableText;

	sf::Font font;

	void recolor()
	{
		if (this->click)
		{
			shape.setFillColor(backClick);
			drawableText.setFillColor(foreClick);
		}
		else if (this->hover)
		{
			shape.setFillColor(backHover);
			drawableText.setFillColor(foreHover);
		}
		else
		{
			shape.setFillColor(backNormal);
			drawableText.setFillColor(foreNormal);
		}
	}

  public:
	Button(const char *text, sf::Rect<int> bounds)
	{
		this->location = bounds;
		this->text = (char *)malloc(sizeof(char) * strlen(text));
		memcpy(this->text, text, strlen(text) * sizeof(char));
		foreNormal = sf::Color::Black;
		backNormal = sf::Color::White;

		foreHover = sf::Color::Black;
		backHover = sf::Color(127, 127, 127, 255);

		foreClick = sf::Color::White;
		backClick = sf::Color(200, 200, 200, 255);

		font.loadFromFile("strasua.ttf");
		drawableText.setFont(font);
		drawableText.setString(text);
		drawableText.setFillColor(foreNormal);
		drawableText.setCharacterSize(15);
		drawableText.setPosition(sf::Vector2f(bounds.left + bounds.width / 2 - drawableText.getLocalBounds().width / 2,
											  bounds.top + bounds.height / 2 - drawableText.getLocalBounds().height / 2));

		shape.setFillColor(backNormal);
		shape.setPosition(sf::Vector2f(bounds.left, bounds.top));
		shape.setSize(sf::Vector2f(bounds.width, bounds.height));
	}

	void setRenderLevel(int lev)
	{
		this->renderlevel = lev;
	}

	void setHover(int x, int y)
	{
		if (this->renderlevel == renderLevel)
		{
			this->hover = this->location.contains(sf::Vector2i(x, y));
			recolor();
		}
	}

	bool release()
	{
		if (this->renderlevel == renderLevel)
		{
			bool temp = this->click;
			this->click = false;
			recolor();
			return temp;
		}
		return false;
	}

	void doclick(int x, int y)
	{
		if (this->renderlevel == renderLevel)
		{
			this->click = location.contains(sf::Vector2i(x, y));
			recolor();
		}
	}

	void render(sf::RenderWindow *window)
	{
		if (this->renderlevel == renderLevel)
		{
			window->draw(shape);
			window->draw(drawableText);
		}
	}
};

std::vector<Button> butts;

void renderMenu(sf::RenderWindow *window)
{
	for (size_t i = 0; i < butts.size(); i++)
	{
		butts.at(i).render(window);
	}
}

void mouseClick(int x, int y)
{
	for (size_t i = 0; i < butts.size(); i++)
	{
		butts.at(i).doclick(x, y);
	}
}

int mouseRelease()
{
	int id = -1;
	for (size_t i = 0; i < butts.size(); i++)
	{
		if (butts.at(i).release())
		{
			id = i;
		}
	}
	return id;
}

void mouseMove(int x, int y)
{
	for (size_t i = 0; i < butts.size(); i++)
	{
		butts.at(i).setHover(x, y);
	}
}

} // namespace menuGraphics