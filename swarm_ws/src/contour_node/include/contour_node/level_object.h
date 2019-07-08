#ifndef LEVELOBJECT_H
#define LEVELOBJECT_H

#include <string>

class levelObject {
public:
    levelObject(void);
    levelObject(std::pair<double, double> _orig, std::string _name);
    levelObject(double _xorg, double _yorig, std::string _name);
    virtual ~levelObject(void);
    
    virtual void nuiManipulate(double _x, double _y, double _z) = 0;
    
    std::pair<double, double> getOrigin(void);
    
    void setName(std::string _s);
    std::string getName(void);
    
    void select(void);
    void deselect(void);
    bool isSelected(void);
    
private:
    std::pair<double, double> origin; // x, y
    std::string name;
    bool selected;
};

#endif /* LEVELOBJECT_H */

