# Flocking Sim Aleks Hatfield

- Alignment causes bodies to match velocity of nearby bodies.
- Cohesion causes bodies to seek the center of mass of the nearest clump of bodies.
- Separation prevents bodies from getting too close to each other.

Base code by Yanar.

## Running the simulation

Install SFML (Simple and Fast Multi-Media Library) in order the run the program.
- Ubuntu / Debian-based distros: `sudo apt-get install libsfml-dev`

Then `cd` into the `src` folder and run `make` to produce an executable file called `flock-sim`.

```bash
./flock-sim
```

to run the simulation. Exit with the `esc`.
