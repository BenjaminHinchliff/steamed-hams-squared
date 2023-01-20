# Steamed Hams Squared

A little algorithm that generates a video where each pixel within is itself a frame of the image.

Algorithm applied to [steamed hams](https://youtu.be/r4gHPz87UP0) (from the simpsons)

Inspired by a comment on [this video](https://youtu.be/oL86w9qISNw)

I'm not the only one to do this, so here are some other approaches:

- https://github.com/Luc-mcgrady/Meta-collage
- https://github.com/Jessdevzz/Steamed-Hams-But-With-Steamed-Hams

## Usage

Due to the fact I made this in a day, everything is hard-coded in constants at the
start of the python file. Edit those if you would like to use this for yourself.
Excpect a non-trivial runtime and RAM usage - around 12 minutes and 1 GiB on my system
(11th gen i7) for the steamed hams video.

The scripts depends on OpenCV, numpy, and scipy - a reqirements.txt file is included.

## Notes for Nerds

The only reason this is vaguely performant is because of how amazing numpy and opencv are
for image processing. For the lookup of the images themselves, I used a k-d tree from scipy,
allowing for O(log(n)) average case lookup times. Thanks to these optimizations I didn't
need to go rent server time to get this to finish.
