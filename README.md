# README

This is a personal passion project of mine. It combines my interests in digital art, colour science, computing, and computer graphics. The goal was to visualize the Munsell data as a 3d volume that you could peer into by taking equi-hue, value, or chroma slices. 

I LOVE [Pteromys' Munsell Color Palette](https://pteromys.melonisland.net/munsell/) page, but I was craving to see it as a solid, like something I could just grab out of the screen with my hands. This was the motivation for the project. 

The point cloud is the original real.dat dataset with white and black added, arranged in 3d space. Note that the scale of the vertical axis was chosen arbitrarily and doesn't necessarily preserve perceptual "distance."

For the dense point cloud, I added new points in line with the existing (hue, value) grid,
and then determined their colours by linearly interpolating in Lab colourspace. I clamped the length of each "spoke" to a length also bilinearly interpolated from its neighbours. 

The surface of the mesh is similarly bilerped. When splitting a quad into triangles, I choose the shorter diagonal to hopefully make the surface smoother. I was a little disappointed by how "wrinkly" it looks, especially near the white pole, but I suppose that's the best I can do given the relatively sparse and discrete dataset.

The interior cut surface geometry is constructed in real time. I load in some lookup tables from jsons to help with this. 

I generated a 3D texture to colour the cut surfaces. Then, I project it to the mesh
using cylindrical coordinates, with lets me maintain the full hue resolution in the centre. 

Something I would love to add is:
- a colour swatch picker
- custom panning controls (so you can pan the solid to wherever you want on the screen, but when you drag, it only rotates the solid instead of rotating your whole screen around some point)