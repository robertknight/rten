# rten-imageproc

Library for pre and post-processing image data stored in matrices. It includes
functionality for:

- Finding contours of objects in segmentation masks
- Working with axis-aligned and oriented bounding boxes / rectangles
- Simplifying polygons
- Simple drawing of shapes

The genesis of this library was a need in the ocrs OCR engine for a Rust
implementation of a subset of the geometry and image processing functionality
provided by libraries like OpenCV and Shapely in Python.
